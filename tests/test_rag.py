"""RAG (kaizen.rag): ingest / document understanding / chunking / store / truy xuất / trích bước / explain có trích dẫn.

Không mạng: embedder băm n-gram (HashEmbedder) — không tải model; Chroma persistent trong tmp.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("pymupdf")

from make_sop_pdf import STD_TIME, STEPS, build_sop_pdf  # noqa: E402

from kaizen.core.config import load_config  # noqa: E402
from kaizen.llm.explain import KnowledgeBase, explain  # noqa: E402
from kaizen.llm.providers import get_provider  # noqa: E402
from kaizen.llm.validator import validate  # noqa: E402
from kaizen.rag import (DocumentStore, HashEmbedder, UnsupportedDocument, chunk_document, extract_steps,  # noqa: E402
                        parse_document, to_station)
from kaizen.rag.ingest import Block, ParsedDoc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CHUNK_ID = re.compile(r"^DOC-[0-9a-f]{6}:p\d+:c\d+$")
F01 = {"id": "F01", "type": "wrist_posture", "severity": "low", "rule": "R7", "confidence": "low",
       "impact_s_per_cycle": 0.0, "rank_score": 0.0,
       "evidence": {"metric": "% thời gian góc cổ tay left lệch > 30°", "value": 21.4, "unit": "%", "hand": "left",
                    "timestamps": [], "trials": ["U001_V001"]}}
F02 = {"id": "F02", "type": "extra_pick", "severity": "med", "rule": "R2", "confidence": "med",
       "impact_s_per_cycle": 0.6, "rank_score": 0.36,
       "evidence": {"metric": "số pick zone bin_planet_gears", "value": 4, "unit": "count/cycle", "extra": 1,
                    "zone": "bin_planet_gears", "sop_required": 3, "timestamps": [], "trials": ["U001_V002"]}}


@pytest.fixture
def store(tmp_path, sop_pdf):
    s = DocumentStore(tmp_path / "store", HashEmbedder())
    s.res = s.ingest(sop_pdf)
    return s


def step_name(s: str) -> str:
    return s.split(". Lưu ý an toàn")[0].rstrip(".")


# ---------------------------------------------------------------- 4a ingest / document understanding
def test_pdf_structure_pages_headings_table(sop_pdf):
    d = parse_document(sop_pdf)
    assert d.kind == "pdf" and d.pages == 2 and d.needs_ocr_pages == [] and d.doc_id.startswith("DOC-")
    heads = [(b.level, b.text) for b in d.blocks if b.kind == "heading"]
    assert heads[0][0] == 1 and "BÁNH RĂNG HÀNH TINH" in heads[0][1] and (2, "2. Trình tự lắp ráp") in heads
    tables = [b for b in d.blocks if b.kind == "table"]
    assert len(tables) == 1 and tables[0].page == 2 and tables[0].text.startswith("|Bước|Thời gian chuẩn (s)|Khay|")
    assert len(tables[0].text.splitlines()) == 14                                          # header + --- + 12 dòng
    steps = [b for b in d.blocks if b.text.startswith("Bước ")]
    assert [b.page for b in steps] == [1] * 6 + [2] * 6 and all(b.bbox for b in steps)
    assert not any("|" in b.text for b in d.blocks if b.kind == "text")                      # text trong bảng không lặp


def test_scanned_page_needs_ocr_without_crash(tmp_path):
    p = build_sop_pdf(tmp_path / "scan.pdf", scanned_page=True)
    d = parse_document(p)
    assert d.pages == 3 and d.needs_ocr_pages == [3] and not any(b.page == 3 for b in d.blocks)
    r = DocumentStore(tmp_path / "s", HashEmbedder()).ingest(p)
    assert r.needs_ocr_pages == [3] and r.chunks > 0


def _docx(paragraphs: list[tuple[str, str]], table: list[list[str]]) -> bytes:
    W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = ""
    for style, text in paragraphs:
        if style == "PAGEBREAK":
            body += '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
            continue
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    body += "<w:tbl>" + "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row)
                                + "</w:tr>" for row in table) + "</w:tbl>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<?xml version="1.0"?><w:document {W}><w:body>{body}</w:body></w:document>')
    return buf.getvalue()


def test_docx_md_txt_parsing():
    data = _docx([("Title", "Quy trình siết vít"), ("Heading1", "Chuẩn bị"), ("", "Bước 1: Lấy lục giác."),
                  ("PAGEBREAK", ""), ("Heading2", "Thực hiện"), ("", "Bước 2: Siết 2 vít.")],
                 [["Bước", "Thời gian chuẩn (s)"], ["1", "3"], ["2", "8"]])
    d = parse_document(data=data, filename="wi.docx")
    assert d.title == "Quy trình siết vít" and d.kind == "docx"
    assert [(b.kind, b.level, b.page) for b in d.blocks[:3]] == [("heading", 1, 1), ("heading", 1, 1), ("text", 0, 1)]
    assert d.blocks[3].page == 2 and d.blocks[-1].kind == "table" and "|2|8|" in d.blocks[-1].text.replace(" ", "")
    md = parse_document(REPO / "config" / "kaizen_kb.md")
    assert md.kind == "md" and md.title.startswith("Kaizen Knowledge Base")
    assert any("[KB-13]" in c.text for c in chunk_document(md))
    t = parse_document(data="Đoạn 1\ndòng 2\n\nĐoạn 2\fTrang hai".encode(), filename="a.txt")
    assert [(b.page, b.text) for b in t.blocks] == [(1, "Đoạn 1 dòng 2"), (1, "Đoạn 2"), (2, "Trang hai")]
    with pytest.raises(UnsupportedDocument):
        parse_document(data=b"x", filename="a.xlsx")


# ---------------------------------------------------------------- chunking
def test_chunking_heading_aware_sizes_overlap():
    words = [f"w{i}" for i in range(1500)]
    blocks = [Block("heading", "Mục A", 1, 1), Block("text", "nhỏ " * 40, 1),
              Block("heading", "Mục B", 1, 1), Block("text", " ".join(words), 1),
              Block("heading", "Mục C", 2, 1), Block("text", "trang hai " * 20, 2)]
    doc = ParsedDoc("DOC-abcdef", "T", "t.md", "md", 2, blocks)
    ch = chunk_document(doc)
    assert all(c.n_tokens <= 600 for c in ch) and [c.chunk_id for c in ch] == [c.chunk_id for c in chunk_document(doc)]
    assert all(CHUNK_ID.match(c.chunk_id) for c in ch)
    assert ch[0].section == "Mục A" and "Mục B" in ch[0].text                    # mục A < 300 token → gộp với B
    long = [c for c in ch if "w100" in c.text or "w900" in c.text]
    a, b = ch[0].text.split(), ch[1].text.split()
    assert a[-50:] == b[:50]                                                    # overlap 50 token khi tách khối dài
    assert ch[-1].page == 2 and ch[-1].chunk_id.endswith(":p2:c0") and ch[-1].section == "Mục C"  # sang trang → cắt
    assert len(long) >= 2


# ---------------------------------------------------------------- 4b store / retrieval
def test_ingest_idempotent_and_metadata(store, sop_pdf):
    r = store.res
    assert r.pages == 2 and r.chunks == len(store.get_chunks(r.doc_id)) > 0 and not r.already_present
    again = store.ingest(sop_pdf)
    assert again.already_present and again.doc_id == r.doc_id and store.count() == r.chunks
    for c in store.get_chunks(r.doc_id):
        assert CHUNK_ID.match(c.chunk_id) and c.doc_id == r.doc_id and c.title and c.page in (1, 2)
        assert store.has_chunk(c.chunk_id)
    assert not store.has_chunk(f"{r.doc_id}:p9:c9")
    assert [d["doc_id"] for d in store.list_documents()] == [r.doc_id]


def test_query_planet_gear_returns_correct_page(store):
    store.ingest(REPO / "config" / "kaizen_kb.md")
    hits = store.search("lắp bánh răng hành tinh", k=3)
    top = hits[0]
    assert top.doc_id == store.res.doc_id and top.page == 1 and "bánh răng hành tinh" in top.text
    assert 0 < top.score <= 1 and top.title and top.chunk_id.startswith(store.res.doc_id)
    assert {"chunk_id", "doc_id", "title", "page", "section", "text", "score"} <= set(top.to_dict())
    only = store.search("Kaizen ECRS loại bỏ", k=5, filter={"doc_id": store.res.doc_id})
    assert {c.doc_id for c in only} == {store.res.doc_id}


def test_reindex_when_embedder_changes(store):
    s2 = DocumentStore(store.path, HashEmbedder(dim=256))
    assert s2.count() == store.res.chunks and s2.search("bánh răng hành tinh", 1)[0].page == 1


# ---------------------------------------------------------------- 4c.2 trích bước chuẩn
def test_extract_steps_dry_run_12_in_order(store, tmp_path):
    spec = extract_steps(store.res.doc_id, get_provider("dry-run"), store, out_dir=tmp_path / "ex", log=lambda *_: None)
    assert [s.order for s in spec.steps] == list(range(1, 13))
    assert [s.name for s in spec.steps] == [step_name(s) for s in STEPS]
    assert [s.page for s in spec.steps] == [1] * 6 + [2] * 6
    assert [s.std_time_s for s in spec.steps] == STD_TIME and spec.steps[0].zone_hint == "Khay phải trên"
    assert "găng tay" in spec.steps[7].safety_note
    saved = json.loads((tmp_path / "ex" / f"{store.res.doc_id}.json").read_text())
    assert len(saved["steps"]) == 12 and saved["steps"][2]["page"] == 1


def test_extract_steps_llm_verified_by_code(store, tmp_path, scripted):
    chunks = store.get_chunks(store.res.doc_id)
    p1 = next(c.chunk_id for c in chunks if c.page == 1)
    p2 = next(c.chunk_id for c in chunks if c.page == 2 and c.kind == "text")
    steps = [{"id": i, "name": step_name(s), "order": i, "zone_hint": None, "safety_note": None,
              "std_time_s": STD_TIME[i - 1], "source_chunk_id": p1 if i <= 6 else p2} for i, s in enumerate(STEPS, 1)]
    steps[2]["source_chunk_id"] = "DOC-000000:p7:c0"          # chunk bịa → định vị lại theo tên → trang 1
    steps[4]["std_time_s"] = 99.9                             # số không có trong tài liệu → bỏ
    good = "Kết quả:\n```json\n" + json.dumps({"steps": steps}, ensure_ascii=False) + "\n```"
    fp = scripted(["không phải json", good])
    spec = extract_steps(store.res.doc_id, fp, store, out_dir=None)
    assert len(fp.calls) == 2 and fp.calls[1]["history"][-1]["role"] == "assistant"   # hỏi lại 1 lần
    assert [s.order for s in spec.steps] == list(range(1, 13)) and spec.prompt == "extract_steps_v1"
    assert spec.steps[2].page == 1 and spec.steps[2].source_chunk_id == p1 and spec.steps[2].warnings
    assert spec.steps[4].std_time_s is None and "99.9" in spec.steps[4].warnings[0]
    assert all(s.page for s in spec.steps)
    with pytest.raises(ValueError):
        extract_steps(store.res.doc_id, scripted(["x", '{"steps": [{"name": ""}]}']), store, out_dir=None)


def test_to_station_draft_is_loadable(store, tmp_path):
    spec = extract_steps(store.res.doc_id, get_provider("dry-run"), store, out_dir=None)
    out = to_station(spec, REPO / "config" / "station.yaml", tmp_path / "draft.yaml")
    cfg = load_config(out)
    assert [s["id"] for s in cfg["steps"]] == list(range(1, 13)) and cfg["steps"][0]["std_time_s"] == 4.0
    assert cfg["zones"] == load_config(REPO / "config" / "station.yaml")["zones"]
    txt = out.read_text()
    assert "BẢN NHÁP" in txt and "# trang 1" in txt and f"Σ std_time_s = {sum(STD_TIME):g}" in txt


# ---------------------------------------------------------------- 4c.1 explain có trích dẫn chunk_id
def test_explain_rag_dry_run_cites_chunks_and_tamper_fails(store):
    for f in ("kaizen_kb.md", "sop.md"):
        store.ingest(REPO / "config" / f)
    kb = KnowledgeBase.from_files(REPO / "config" / "kaizen_kb.md", REPO / "config" / "sop.md")
    res = explain([F01, F02], kb, get_provider("dry-run"), retriever=store, k=3)
    assert res["pass_rate"] == 1.0 and res["rag"]["prompt"] == "explain_rag_v1"
    for x in res["explanations"]:
        assert x["final"]["citations"] and all(CHUNK_ID.match(c) for c in x["final"]["citations"])
        assert all(store.has_chunk(c) for c in x["final_check"]["citations_normalized"])
    tampered = dict(res["explanations"][0]["final"], citations=["DOC-ffffff:p9:c9"])
    v = validate(tampered, F01, set(kb.ids), store.has_chunk)
    assert v["status"] == "failed" and v["bad_citations"] == ["DOC-ffffff:p9:c9"]


def test_explain_rag_llm_retry_on_unknown_chunk(store, scripted):
    store.ingest(REPO / "config" / "kaizen_kb.md")
    real = store.search("tư thế cổ tay Muri", 1)[0].chunk_id
    ans = lambda cid: json.dumps([{"finding_id": "F01", "observation": "Tay trái lệch trong 21.4% thời gian.",  # noqa: E731
                                   "hypothesis": "Giả thuyết: khay cao.", "recommendation": "Hạ khay.",
                                   "citations": [cid]}])
    fp = scripted([ans("DOC-123456:p1:c0"), ans(real)])
    res = explain([F01], KnowledgeBase(), fp, retriever=store, k=2)
    x = res["explanations"][0]
    assert x["validation"] == "passed" and len(x["attempts"]) == 2
    assert "citation không tồn tại" in fp.calls[1]["user"] and "<documents>" in fp.calls[0]["user"]
    assert "relevant_chunks" in fp.calls[0]["user"]


def test_cli_07_use_rag_on_existing_run(tmp_path, sop_pdf):
    """Acceptance: 07_explain --use-rag trên run có sẵn → citation chunk_id, validator pass; sửa 1 citation → fail."""
    rid = "20260101-000000_oracle_mediapipe_dry_rag"
    run = tmp_path / "outputs" / "runs" / rid
    (run / "findings").mkdir(parents=True)
    (run / "findings" / "findings.json").write_text(json.dumps({"source": "oracle", "findings": [F01, F02]}))
    (run / "manifest.json").write_text(json.dumps({"run_id": rid}))
    (tmp_path / "outputs" / "runs" / "LATEST").write_text(rid)
    shutil.copytree(REPO / "config", tmp_path / "config")
    env = {**os.environ, "KAIZEN_EMBEDDER": "hash", "KAIZEN_RAG_STORE": str(tmp_path / "store")}
    r = subprocess.run([sys.executable, str(REPO / "src" / "07_explain.py"), "--run", rid, "--use-rag", "--dry-run"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ex = json.loads((run / "explained" / "explained.json").read_text())
    assert ex["pass_rate"] == 1.0 and all(CHUNK_ID.match(c) for x in ex["explanations"] for c in x["final"]["citations"])
    man = json.loads((run / "manifest.json").read_text())
    assert man["stages"]["07_explain"]["prompt"] == "explain_rag_v1" and man["stages"]["07_explain"]["rag"]["chunks"] > 0
    s = DocumentStore(tmp_path / "store", HashEmbedder())
    bad = dict(ex["explanations"][0]["final"], citations=["DOC-ffffff:p1:c0"])
    assert validate(bad, F01, set(), s.has_chunk)["status"] == "failed"
    assert validate(ex["explanations"][0]["final"], F01, set(), s.has_chunk)["status"] == "passed"

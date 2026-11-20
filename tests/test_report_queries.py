"""kaizen.rag.report_queries — TG chuẩn / ngưỡng / trích dẫn cho report, mỗi giá trị từ tài liệu kèm citation;
store rỗng → degrade (KB-xx tĩnh, std từ station.yaml, ngưỡng mặc định). Offline (HashEmbedder)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from kaizen.rag import DocumentStore, HashEmbedder  # noqa: E402
from kaizen.rag import report_queries as rq  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WI_MD = """# Hướng dẫn thao tác chuẩn trạm TST

## 1. Trình tự thao tác chuẩn
Thực hiện đúng thứ tự. Thời gian chuẩn là thời gian tham khảo; chênh lệch kéo dài trên 25% cần được ghi nhận.

| Bước | Tên bước | Khay | TG<br>chuẩn<br>(s) |
|---|---|---|---|
| 1 | Pick up/Place Carrier | K1 | 4.0 |
| 2 | Pick up/Place Gear<br>Bearings (x3) | K2 | 9.5 |

## 3. An toàn & công thái học
Giữ cổ tay ở tư thế trung tính khi ấn vòng bi; không gập cổ tay quá 35° kéo dài.
"""
CFG = {"muri_dev_deg": 30, "steps": [{"id": 1, "name": "Pick up/Place Carrier"},
                                     {"id": 2, "name": "Pick up/Place Gear Bearings (x3)"},
                                     {"id": 3, "name": "Pick up/Place Planet Gears (x3)"}]}
F_WRIST = {"id": "F01", "type": "wrist_posture", "evidence": {"metric": "% thời gian góc cổ tay left lệch > 30°",
                                                              "hand": "left", "trials": ["U001_V001"]}}


@pytest.fixture
def store(tmp_path):
    p = tmp_path / "WI-TST-002_work_instruction.md"
    p.write_text(WI_MD, encoding="utf-8")
    s = DocumentStore(tmp_path / "store", HashEmbedder())
    s.res = s.ingest(p)
    return s


def test_std_times_from_table_with_citation(store):
    st = rq.retrieve_std_times(CFG, store)
    assert st[1].std_s == 4.0 and st[2].std_s == 9.5 and st[3].std_s is None       # bước 3 không có → "—"
    c = st[2].citation
    assert st[2].source == "document" and c.short == "[WI-TST-002 · tr.1]" and c.doc_id == store.res.doc_id
    assert store.has_chunk(c.chunk_id) and "Bearings" in c.excerpt and len(c.excerpt.split()) <= 26
    manual = rq.retrieve_std_times({**CFG, "steps": [dict(CFG["steps"][0], std_time_s=5.0, std_source="IE đo 2026")]},
                                   store)
    assert manual[1].std_s == 5.0 and manual[1].source == "station.yaml" and manual[1].citation.short == "[station.yaml]"


def test_thresholds_from_document_else_default(store):
    t = rq.retrieve_threshold("lệch thời gian chuẩn", CFG, store)
    assert (t.value, t.unit, t.source) == (25.0, "%", "document") and "25%" in t.citation.excerpt
    w = rq.retrieve_threshold("góc cổ tay", CFG, store)
    assert w.value == 35.0 and w.citation.short == "[WI-TST-002 · tr.1]" and "muri_dev_deg = 30" in w.note
    d = rq.retrieve_threshold("lệch thời gian chuẩn", CFG, None)
    assert d.value == 30.0 and d.source == "default" and d.note == "ngưỡng mặc định hệ thống"
    with pytest.raises(KeyError):
        rq.retrieve_threshold("không có", CFG, store)


def test_retrieve_for_finding_and_empty_store_fallback(store):
    cites = rq.retrieve_for_finding(F_WRIST, store)
    assert 0 < len(cites) <= 4 and all(store.has_chunk(c.chunk_id) and c.page == 1 for c in cites)
    assert any("cổ tay" in c.excerpt for c in cites)
    kb = rq.retrieve_for_finding(F_WRIST, None)
    assert [c.short for c in kb] == ["[KB-13]", "[KB-01]"] and all(c.kind == "kb" for c in kb)


def test_resolve_standards_json_and_cache_invalidation(store, tmp_path):
    std = rq.resolve_standards(CFG, store)
    json.dumps(std)                                                       # JSON-able (findings.json["standards"])
    assert std["std_times"]["1"]["citation"]["short"] == "[WI-TST-002 · tr.1]" and std["store"]["documents"] == 1
    back = rq.Standards.from_dict(std)
    assert back.std_times[2].std_s == 9.5 and back.thresholds["wrist_angle_deg"].value == 35.0
    assert rq.retrieve_std_times(CFG, store)[3].std_s is None
    extra = tmp_path / "WI-TST-003.md"
    extra.write_text("| Bước | Tên bước | TG chuẩn (s) |\n|---|---|---|\n| 3 | Pick up/Place Planet Gears (x3) | 8 |\n")
    store.ingest(extra)                                                   # registry đổi → cache theo mtime hết hạn
    assert rq.retrieve_std_times(CFG, store)[3].std_s == 8.0
    empty = rq.resolve_standards(CFG, rq.open_store(tmp_path / "nothing"))
    assert empty["store"] is None and all(v["std_s"] is None for v in empty["std_times"].values())
    assert not (tmp_path / "nothing").exists()                             # store rỗng: không tạo thư mục


def test_annotate_citations_short_labels(store):
    res = {"explanations": [{"final_check": {"citations_normalized": [store.get_chunks(store.res.doc_id)[0].chunk_id,
                                                                      "KB-13"]}}]}
    rq.annotate_citations(res, store, {"KB-13": "[KB-13] Muri — tư thế cổ tay."})
    det = res["explanations"][0]["citations_detail"]
    assert [d["short"] for d in det] == ["[WI-TST-002 · tr.1]", "[KB-13]"] and det[1]["excerpt"].startswith("Muri")
    assert rq.doc_code("KAIZEN-GL-001_guideline.pdf", None, None) == "KAIZEN-GL-001"
    assert rq.doc_code("kaizen_kb.md", "Kaizen KB", "DOC-1") == "kaizen_kb"

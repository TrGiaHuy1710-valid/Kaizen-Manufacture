"""Lớp truy vấn RAG phục vụ 06/07/08: trích dẫn cho finding, thời gian chuẩn từng bước, ngưỡng rule — mọi giá trị
lấy từ tài liệu đều mang Citation (tài liệu, trang, chunk_id) để report truy vết được.

    retrieve_for_finding(finding, store) -> list[Citation]      k=4, câu truy vấn theo type + ngữ cảnh finding
    retrieve_std_times(cfg, store)      -> {step_id: StdTime}   station.yaml (std_time_s) > bảng "TG chuẩn" trong store
    retrieve_threshold(name, cfg, store) -> Threshold           số trong câu tài liệu (vd "trên 30%", "quá 30°")
                                                                 hoặc mặc định cấu hình ("ngưỡng mặc định hệ thống")
Degrade: store None / rỗng → citation KB-xx tĩnh (config/kaizen_kb.md), std chỉ từ station.yaml, ngưỡng mặc định.
Cache theo (store, mtime registry, truy vấn): ingest tài liệu mới → mtime đổi → truy vấn lại.
Hiển thị: Citation.short = "[WI-EGT-001 · tr.1]" (mã tài liệu từ tên file / tiêu đề), chunk_id đầy đủ ở bảng tham chiếu.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from kaizen.llm.explain import finding_query
from kaizen.llm.prompts import DRY_CITES
from kaizen.rag.store import count_chunks_fast, default_store_path

if TYPE_CHECKING:
    from kaizen.rag.store import DocumentStore

KB_TITLE = "Kaizen Knowledge Base (config/kaizen_kb.md)"
SOP_TITLE = "SOP trạm (config/sop.md)"
DOC_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,8}(?:-[A-Z0-9]{2,8}){1,3})(?![A-Za-z0-9])")
EXCERPT_WORDS = 25
_CACHE: dict[tuple, Any] = {}
_STORES: dict[tuple, "DocumentStore"] = {}


@dataclass
class Citation:
    """1 nguồn trích dẫn. kind: chunk (RAG store) | kb (KB-xx/SOP-xx tĩnh) | manual (station.yaml) | default."""
    id: str                                   # chunk_id | "KB-13" | "station.yaml" | "default"
    kind: str = "chunk"
    doc_id: Optional[str] = None
    doc_title: str = ""
    code: str = ""                            # mã ngắn tài liệu: WI-EGT-001 / KB-13 / station.yaml
    page: Optional[int] = None
    score: Optional[float] = None
    excerpt: str = ""

    @property
    def chunk_id(self) -> Optional[str]:
        """chunk_id đầy đủ (None nếu không phải chunk RAG)."""
        return self.id if self.kind == "chunk" else None

    @property
    def short(self) -> str:
        """Nhãn ngắn trong văn bản: [WI-EGT-001 · tr.1] / [KB-13] / [station.yaml]."""
        return f"[{self.code} · tr.{self.page}]" if self.kind == "chunk" and self.page else f"[{self.code or self.id}]"

    def to_dict(self) -> dict:
        """dict JSON-able (kèm short)."""
        return {**asdict(self), "short": self.short}

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["Citation"]:
        """Ngược của to_dict (bỏ key thừa)."""
        if not d:
            return None
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class StdTime:
    """Thời gian chuẩn 1 bước + nguồn (station.yaml ghi tay hoặc chunk bảng trong tài liệu)."""
    std_s: Optional[float]
    citation: Optional[Citation] = None
    source: str = "none"                      # station.yaml | document | none

    def to_dict(self) -> dict:
        """dict JSON-able."""
        return {"std_s": self.std_s, "source": self.source,
                "citation": self.citation.to_dict() if self.citation else None}


@dataclass
class Threshold:
    """Ngưỡng dùng trong rule: từ tài liệu (kèm citation) hoặc mặc định cấu hình."""
    key: str
    name: str
    value: float
    unit: str
    source: str = "default"                   # document | default
    citation: Optional[Citation] = None
    default_value: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        """dict JSON-able."""
        return {**{k: v for k, v in asdict(self).items() if k != "citation"},
                "citation": self.citation.to_dict() if self.citation else None}


@dataclass(frozen=True)
class ThresholdSpec:
    """Cách tìm 1 ngưỡng trong tài liệu: truy vấn + mẫu số + từ khoá bắt buộc trong cùng câu."""
    key: str
    name: str
    query: str
    unit: str
    pattern: str
    must: tuple[str, ...]
    any_of: tuple[str, ...]
    cfg_key: str
    default: float


THRESHOLDS: dict[str, ThresholdSpec] = {s.key: s for s in [
    ThresholdSpec("std_dev_pct", "lệch thời gian chuẩn", "thời gian chuẩn chênh lệch kéo dài trên phần trăm cần ghi nhận",
                  "%", r"(\d+(?:[.,]\d+)?)\s*%", ("chuẩn",), ("lệch", "chênh", "vượt", "deviation"),
                  "std_dev_threshold_pct", 30.0),
    ThresholdSpec("wrist_angle_deg", "góc cổ tay", "tư thế cổ tay trung tính không gập quá độ kéo dài",
                  "°", r"(\d+(?:[.,]\d+)?)\s*(?:°|độ)", ("cổ tay",), ("gập", "lệch", "nghiêng", "duỗi", "bẻ"),
                  "muri_dev_deg", 30.0),
]}
ALIASES = {"lệch thời gian chuẩn": "std_dev_pct", "thời gian chuẩn": "std_dev_pct", "góc cổ tay": "wrist_angle_deg",
           "cổ tay": "wrist_angle_deg"}


# ---------------------------------------------------------------- store + cache
def open_store(path: str | Path | None = None, embedder: str | None = None) -> Optional["DocumentStore"]:
    """DocumentStore nếu store có chunk, ngược lại None (không tạo thư mục, không nạp model khi rỗng)."""
    p = Path(path or default_store_path())
    if count_chunks_fast(p) == 0:
        return None
    key = (str(p.resolve()), embedder)
    if key not in _STORES:
        from kaizen.rag.store import DocumentStore
        _STORES[key] = DocumentStore(p, embedder)
    return _STORES[key]


def _cached(store: Optional["DocumentStore"], kind: str, key: tuple, fn):
    k = (str(store.path.resolve()) if store else None, store.mtime() if store else 0.0, kind, key)
    if k not in _CACHE:
        _CACHE[k] = fn()
    return _CACHE[k]


def clear_cache() -> None:
    """Xoá cache truy vấn (test)."""
    _CACHE.clear()
    _STORES.clear()


# ---------------------------------------------------------------- citation helpers
def doc_code(filename: str | None, title: str | None, doc_id: str | None) -> str:
    """Mã ngắn tài liệu: mã dạng WI-EGT-001 trong tên file / tiêu đề, không có → tên file (bỏ đuôi) / doc_id."""
    for s in (Path(filename).stem if filename else "", title or ""):
        m = DOC_CODE_RE.search(s)
        if m and any(ch.isdigit() for ch in m.group(1)):
            return m.group(1)
    return Path(filename).stem if filename else (doc_id or "?")


def _words(s: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", s.lower()))


def clean_cell(s: str) -> str:
    """Ô bảng markdown → text 1 dòng (bỏ <br>, **)."""
    return " ".join(str(s).replace("<br>", " ").replace("**", "").split())


def excerpt(text: str, query: str = "", n: int = EXCERPT_WORDS) -> str:
    """Câu khớp truy vấn nhất (theo số từ chung), cắt ≤ n từ."""
    sents = [x for ln in text.replace("|", " ").split("\n") for x in re.split(r"(?<=[.!?;])\s+", clean_cell(ln))
             if x.strip()] or [clean_cell(text)]
    q = set(_words(query))
    best = max(sents, key=lambda x: len(q & set(_words(x)))) if q else sents[0]
    w = best.split()
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def chunk_citation(store: "DocumentStore", chunk: Any, query: str = "", text: str | None = None) -> Citation:
    """Citation cho 1 chunk của store (mã tài liệu từ registry)."""
    info = store.get_document(chunk.doc_id) or {}
    return Citation(id=chunk.chunk_id, kind="chunk", doc_id=chunk.doc_id, doc_title=info.get("title") or chunk.title,
                    code=doc_code(info.get("filename"), info.get("title") or chunk.title, chunk.doc_id),
                    page=int(chunk.page), score=round(float(chunk.score), 4) if getattr(chunk, "score", None) else None,
                    excerpt=excerpt(text if text is not None else chunk.text, query))


def kb_citation(cid: str, kb_ids: dict[str, str] | None = None) -> Citation:
    """Citation tĩnh KB-xx / SOP-xx (excerpt từ đoạn KB nếu có)."""
    title = SOP_TITLE if cid.startswith("SOP") else KB_TITLE
    txt = (kb_ids or {}).get(cid, "")
    return Citation(id=cid, kind="kb", doc_title=title, code=cid,
                    excerpt=excerpt(re.sub(r"^\[[A-Z]+-\d+\]\s*", "", txt)) if txt else "")


def citation_for_id(cid: str, store: Optional["DocumentStore"], kb_ids: dict[str, str] | None = None) -> Citation:
    """Citation đầy đủ từ id đã chuẩn hoá (chunk_id → tra store; KB-xx → KB tĩnh)."""
    if cid.startswith("DOC-") and store is not None:
        c = store.get_chunk(cid)
        if c is not None:
            return chunk_citation(store, c)
    if cid.startswith("DOC-"):
        return Citation(id=cid, kind="chunk", code=cid.split(":")[0], doc_id=cid.split(":")[0])
    return kb_citation(cid, kb_ids)


# ---------------------------------------------------------------- Task 1: 3 truy vấn
def retrieve_for_finding(finding: dict, store: Optional["DocumentStore"], k: int = 4,
                         kb_ids: dict[str, str] | None = None) -> list[Citation]:
    """Top-k Citation cho 1 finding (store rỗng → KB-xx tĩnh theo type)."""
    q = finding_query(finding)
    if store is None:
        return [kb_citation(c, kb_ids) for c in DRY_CITES.get(finding.get("type", ""), ["KB-01"])]
    hits = _cached(store, "finding", (q, k), lambda: store.search(q, k))
    return [chunk_citation(store, c, q) for c in hits]


def _norm_name(s: str) -> str:
    return " ".join(_words(clean_cell(s)))


def _time_col(head: list[str]) -> int | None:
    comp = [re.sub(r"\s+", "", clean_cell(h)).lower() for h in head]
    for keys in (("tgchuẩn", "thờigianchuẩn", "stdtime", "standardtime", "tgc"), ("(s)", "time", "thờigian")):
        i = next((i for i, h in enumerate(comp) if any(x in h for x in keys)), None)
        if i is not None:
            return i
    return None


def _num(s: str) -> float | None:
    m = re.search(r"(?<![\w.])\d+(?:[.,]\d+)?", clean_cell(s))
    return float(m.group(0).replace(",", ".")) if m else None


def _std_from_tables(store: "DocumentStore", names: dict[int, str]) -> dict[int, StdTime]:
    """Tên bước → dòng bảng có ô trùng tên (chuẩn hoá) → số ở cột 'TG chuẩn'. Ưu tiên chunk bảng do search trả về."""
    from kaizen.rag.extract import table_rows
    every = [c for d in store.list_documents() for c in store.get_chunks(d["doc_id"]) if c.kind == "table"]
    out: dict[int, StdTime] = {}
    for sid, name in names.items():
        hits = [c for c in store.search(name, 6) if c.kind == "table"]
        seen = {c.chunk_id for c in hits}
        for c in hits + [c for c in every if c.chunk_id not in seen]:
            head, rows = table_rows(c.text)
            ti = _time_col(head)
            if ti is None:
                continue
            row = next((r for r in rows if any(_norm_name(cell) == _norm_name(name) for cell in r)), None)
            if row is None or ti >= len(row) or _num(row[ti]) is None:
                continue
            cit = chunk_citation(store, c, name, text=" · ".join(clean_cell(x) for x in row))
            out[sid] = StdTime(_num(row[ti]), cit, "document")
            break
    return out


def retrieve_std_times(cfg: dict, store: Optional["DocumentStore"]) -> dict[int, StdTime]:
    """{step_id: StdTime}: station.yaml std_time_s (nguồn ghi tay) > bảng TG chuẩn trong store > None ('—')."""
    steps = {int(s["id"]): s for s in cfg.get("steps", [])}
    out: dict[int, StdTime] = {}
    for sid, s in steps.items():
        if s.get("std_time_s") is not None:
            src = s.get("std_source") or "station.yaml (ghi tay)"
            out[sid] = StdTime(float(s["std_time_s"]), Citation(id="station.yaml", kind="manual", doc_title=src,
                                                                code="station.yaml", excerpt=str(src)), "station.yaml")
    todo = {sid: s.get("name", "") for sid, s in steps.items() if sid not in out and s.get("name")}
    if store is not None and todo:
        found = _cached(store, "std", tuple(sorted(todo.items())), lambda: _std_from_tables(store, todo))
        out.update(found)
    for sid in steps:
        out.setdefault(sid, StdTime(None))
    return dict(sorted(out.items()))


def _sentences(text: str) -> list[str]:
    return [x for ln in text.replace("|", " ").split("\n") for x in re.split(r"(?<=[.!?])\s+", clean_cell(ln))
            if x.strip()]


def _find_threshold(store: "DocumentStore", spec: ThresholdSpec) -> Optional[tuple[float, Any, str]]:
    hits = store.search(spec.query, 6)
    seen = {c.chunk_id for c in hits}
    rest = [c for d in store.list_documents() for c in store.get_chunks(d["doc_id"]) if c.chunk_id not in seen]
    for c in hits + rest:
        for sent in _sentences(c.text):
            low = sent.lower()
            if not all(m in low for m in spec.must) or not any(a in low for a in spec.any_of):
                continue
            m = re.search(spec.pattern, sent)
            if m:
                return float(m.group(1).replace(",", ".")), c, sent
    return None


def retrieve_threshold(name: str, cfg: dict | None = None, store: Optional["DocumentStore"] = None) -> Threshold:
    """Ngưỡng theo tên ("lệch thời gian chuẩn", "góc cổ tay" hoặc key) — tài liệu (kèm citation) > mặc định cấu hình."""
    key = ALIASES.get(name, name)
    if key not in THRESHOLDS:
        raise KeyError(f"ngưỡng không hỗ trợ: {name!r} (có: {sorted(THRESHOLDS)})")
    spec, cfg = THRESHOLDS[key], cfg or {}
    default = float(cfg.get(spec.cfg_key, spec.default))
    found = _cached(store, "threshold", (key,), lambda: _find_threshold(store, spec)) if store is not None else None
    if found:
        v, chunk, sent = found
        note = "" if v == default else f"tài liệu ghi {v:g}{spec.unit}, cấu hình {spec.cfg_key} = {default:g}"
        cit = chunk_citation(store, chunk, query=f"{v:g}", text=sent)
        return Threshold(key, spec.name, v, spec.unit, "document", cit, default, note)
    return Threshold(key, spec.name, default, spec.unit, "default",
                     Citation(id="default", kind="default", doc_title="ngưỡng mặc định hệ thống",
                              code=f"mặc định {spec.cfg_key}"), default, "ngưỡng mặc định hệ thống")


def resolve_standards(cfg: dict, store: Optional["DocumentStore"]) -> dict:
    """Gói chuẩn dùng cho 06/08: {std_times, thresholds, store} (JSON-able, ghi vào findings.json["standards"])."""
    return {"std_times": {str(k): v.to_dict() for k, v in retrieve_std_times(cfg, store).items()},
            "thresholds": {k: retrieve_threshold(k, cfg, store).to_dict() for k in THRESHOLDS},
            "store": store.describe() if store is not None else None}


@dataclass
class Standards:
    """Đọc lại gói chuẩn (findings.json["standards"]) cho 08 / R8."""
    std_times: dict[int, StdTime] = field(default_factory=dict)
    thresholds: dict[str, Threshold] = field(default_factory=dict)
    store: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "Standards":
        """Từ dict JSON (None → rỗng)."""
        d = d or {}
        st = {int(k): StdTime(v.get("std_s"), Citation.from_dict(v.get("citation")), v.get("source", "none"))
              for k, v in (d.get("std_times") or {}).items()}
        th = {}
        for k, v in (d.get("thresholds") or {}).items():
            th[k] = Threshold(**{**{f: v.get(f) for f in Threshold.__dataclass_fields__ if f != "citation"},
                                 "citation": Citation.from_dict(v.get("citation"))})
        return cls(st, th, d.get("store"))


def annotate_citations(res: dict, store: Optional["DocumentStore"], kb_ids: dict[str, str] | None = None) -> dict:
    """explained.json: thêm "citations_detail" (Citation đầy đủ: tài liệu, trang, chunk_id, short) cho mỗi explanation
    → 08 render [WI-EGT-001 · tr.x] không phải truy vấn lại. Sửa tại chỗ và trả lại res."""
    for x in res.get("explanations", []):
        ids = (x.get("final_check") or {}).get("citations_normalized") or []
        x["citations_detail"] = [citation_for_id(c, store, kb_ids).to_dict() for c in ids]
    return res

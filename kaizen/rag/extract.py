"""Trích BƯỚC CHUẨN từ tài liệu SOP → StepsSpec → bản nháp station.yaml cho kỹ sư sửa.

    extract_steps(doc_id, provider, store) -> StepsSpec   (lưu outputs/extractions/<doc_id>.json)
    to_station(spec, base_config) -> Path                  (outputs/extractions/<doc_id>_station.yaml)

Provider thật: LLM đọc toàn bộ chunk (theo thứ tự, kèm chunk_id), trả JSON {steps: [...]} (prompt EXTRACT_STEPS_V1),
validate bằng pydantic; parse/validate hỏng → hỏi lại 1 lần.
dry-run: trích bằng code (dòng "Bước N: …" / "Step N: …" + đọc bảng có cột bước/thời gian/khay) — không mạng.
Hậu kiểm bằng CODE cho mọi provider (LLM không được bịa):
  - source_chunk_id phải thuộc tài liệu; sai/thiếu → định vị lại bằng so khớp tên bước với nội dung chunk.
  - page = trang của chunk nguồn.
  - std_time_s phải xuất hiện (dạng số) trong văn bản tài liệu; không có → bỏ (None) + cảnh báo.
  - order phải là 1..n không trùng; sai → đánh số lại theo thứ tự trả về + cảnh báo.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from kaizen.llm.explain import render_documents
from kaizen.llm.prompts import EXTRACT_STEPS_V1, ONLY_JSON_V1
from kaizen.llm.providers import LLMProvider
from kaizen.llm.validator import normalize_citation
from kaizen.rag.chunking import Chunk

if TYPE_CHECKING:
    from kaizen.rag.store import DocumentStore

OUT_DIR = Path("outputs/extractions")
# 1 đoạn = 1 dòng trong chunk (parser đã nối các dòng xuống dòng mềm của PDF) → tên bước = phần còn lại của dòng
STEP_RE = re.compile(r"^[ \t]*(?:Bước|Buoc|Step|STEP|BƯỚC)[ \t]*(\d{1,3})[ \t]*[:.)\-–—][ \t]*(.+)$", re.M)
SAFETY_RE = re.compile(r"\s*(?:Lưu ý an toàn|An toàn|Safety|Cảnh báo)\s*:\s*(.+)$", re.I | re.S)
NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


class ExtractedStep(BaseModel):
    """1 bước chuẩn trích từ tài liệu."""
    id: int
    name: str
    order: int
    zone_hint: Optional[str] = None
    std_time_s: Optional[float] = None
    safety_note: Optional[str] = None
    source_chunk_id: Optional[str] = None
    page: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        v = " ".join(str(v).split())
        if not v:
            raise ValueError("name rỗng")
        return v


class StepsSpec(BaseModel):
    """Kết quả trích bước của 1 tài liệu (outputs/extractions/<doc_id>.json)."""
    doc_id: str
    title: str
    provider: str
    model: Optional[str] = None
    prompt: Optional[str] = None
    extracted_at: str
    steps: list[ExtractedStep]
    warnings: list[str] = Field(default_factory=list)


class _LLMSteps(BaseModel):
    steps: list[ExtractedStep]


# ---------------------------------------------------------------- dry-run: trích bằng code
def _table_rows(md: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.strip() for ln in md.split("\n") if ln.strip().startswith("|")]
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
    return (cells[0], cells[1:]) if cells else ([], [])


def _num(s: str) -> float | None:
    m = NUM_RE.search(s or "")
    return float(m.group(0).replace(",", ".")) if m else None


def table_step_info(chunks: list[Chunk]) -> dict[int, dict]:
    """Đọc bảng có cột bước (+ thời gian chuẩn / khay) → {step: {std_time_s, zone_hint}}."""
    out: dict[int, dict] = {}
    for c in chunks:
        if c.kind != "table":
            continue
        head, rows = _table_rows(c.text)
        low = [h.lower() for h in head]
        i_step = next((i for i, h in enumerate(low) if re.search(r"bước|step|stt", h)), None)
        i_time = next((i for i, h in enumerate(low) if re.search(r"thời gian|time|std|\(s\)", h)), None)
        i_zone = next((i for i, h in enumerate(low) if re.search(r"khay|zone|vị trí|bin|tray", h)), None)
        if i_step is None:
            continue
        for r in rows:
            n = _num(r[i_step]) if i_step < len(r) else None
            if n is None:
                continue
            info = out.setdefault(int(n), {})
            if i_time is not None and i_time < len(r) and _num(r[i_time]) is not None:
                info["std_time_s"] = _num(r[i_time])
            if i_zone is not None and i_zone < len(r) and r[i_zone]:
                info["zone_hint"] = r[i_zone]
    return out


def rule_based_steps(chunks: list[Chunk]) -> list[dict]:
    """Dòng 'Bước N: …' theo thứ tự xuất hiện (bỏ trùng số) + thông tin bảng."""
    found: dict[int, dict] = {}
    for c in chunks:
        if c.kind == "table":
            continue
        for m in STEP_RE.finditer(c.text):
            n = int(m.group(1))
            if n in found:
                continue
            name = " ".join(m.group(2).split())
            safety = None
            sm = SAFETY_RE.search(name)
            if sm:
                safety, name = sm.group(1).strip(), name[: sm.start()].strip()
            found[n] = {"id": n, "name": name.rstrip("."), "safety_note": safety, "source_chunk_id": c.chunk_id}
    tbl = table_step_info(chunks)
    steps = []
    for k, n in enumerate(sorted(found), 1):
        s = {**found[n], "order": k, **tbl.get(n, {})}
        steps.append(s)
    return steps


# ---------------------------------------------------------------- LLM
def _ask_llm(provider: LLMProvider, chunks: list[Chunk], title: str) -> tuple[list[dict], list[str]]:
    user = EXTRACT_STEPS_V1.render(documents=render_documents(chunks), title=title)
    warns: list[str] = []
    res = provider.complete(EXTRACT_STEPS_V1.system, user, json_mode=True, max_tokens=8000)
    for attempt in (1, 2):
        data = res.parsed
        if isinstance(data, list):
            data = {"steps": data}
        try:
            if not isinstance(data, dict):
                raise ValueError(f"không parse được JSON ({res.parse_error})")
            return [s.model_dump() for s in _LLMSteps.model_validate(data).steps], warns
        except (ValidationError, ValueError) as e:
            msg = str(e).splitlines()[0]
            warns.append(f"lần {attempt}: response không hợp lệ ({msg})")
            if attempt == 2:
                raise ValueError(f"LLM không trả được danh sách bước hợp lệ: {msg}") from e
            res = provider.complete(EXTRACT_STEPS_V1.system,
                                    ONLY_JSON_V1.user.replace("JSON array", 'JSON object {"steps": [...]}')
                                    + f" Lỗi: {msg}", json_mode=True, max_tokens=8000,
                                    history=[{"role": "user", "content": user},
                                             {"role": "assistant", "content": res.text}])
    return [], warns


# ---------------------------------------------------------------- hậu kiểm bằng code
def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _locate(name: str, chunks: list[Chunk]) -> Chunk | None:
    want = _tokens(name)
    best, score = None, 0.0
    for c in chunks:
        if c.kind == "table":
            continue
        ov = len(want & _tokens(c.text)) / (len(want) or 1)
        if ov > score:
            best, score = c, ov
    return best if score >= 0.5 else None


def verify_steps(raw: list[dict], chunks: list[Chunk]) -> tuple[list[ExtractedStep], list[str]]:
    """Kiểm nguồn trang + số thời gian chuẩn + thứ tự bằng code (xem docstring module)."""
    by_id = {c.chunk_id: c for c in chunks}
    doc_nums = {round(float(x.replace(",", ".")), 1) for c in chunks for x in NUM_RE.findall(c.text)}
    warns: list[str] = []
    steps: list[ExtractedStep] = []
    for s in raw:
        st = ExtractedStep.model_validate(s)
        cid = normalize_citation(st.source_chunk_id) if st.source_chunk_id else None
        if cid not in by_id:
            c = _locate(st.name, chunks)
            if st.source_chunk_id:
                st.warnings.append(f"source_chunk_id {st.source_chunk_id!r} không thuộc tài liệu → định vị lại theo tên")
            cid = c.chunk_id if c else None
        st.source_chunk_id = cid
        st.page = by_id[cid].page if cid else None
        if st.page is None:
            st.warnings.append("không xác định được trang nguồn")
        if st.std_time_s is not None and round(float(st.std_time_s), 1) not in doc_nums:
            st.warnings.append(f"std_time_s={st.std_time_s:g} không có trong tài liệu → bỏ")
            st.std_time_s = None
        steps.append(st)
    orders = [s.order for s in steps]
    if sorted(orders) != list(range(1, len(steps) + 1)):
        warns.append(f"order không phải 1..{len(steps)} ({orders}) → đánh số lại theo thứ tự trả về")
        for k, s in enumerate(steps, 1):
            s.order = k
    steps.sort(key=lambda s: s.order)
    return steps, warns


def extract_steps(doc_id: str, provider: LLMProvider, store: "DocumentStore", *,
                  out_dir: str | Path | None = OUT_DIR, log: Callable[[str], None] = print) -> StepsSpec:
    """Trích bước chuẩn của doc_id trong store; lưu <out_dir>/<doc_id>.json (out_dir=None → không lưu)."""
    info = store.get_document(doc_id)
    chunks = store.get_chunks(doc_id)
    if info is None or not chunks:
        raise KeyError(f"không có tài liệu {doc_id} trong store {store.path}")
    if provider.name == "dry-run":
        raw, warns, prompt = rule_based_steps(chunks), ["dry-run: trích bằng code (regex 'Bước N' + bảng)"], None
    else:
        raw, warns = _ask_llm(provider, chunks, info["title"])
        prompt = EXTRACT_STEPS_V1.id
    steps, vwarn = verify_steps(raw, chunks)
    spec = StepsSpec(doc_id=doc_id, title=info["title"], provider=provider.name, model=provider.model, prompt=prompt,
                     extracted_at=dt.datetime.now().isoformat(timespec="seconds"), steps=steps,
                     warnings=warns + vwarn)
    if out_dir is not None:
        p = Path(out_dir) / f"{doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[rag] {len(steps)} bước → {p}")
    return spec


def load_extraction(doc_id: str, out_dir: str | Path = OUT_DIR) -> StepsSpec:
    """Đọc outputs/extractions/<doc_id>.json."""
    return StepsSpec.model_validate_json(Path(out_dir, f"{doc_id}.json").read_text(encoding="utf-8"))


def to_station(spec: StepsSpec, base_config: str | Path = "config/station.yaml", out: str | Path | None = None) -> Path:
    """Bản nháp station.yaml: giữ cấu hình chung của base_config, thay `steps` bằng bước trích từ tài liệu."""
    base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8")) if Path(base_config).exists() else {}
    base = {k: v for k, v in (base or {}).items() if k not in ("steps",)}
    out = Path(out or OUT_DIR / f"{spec.doc_id}_station.yaml")
    times = [s.std_time_s for s in spec.steps]
    lines = [f"# BẢN NHÁP station.yaml sinh từ tài liệu \"{spec.title}\" ({spec.doc_id}) — provider {spec.provider}",
             f"# lúc {spec.extracted_at}. KỸ SƯ IE CẦN DUYỆT: zone (khay polygon trong `zones`), reps (số lần lặp), va.",
             "# Mỗi bước kèm trang nguồn + gợi ý vị trí/an toàn trong tài liệu ở comment.",
             "# Cấu hình chung (fps, ngưỡng, zones) chép từ " + str(base_config) + "."]
    if times and all(t is not None for t in times):
        lines.append(f"# Σ std_time_s = {sum(times):g} s (code cộng từ tài liệu) — gợi ý cho cycle_time_target_s.")
    lines += [yaml.safe_dump(base, allow_unicode=True, sort_keys=False, width=120).rstrip(), "", "steps:"]
    for s in spec.steps:
        d = {"id": s.order, "name": s.name, "zone": None, "reps": 1, "va": True}
        if s.std_time_s is not None:
            d["std_time_s"] = s.std_time_s
        flow = yaml.safe_dump(d, allow_unicode=True, default_flow_style=True, sort_keys=False, width=1000).strip()
        notes = [f"trang {s.page}" if s.page else "trang ?"]
        if s.zone_hint:
            notes.append(f"vị trí: {s.zone_hint}")
        if s.safety_note:
            notes.append(f"an toàn: {s.safety_note}")
        lines.append(f"  - {flow}   # {' · '.join(notes)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

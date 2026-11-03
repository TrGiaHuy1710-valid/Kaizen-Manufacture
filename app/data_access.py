"""Lớp DUY NHẤT UI dùng để đọc artifact / gọi backend (kaizen.*). UI không mở file, không tính chỉ số nào —
kể cả 3M tổng hợp (chỉ gọi kaizen.analyzer.aggregate.aggregate_3m).

Mọi hàm chịu được dữ liệu thiếu: trả None / [] / dict rỗng để UI hiện thông báo ngắn thay vì traceback.
Cache: st.cache_data theo (đường dẫn, mtime) — artifact đổi là đọc lại; RAG store giữ bằng st.cache_resource.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kaizen.analyzer import aggregate as _agg  # noqa: E402
from kaizen.analyzer.report import (References, finding_sources, problem_phrase, standards_view,  # noqa: E402
                                    std_rows)
from kaizen.core import pipeline as _pipe  # noqa: E402
from kaizen.core import projects as _proj  # noqa: E402
from kaizen.core.config import DEFAULT_CONFIG, DEFAULT_DATA_ROOT, load_config  # noqa: E402
from kaizen.skeleton import ha4m_io  # noqa: E402
from kaizen.skeleton.overlay import SOURCE_LABEL, step_color  # noqa: E402

STATE_VI = _proj.STATE_VI
STEPS = _pipe.STEPS
BENCH_NOTE = _proj.BENCH_NOTE
MODEL_LINE = _proj.MODEL_LINE
POSE_MODEL_LABEL = _proj.POSE_MODEL_LABEL
JobBusy = _proj.JobBusy
ProjectError = _proj.ProjectError
SOURCE_BADGE = {"oracle": "🟢 oracle (nhãn chuẩn)", "pred": "🟠 pred (dự đoán)", "mixed": "🟡 trộn oracle + pred"}
color_of_step = step_color


def source_badge(src: str | None) -> str:
    """Badge nguồn bước (🟢 oracle / 🟠 pred)."""
    return SOURCE_BADGE.get(src or "", "⚪ chưa có")


def _mt(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, TypeError):
        return -1.0


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float):
    if mtime < 0:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_json(p: Path | None):
    return _json(str(p), _mt(p)) if p else None


# ---------------------------------------------------------------- demo mode
def demo_default() -> bool:
    """Demo mode mặc định bật khi env KAIZEN_DEMO=1."""
    return os.environ.get("KAIZEN_DEMO", "").strip() in ("1", "true", "yes")


def demo_speed() -> float:
    """Hệ số thời lượng tiến độ mô phỏng (env KAIZEN_DEMO_SPEED, mặc định 1; test đặt 0)."""
    try:
        return max(0.0, float(os.environ.get("KAIZEN_DEMO_SPEED", "1")))
    except ValueError:
        return 1.0


# ---------------------------------------------------------------- dự án
def list_projects() -> list[dict]:
    """Danh sách dự án + trạng thái + tóm tắt LATEST run + thumbnail (data URI)."""
    try:
        out = _proj.list_projects()
    except Exception:
        return []
    for p in out:
        p["thumb_uri"] = _proj.thumb_data_uri(p.get("thumb"))
    return out


def get_project(pid: str | None) -> dict | None:
    """project.yaml + status (None nếu không có)."""
    if not pid:
        return None
    try:
        info = _proj.load_project(pid)
    except _proj.ProjectError:
        return None
    return {**info, "status": _proj.project_status(pid), **_proj.run_summary(pid, info)}


def project_status(pid: str) -> dict:
    """status.json (tiến độ 6 bước, lỗi, log) + trạng thái suy ra."""
    s = dict(_proj.read_status(pid))
    s["state_now"] = _proj.project_status(pid)
    return s


def log_tail(pid: str, n: int = 40) -> str:
    """n dòng cuối log của lần chạy gần nhất."""
    s = _proj.read_status(pid)
    p = Path(s.get("log") or "")
    if s.get("log_tail"):
        return s["log_tail"]
    try:
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def ha4m_trials(data_root: str = DEFAULT_DATA_ROOT) -> list[str]:
    """Trial HA4M có trong data/ha4m (U001_V001...)."""
    try:
        return [f"{s}_{t}" for s, t in ha4m_io.list_trials(data_root)]
    except Exception:
        return []


def create_ha4m(stem: str, name: str | None = None, data_root: str = DEFAULT_DATA_ROOT) -> dict:
    """Dự án từ trial HA4M."""
    sub, tri = stem.split("_")[:2]
    return _proj.create_project(f"ha4m://{data_root}?subject={sub}&trial={tri}", name or None)


def create_upload(data: bytes, filename: str, name: str | None = None, max_seconds: float | None = None) -> dict:
    """Dự án từ file video upload (lưu trong dự án)."""
    return _proj.create_project_from_upload(data, filename, name or None, max_seconds=max_seconds or None)


def create_frames(folder: str, fps: float, name: str | None = None) -> dict:
    """Dự án từ thư mục chuỗi ảnh."""
    return _proj.create_project(f"frames://{folder}?fps={fps:g}", name or Path(folder).name)


def unadopted_runs() -> list[str]:
    """Run cũ trong outputs/runs/ chưa thuộc dự án nào."""
    try:
        return _proj.unadopted_runs()
    except Exception:
        return []


def adopt(run_id: str) -> list[str]:
    """Bọc run cũ thành dự án."""
    return _proj.adopt(run_id)


def current_job() -> dict | None:
    """Job pipeline đang chạy toàn app (None nếu rảnh)."""
    return _proj.current_job()


def start_job(pid: str, only: str | None = None) -> tuple[bool, str]:
    """Chạy nền full luồng (hoặc only='render'). (ok, thông báo)."""
    try:
        _pipe.start_job(pid, only=only)
        return True, "đã bắt đầu"
    except _proj.JobBusy as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"không khởi động được: {e}"


def demo_ready(pid: str) -> tuple[bool, list[str]]:
    """Dự án có đủ artifact tính sẵn cho demo mode không (+ thiếu gì)."""
    return _proj.demo_ready(pid)


def demo_video(pid: str) -> tuple[Path | None, dict | None]:
    """(media/demo.mp4, meta nguồn nhãn) — (None, None) nếu chưa render."""
    p = _proj.media_path(pid, "demo.mp4")
    return (p, _proj.demo_video_meta(pid)) if p.exists() else (None, None)


def source_label(src: str | None) -> str:
    """'oracle' → 'chuẩn (ground truth)'."""
    return SOURCE_LABEL.get(src or "", "không rõ")


def benchmark(pid: str) -> dict:
    """Số benchmark per-video + chú thích baseline cố định."""
    try:
        return _proj.benchmark(pid)
    except Exception:
        return {"rows": [], "note": BENCH_NOTE, "model_line": MODEL_LINE, "labels_available": False}


def report_path(pid: str) -> Path | None:
    """report.md của LATEST run (None nếu chưa có)."""
    r = _proj.run_path(pid)
    p = r / "report" / "report.md" if r else None
    return p if p and p.exists() else None


def project_view(pid: str) -> dict | None:
    """Mọi thứ Trang 2 cần từ LATEST run: bảng bước + TG chuẩn, findings (card), tham chiếu, nguồn bước, ghi chú.
    Chỉ đọc artifact (số do 05/06 tính; lệch % do backend kaizen.analyzer.report.std_rows tính)."""
    info = get_project(pid)
    r = _proj.run_path(pid)
    if not info or not r:
        return None
    met = _read_json(r / "metrics" / "metrics.json") or {}
    fdoc = _read_json(r / "findings" / "findings.json") or {}
    edoc = _read_json(r / "explained" / "explained.json") or {}
    m = (met.get("trials") or {}).get(info["stem"])
    station = r / "station.yaml"
    try:
        cfg = load_config(station if station.exists() else (info.get("station") or DEFAULT_CONFIG))
    except Exception:
        cfg = {}
    std, thr_std, thr_wrist = standards_view(fdoc.get("standards"), cfg)
    refs = References()
    steps = []
    if m:
        for row in std_rows(m, std, float(thr_std["value"])):
            lab = refs.use(row["citation"], "bảng bước") if row["citation"] else ""
            steps.append({"Bước": row["step_id"], "Tên": row["name"], "Thời lượng (s)": row["dur_s"],
                          "TG chuẩn (s)": row["std_s"], "Lệch %": row["dev_pct"], "Vượt ngưỡng": row["over"],
                          "Nguồn chuẩn": lab or "—"})
    exps = {e["finding_id"]: e for e in edoc.get("explanations", [])}
    cards = []
    for f in fdoc.get("findings", []):
        x = exps.get(f["id"]) or {}
        fin = x.get("final") or {}
        labels = [refs.use(c, f["id"]) for c in finding_sources(f, x, std, thr_std, thr_wrist)]
        cards.append({"id": f["id"], "rule": f["rule"], "type": f["type"], "severity": f["severity"],
                      "problem": problem_phrase(f), "metric": f["evidence"].get("metric"),
                      "value": f["evidence"].get("value"), "unit": f["evidence"].get("unit"),
                      "impact": f.get("impact_s_per_cycle"), "confidence": f.get("confidence"),
                      "observation": fin.get("observation"), "hypothesis": fin.get("hypothesis"),
                      "recommendation": fin.get("recommendation"), "ecrs": fin.get("ecrs"),
                      "remeasure": fin.get("remeasure"), "validation": x.get("validation"),
                      "sources": [lab for lab in dict.fromkeys(labels) if lab]})
    thr_lab = refs.use(thr_std.get("citation"), "ngưỡng lệch TG chuẩn (R8)") or "ngưỡng mặc định hệ thống"
    return {"info": info, "run_id": r.name, "source": (met.get("meta") or {}).get("source"), "steps": steps,
            "threshold": {"value": thr_std["value"], "label": thr_lab}, "findings": cards,
            "notes": fdoc.get("notes", []), "references": refs.rows_list(),
            "explain": {k: edoc.get(k) for k in ("provider", "llm_model", "pass_rate", "pass_rate_first_attempt",
                                                 "citation_mode", "citation_note", "mode", "prompt")},
            "has_metrics": m is not None}


# ---------------------------------------------------------------- 3M tổng hợp
def compute_3m(project_ids: list[str]):
    """Gọi aggregate_3m (lưu aggregates/<agg_id>/). Trả Aggregate3M (agg.error nếu < 2 video)."""
    return _agg.aggregate_3m(project_ids)


def list_aggregates() -> list[dict]:
    """Aggregate đã lưu (mới nhất trước)."""
    try:
        return _agg.list_aggregates()
    except Exception:
        return []


def load_aggregate(agg_id: str):
    """Mở lại aggregate (None nếu không có)."""
    try:
        return _agg.load_aggregate(agg_id)
    except (FileNotFoundError, TypeError, ValueError):
        return None


def save_review(agg_id: str, note: str) -> None:
    """Ghi chú đánh giá của người xem → aggregate.json."""
    _agg.set_review(agg_id, note)


# ---------------------------------------------------------------- RAG
@st.cache_resource(show_spinner="Đang nạp RAG store (lần đầu tải model embedding)…")
def _store(path: str):
    from kaizen.rag.store import DocumentStore
    return DocumentStore(path)


def rag_store_path() -> str:
    """Thư mục RAG store (env KAIZEN_RAG_STORE / data/rag_store)."""
    from kaizen.rag.store import default_store_path
    return str(default_store_path())


def rag_documents() -> list[dict]:
    """Tài liệu đã ingest (đọc registry, không nạp model)."""
    p = Path(rag_store_path()) / "documents.json"
    reg = _read_json(p) or {}
    return sorted(reg.values(), key=lambda d: d.get("ingested_at", ""), reverse=True)


def rag_ingest(data: bytes, filename: str) -> dict:
    """Ingest 1 tài liệu (PDF/DOCX/MD/TXT) → {doc_id, title, pages, chunks, needs_ocr_pages, already_present}."""
    from kaizen.rag import report_queries
    res = _store(rag_store_path()).ingest(data=data, filename=filename).to_dict()
    report_queries.clear_cache()
    return res


def rag_query(q: str, k: int = 4) -> list[dict]:
    """Top-k chunk: mã tài liệu · trang · excerpt · score · chunk_id (+ toàn văn)."""
    from kaizen.rag.report_queries import chunk_citation
    s = _store(rag_store_path())
    out = []
    for c in s.search(q, k):
        cit = chunk_citation(s, c, q)
        out.append({"label": cit.short, "title": cit.doc_title, "page": c.page, "section": c.section,
                    "score": c.score, "excerpt": cit.excerpt, "chunk_id": c.chunk_id, "text": c.text})
    return out


def aggregate_cards(agg) -> list[dict]:
    """Card findings tổng hợp (R1/R6) — vấn đề ngắn + bằng chứng (số từ findings.json của aggregate)."""
    out = []
    for f in (agg.findings or {}).get("findings", []):
        e = f["evidence"]
        out.append({"id": f["id"], "rule": f["rule"], "type": f["type"], "severity": f["severity"],
                    "problem": problem_phrase(f), "metric": e.get("metric"), "value": e.get("value"),
                    "unit": e.get("unit"), "impact": f.get("impact_s_per_cycle"), "trials": e.get("trials", [])})
    return out

"""Toàn bộ việc ĐỌC artifact cho UI (UI không mở file trực tiếp). Không tính lại chỉ số nào.

Cache: st.cache_data với key (đường dẫn, mtime) → artifact đổi là tự đọc lại, không cần restart.
Mọi hàm trả None khi artifact thiếu để UI hiện cảnh báo nhẹ thay vì crash.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "tools"))

import compare_runs  # noqa: E402
import ha4m_io  # noqa: E402
import overlay  # noqa: E402
import run_io  # noqa: E402
from common import load_config, parse_stem  # noqa: E402
from llm_validate import load_ids, normalize_citation  # noqa: E402

RUNS = REPO / "outputs" / "runs"
KB_FILES = (REPO / "config" / "kaizen_kb.md", REPO / "config" / "sop.md")


def _mt(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return -1.0


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float):
    if mtime < 0:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def _csv(path: str, mtime: float):
    return pd.read_csv(path) if mtime >= 0 else None


@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: float):
    return pd.read_parquet(path) if mtime >= 0 else None


def _read_json(p: Path):
    return _json(str(p), _mt(p))


# ---------------- Run ----------------
def run_root(run_id: str) -> Path:
    return RUNS / run_id


def latest_run() -> str | None:
    return run_io.latest_id()


def list_runs() -> list[dict]:
    """Manifest mọi run, MỚI NHẤT trước (đọc lại khi thư mục runs đổi)."""
    if not RUNS.exists():
        return []
    out = []
    for d in RUNS.iterdir():
        if d.is_dir():
            m = _read_json(d / "manifest.json")
            if m:
                m = dict(m)
                m.setdefault("run_id", d.name)
                out.append(m)
    return sorted(out, key=lambda m: (m.get("created_at") or "", m["run_id"]), reverse=True)


def run_label(m: dict) -> str:
    """Nhãn dropdown giống compare_runs --list: thời điểm · nguồn · pose · LLM · tag."""
    t = (m.get("created_at") or "").replace("T", " ")[:16]
    parts = [t, m.get("source") or "?", m.get("pose_model") or "?", compare_runs.llm_label(m)]
    if m.get("tag"):
        parts.append(f"#{m['tag']}")
    return " · ".join(parts)


def load_manifest(run_id: str) -> dict | None:
    return _read_json(run_root(run_id) / "manifest.json")


def load_metrics(run_id: str) -> dict | None:
    return _read_json(run_root(run_id) / "metrics" / "metrics.json")


def load_findings(run_id: str) -> dict | None:
    return _read_json(run_root(run_id) / "findings" / "findings.json")


def load_explained(run_id: str) -> dict | None:
    return _read_json(run_root(run_id) / "explained" / "explained.json")


def load_eval(run_id: str, trial: str) -> dict | None:
    return _read_json(run_root(run_id) / "steps" / f"{trial}_eval.json")


def load_steps(run_id: str, trial: str, kind: str) -> pd.DataFrame | None:
    """kind = 'oracle' | 'pred'."""
    p = run_root(run_id) / "steps" / f"{trial}_{kind}.csv"
    return _csv(str(p), _mt(p))


def load_events(run_id: str, trial: str) -> pd.DataFrame | None:
    p = run_root(run_id) / "events" / f"{trial}.csv"
    return _csv(str(p), _mt(p))


def load_features(run_id: str, trial: str) -> pd.DataFrame | None:
    p = run_root(run_id) / "features" / f"{trial}.parquet"
    return _parquet(str(p), _mt(p))


def run_source(run_id: str) -> str | None:
    """Nguồn bước dùng cho metrics của run: oracle | pred (ưu tiên metrics.json, fallback manifest)."""
    met = load_metrics(run_id) or {}
    return (met.get("meta") or {}).get("source") or (load_manifest(run_id) or {}).get("source")


def run_trials(run_id: str) -> list[str]:
    """Trial của run: hợp của metrics.json và manifest (có trial chỉ có features, vd thiếu nhãn ở run oracle)."""
    met = load_metrics(run_id) or {}
    man = load_manifest(run_id) or {}
    return sorted(set(met.get("trials", {})) | set(man.get("trials", {})))


def station(run_id: str) -> dict:
    p = run_root(run_id) / "station.yaml"
    return load_config(p if p.exists() else REPO / "config" / "station.yaml")


def station_diff(run_id: str) -> list[str]:
    """Khoá cấp 1 của station.yaml snapshot khác config/station.yaml hiện tại."""
    p = run_root(run_id) / "station.yaml"
    cur = REPO / "config" / "station.yaml"
    if not p.exists() or not cur.exists():
        return []
    a = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    b = yaml.safe_load(cur.read_text(encoding="utf-8")) or {}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def report_path(run_id: str) -> Path:
    return run_root(run_id) / "report" / "report.md"


def asset(run_id: str, name: str) -> Path | None:
    p = run_root(run_id) / "report" / "report_assets" / name
    return p if p.exists() else None


def demo_video(run_id: str, trial: str) -> Path:
    return run_root(run_id) / "report" / f"demo_{trial}.mp4"


@st.cache_data(show_spinner=False, max_entries=8)
def _b64(path: str, mtime: float) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def video_b64(path: Path) -> str | None:
    """mp4 → base64 (cache theo mtime) để nhúng vào trình phát so sánh."""
    return _b64(str(path), _mt(path)) if path.exists() else None


def steps_for_run(run_id: str, trial: str, prefer: str | None = None) -> tuple[pd.DataFrame | None, str]:
    """Steps theo `prefer` (oracle|pred) hoặc nguồn của run; fallback nguồn còn lại. Trả (df, nguồn)."""
    src = prefer or run_source(run_id) or "oracle"
    for k in (src, "pred" if src == "oracle" else "oracle"):
        df = load_steps(run_id, trial, k)
        if df is not None:
            return df, k
    return None, ""


# ---------------- KB / SOP ----------------
@st.cache_data(show_spinner=False)
def _kb(mtimes: tuple) -> dict[str, str]:
    return load_ids(*KB_FILES)


def kb_lookup(citation: str) -> str | None:
    """Nguyên văn đoạn [KB-xx]/[SOP-xx] (chấp nhận kb13, [KB-13]...)."""
    cid = normalize_citation(citation)
    return _kb(tuple(_mt(p) for p in KB_FILES)).get(cid) if cid else None


# ---------------- Frame + overlay ----------------
@st.cache_data(show_spinner=False, max_entries=64)
def _frame(run_id: str, trial: str, frame_idx: int, width: int, mt_key: tuple):
    man = load_manifest(run_id) or {}
    root = man.get("data_root") or ha4m_io.DEFAULT_ROOT
    if not Path(str(root).split(",")[0]).is_absolute():
        root = ",".join(str(REPO / r.strip()) for r in str(root).split(","))
    sub, tri = parse_stem(trial)
    try:
        img = ha4m_io.read_frame(root, sub, tri, frame_idx)
    except FileNotFoundError:
        return None
    if img is None:
        return None
    cfg = station(run_id)
    od = overlay.load_overlay_data(run_root(run_id), trial, cfg, run_source(run_id) or "")
    import cv2
    out = overlay.render_frame(img, frame_idx / float(cfg["fps"]), od, width)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def load_frame(run_id: str, trial: str, t: float, width: int = 960):
    """Ảnh RGB tại thời điểm t (giây) có overlay; None nếu không đọc được ảnh gốc."""
    fps = float(station(run_id)["fps"])
    r = run_root(run_id)
    key = tuple(_mt(p) for p in (r / "features" / f"{trial}.parquet", r / "events" / f"{trial}.csv",
                                 r / "steps" / f"{trial}_oracle.csv", r / "steps" / f"{trial}_pred.csv"))
    return _frame(run_id, trial, int(round(t * fps)), width, key)


# ---------------- So sánh run ----------------
def compare(run_ids: list[str]) -> dict:
    return compare_runs.compare(run_ids)


COMPARE_COLS = compare_runs.SUMMARY_COLS

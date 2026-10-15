"""Tiện ích dùng chung cho các script pipeline: config, đường dẫn, chọn trial, ghi file an toàn."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ha4m_io  # noqa: E402

DEFAULT_CONFIG = "config/station.yaml"
OUT = Path("outputs")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Đọc station.yaml, điền giá trị mặc định."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("fps", 30)
    cfg.setdefault("px_per_cm", None)
    cfg.setdefault("moving_speed_px_s", 120)
    cfg.setdefault("min_event_s", 0.25)
    cfg.setdefault("idle_threshold_s", 3.0)
    cfg.setdefault("zones", {})
    cfg.setdefault("steps", [])
    cfg.setdefault("cycle_time_target_s", None)
    return cfg


def setup(a: argparse.Namespace, tag: str, need_run: bool = True):
    """Resolve run (--run/LATEST/legacy) và điền a.data_root, a.config mặc định. Trả RunContext | None."""
    import run_io
    ctx = run_io.resolve(a.run, tag) if need_run else None
    man = ctx.manifest if ctx else {}
    if not a.data_root:
        a.data_root = man.get("data_root") or ha4m_io.DEFAULT_ROOT
    if not a.config:
        a.config = str(ctx.station_path) if ctx and ctx.station_path.exists() else DEFAULT_CONFIG
    return ctx


def stage_dir(ctx, override: str | None, stage: str, legacy: str) -> Path:
    """Đường dẫn stage: --xxx-dir tường minh > thư mục trong run > đường dẫn phẳng cũ."""
    if override:
        return Path(override)
    if ctx is not None:
        return ctx.dir(stage)
    return Path(legacy)


def stage_file(ctx, override: str | None, stage: str, filename: str, legacy: str) -> Path:
    if override:
        return Path(override)
    if ctx is not None:
        return ctx.path(stage, filename)
    return Path(legacy)


def stem(subject: str, trial: str) -> str:
    return f"{subject}_{trial}"


def parse_stem(s: str) -> tuple[str, str]:
    subject, trial = s.split("_")[:2]
    return subject, trial


def base_parser(desc: str) -> argparse.ArgumentParser:
    """Args chung: --run, --data-root, --config, --subjects, --trials, --force."""
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="run_id hoặc 'latest' (mặc định: outputs/runs/LATEST; chưa có run → đường dẫn phẳng cũ)")
    ap.add_argument("--data-root", default=None,
                    help=f"thư mục dataset, nhiều thư mục cách nhau dấu phẩy (mặc định: data_root trong manifest "
                         f"của run, rồi env HA4M_ROOT, rồi {ha4m_io.DEFAULT_ROOT})")
    ap.add_argument("--config", default=None, help="mặc định: station.yaml chụp trong run, không có thì " + DEFAULT_CONFIG)
    ap.add_argument("--subjects", default="", help="vd U001,U002 (rỗng = tất cả)")
    ap.add_argument("--trials", default="", help="vd V001,V002 (rỗng = tất cả)")
    ap.add_argument("--force", action="store_true", help="ghi đè output đã có")
    return ap


def _split(s: str) -> set[str]:
    xs = {x.strip() for x in (s or "").split(",") if x.strip()}
    return set() if xs & {"all", "ALL"} else xs


def select_stems(stems: list[str], subjects: str = "", trials: str = "") -> list[str]:
    """Lọc danh sách '<subject>_<trial>' theo --subjects/--trials."""
    ss, tt = _split(subjects), _split(trials)
    out = []
    for s in sorted(set(stems)):
        sub, tri = parse_stem(s)
        if (not ss or sub in ss) and (not tt or tri in tt):
            out.append(s)
    return out


def stems_from_dir(d: Path, pattern: str, suffix_strip: str) -> list[str]:
    """Liệt kê stem từ file trong d khớp pattern (vd '*.parquet')."""
    return [p.name[: -len(suffix_strip)] if suffix_strip else p.stem for p in sorted(d.glob(pattern))]


def can_write(path: Path, force: bool, tag: str) -> bool:
    """False (kèm log) nếu file đã tồn tại và không có --force."""
    if path.exists() and not force:
        print(f"[{tag}] bỏ qua {path} (đã có, dùng --force để ghi đè)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def r(x, nd: int = 2):
    """Làm tròn an toàn (None/NaN → None)."""
    try:
        import math
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None

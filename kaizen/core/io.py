"""Tiện ích I/O dùng chung: stem trial, chọn trial, ghi file an toàn, JSON (numpy-aware), làm tròn."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def stem(subject: str, trial: str) -> str:
    """'U001', 'V001' → 'U001_V001'."""
    return f"{subject}_{trial}"


def parse_stem(s: str) -> tuple[str, str]:
    """'U001_V001[_...]' → ('U001', 'V001')."""
    subject, trial = s.split("_")[:2]
    return subject, trial


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


def write_json(path: Path, obj: object) -> None:
    """Ghi JSON UTF-8 indent 2; numpy scalar/array và pydantic model được chuyển tự động."""
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
    if hasattr(o, "to_json_dict"):
        return o.to_json_dict()
    raise TypeError(type(o))


def read_json(path: Path) -> Any:
    """Đọc JSON UTF-8."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def r(x: object, nd: int = 2) -> float | None:
    """Làm tròn an toàn (None/NaN → None)."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None

"""Hàm định dạng hiển thị cho UI (không tính chỉ số — chỉ đọc/format)."""
from __future__ import annotations


def val(m: dict | None, key: str):
    """Giá trị metric {value, unit, definition} hoặc None."""
    x = (m or {}).get(key)
    return x.get("value") if isinstance(x, dict) else None


def unit(m: dict | None, key: str, default: str = "") -> str:
    x = (m or {}).get(key)
    return x.get("unit", default) if isinstance(x, dict) else default


def fmt(v, suffix: str = "", nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        s = f"{v:.{nd}f}"
        if "." in s:                       # chỉ bỏ số 0 thừa SAU dấu thập phân (18660 giữ nguyên)
            s = s.rstrip("0").rstrip(".")
        return s + suffix
    return f"{v}{suffix}"

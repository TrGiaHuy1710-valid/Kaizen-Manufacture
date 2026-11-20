"""Cấu hình trạm (station.yaml) + đường dẫn mặc định. Mọi đường dẫn tương đối tính từ thư mục chạy (repo root)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "config/station.yaml"
DEFAULT_DATA_ROOT = os.environ.get("HA4M_ROOT", "data/ha4m")
OUT = Path("outputs")

# Kiểu dùng trong chữ ký hàm: dict đọc từ station.yaml (đã điền mặc định) hoặc StationConfig (pydantic).
StationLike = Mapping[str, Any]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Đọc station.yaml, điền giá trị mặc định."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return with_defaults(cfg)


def with_defaults(cfg: dict) -> dict:
    """Điền giá trị mặc định vào dict cấu hình trạm (sửa tại chỗ, trả lại chính nó)."""
    cfg.setdefault("fps", 30)
    cfg.setdefault("px_per_cm", None)
    cfg.setdefault("moving_speed_px_s", 120)
    cfg.setdefault("min_event_s", 0.25)
    cfg.setdefault("idle_threshold_s", 3.0)
    cfg.setdefault("zones", {})
    cfg.setdefault("steps", [])
    cfg.setdefault("cycle_time_target_s", None)
    return cfg


def as_cfg(station: StationLike) -> dict:
    """StationConfig | dict → dict cấu hình (các hàm tính toán làm việc trên dict để giữ nguyên hành vi cũ)."""
    if hasattr(station, "to_dict"):
        return station.to_dict()  # type: ignore[union-attr]
    return station if isinstance(station, dict) else dict(station)

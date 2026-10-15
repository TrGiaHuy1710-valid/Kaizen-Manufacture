"""Tiện ích batch trích keypoints (stage 01): đọc meta .npz, giới hạn trial / subject."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUSPECT_DETECT = 0.85


def npz_meta(p: Path) -> dict:
    """meta + số frame của 1 .npz (đọc lười, không nạp kpts)."""
    try:
        with np.load(p, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
            meta["n_frames"] = int(d["t"].shape[0])
        return meta
    except Exception as e:  # file hỏng
        return {"error": str(e)}


def limit_per_subject(stems: list[str], n: int) -> list[str]:
    """Giữ tối đa n trial đầu mỗi subject (0 = tất cả)."""
    if n <= 0:
        return stems
    out, cnt = [], {}
    for st in stems:
        sub = st.split("_")[0]
        if cnt.get(sub, 0) < n:
            out.append(st)
            cnt[sub] = cnt.get(sub, 0) + 1
    return out

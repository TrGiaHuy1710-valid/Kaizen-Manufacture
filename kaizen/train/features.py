"""Đặc trưng theo frame cho model phân đoạn bước (spec `kp_v1`) — numpy thuần, DÙNG CHUNG cho train và suy luận.

Từ .npz keypoints (schema chung của kaizen.skeleton.extract), mỗi frame → vector D = 7·J chiều với J = 15 khớp:
  - toạ độ tuyệt đối  x/W, y/H                         (2J) — vị trí khay/vùng lắp cố định theo camera
  - toạ độ theo thân  (p − giữa vai) / độ rộng vai      (2J) — bất biến với vị trí đứng của người
  - vận tốc tuyệt đối Δ(x/W, y/H) · fps                (2J)
  - mask hợp lệ       score ≥ 0.3                       (J)
Khớp thiếu trong model pose (vd COCO-17 không có điểm bàn tay) → 0 và mask 0, nên D luôn cố định.
Đoạn mất dấu ≤ 15 frame được nội suy tuyến tính (giống 02_features), phần còn lại điền 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURE_VERSION = "kp_v1"
JOINTS = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
          "lh_middle_1", "rh_middle_1", "lh_index_4", "rh_index_4", "lh_thumb_4", "rh_thumb_4"]
SCORE_MIN = 0.3
MAX_GAP = 15


def feature_dim() -> int:
    return 7 * len(JOINTS)


def feature_names() -> list[str]:
    out = []
    for kind in ("abs", "rel", "vel"):
        for j in JOINTS:
            out += [f"{kind}_{j}_x", f"{kind}_{j}_y"]
    return out + [f"mask_{j}" for j in JOINTS]


def _interp_short(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Nội suy tuyến tính các đoạn NaN ≤ max_gap nằm giữa 2 điểm hợp lệ (theo trục 0)."""
    x = x.copy()
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return x
    idx = np.arange(len(x))
    i = 0
    while i < len(x):
        if ok[i]:
            i += 1
            continue
        j = i
        while j < len(x) and not ok[j]:
            j += 1
        if i > 0 and j < len(x) and j - i <= max_gap:
            x[i:j] = np.interp(idx[i:j], [i - 1, j], [x[i - 1], x[j]])
        i = j
    return x


def load_npz(path: str | Path) -> dict:
    """{kpts, t, frame, names, meta} từ .npz keypoints."""
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    t = d["t"].astype(np.float64)
    frame = d["frame"].astype(np.int64) if "frame" in d.files else np.round(t * meta.get("src_fps", 30)).astype(np.int64)
    return {"kpts": d["kpts"].astype(np.float64), "t": t, "frame": frame,
            "names": [str(n) for n in d["names"]], "meta": meta}


def frame_features(kp: dict, fps: float | None = None) -> np.ndarray:
    """(T, D) float32 theo spec kp_v1 (xem docstring module)."""
    kpts, names, meta = kp["kpts"], kp["names"], kp["meta"]
    T = len(kpts)
    W = float(meta.get("width") or np.nanmax(kpts[..., 0]) or 1.0)
    H = float(meta.get("height") or np.nanmax(kpts[..., 1]) or 1.0)
    fps = float(fps or meta.get("src_fps") or 30.0)
    J = len(JOINTS)
    xy = np.full((T, J, 2), np.nan)
    for j, name in enumerate(JOINTS):
        if name not in names:
            continue
        k = kpts[:, names.index(name)]
        good = ~np.isnan(k[:, 2]) & (k[:, 2] >= SCORE_MIN)
        for c in range(2):
            xy[:, j, c] = _interp_short(np.where(good, k[:, c], np.nan), MAX_GAP)
    mask = (~np.isnan(xy[..., 0])).astype(np.float64)                  # (T, J)
    absn = xy / np.array([W, H])                                       # (T, J, 2)
    ls, rs = absn[:, JOINTS.index("l_shoulder")], absn[:, JOINTS.index("r_shoulder")]
    mid = (ls + rs) / 2
    width = np.linalg.norm(ls - rs, axis=1)
    med_w = np.nanmedian(width) if np.isfinite(width).any() else 0.1
    width = np.where(np.isfinite(width) & (width > 0.2 * med_w), width, med_w)
    mid = np.where(np.isnan(mid), np.nanmedian(mid, axis=0) if np.isfinite(mid).any() else 0.5, mid)
    rel = (absn - mid[:, None, :]) / width[:, None, None]
    vel = np.zeros_like(absn)
    vel[1:] = (absn[1:] - absn[:-1]) * fps
    feats = np.concatenate([absn.reshape(T, -1), rel.reshape(T, -1), vel.reshape(T, -1), mask], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

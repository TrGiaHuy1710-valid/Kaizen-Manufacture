"""Trích keypoints từ 1 Source → .npz (schema chung, không phụ thuộc model).

Schema output (.npz) — KHÔNG đổi:
  kpts   : float32 (T, K, 3)  -> x_px, y_px, score   (NaN nếu không thấy), toạ độ theo ảnh GỐC
  t      : float32 (T,)       -> thời gian (giây) = frame_idx / fps
  frame  : int32 (T,)         -> frame index gốc
  names  : (K,)               -> tên keypoint
  meta   : json string        -> core.schemas.KeypointsMeta (model, fps gốc, stride, kích thước, roi, proc_fps...)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from kaizen.skeleton.pose import PoseBackend, draw, get_pose_backend
from kaizen.skeleton.sources import Source

WRIST_KEYS = ("l_wrist", "r_wrist", "lh_wrist", "rh_wrist", "lh_middle_1", "rh_middle_1")


def parse_roi(s: str | None) -> tuple[int, int, int, int] | None:
    """'x,y,w,h' → tuple; rỗng → None."""
    if not s:
        return None
    x, y, w, h = (int(v) for v in s.split(","))
    return x, y, w, h


def npz_path_for(source: Source, model: str, out: str | Path, name: str = "") -> Path:
    """out là file .npz → giữ nguyên; là thư mục → <out>/<stem>_<model>.npz."""
    out = Path(out)
    if out.suffix == ".npz":
        return out
    return out / f"{name or source.meta()['stem']}_{model}.npz"


def extract(source: Source, model: str | PoseBackend = "mediapipe", stride: int = 1,
            out: str | Path = "data/keypoints", *, name: str = "", max_frames: int = 0,
            roi: tuple[int, int, int, int] | None = None, hand_crop: bool | None = None,
            preview: bool = False, preview_scale: float = 0.5, rtmw_mode: str = "balanced", device: str = "cpu",
            model_name: str | None = None, log: Callable[[str], None] = print) -> Path:
    """Chạy pose trên mọi frame (1/stride) của source, ghi .npz schema cũ; trả đường dẫn .npz.

    model: tên trong POSE_BACKENDS hoặc 1 PoseBackend đã khởi tạo (khi đó đặt model_name cho tên file/meta).
    hand_crop=None → bật cho nguồn chuỗi ảnh (ha4m/frames — ảnh lớn, tay nhỏ), tắt cho video.
    """
    meta_src = source.meta()
    is_frames = meta_src.get("frames_dir") is not None
    if isinstance(model, str):
        mname = model
        hc = is_frames if hand_crop is None else hand_crop
        be = get_pose_backend(model, hand_crop=hc, rtmw_mode=rtmw_mode, device=device)
    else:
        mname, be = model_name or type(model).__name__.lower(), model
    npz_path = npz_path_for(source, mname, out, name)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(source.fps())
    W, H = int(meta_src["width"]), int(meta_src["height"])

    writer = None
    if preview:
        pw, ph = int(W * preview_scale), int(H * preview_scale)
        writer = cv2.VideoWriter(str(npz_path.with_name(npz_path.stem + "_preview.mp4")),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (pw, ph))

    kpts, ts, fidx, t0 = [], [], [], time.time()
    for i, frame in source.frames(stride, max_frames):
        t_ms = int(round(i * 1000 / fps))
        if roi:
            x, y, w, h = roi
            kp = be(np.ascontiguousarray(frame[y:y + h, x:x + w]), t_ms)
            kp[:, 0] += x; kp[:, 1] += y
        else:
            kp = be(frame, t_ms)
        kpts.append(kp); ts.append(i / fps); fidx.append(i)
        if writer:
            vis = draw(frame.copy(), kp)
            if roi:
                cv2.rectangle(vis, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 128, 0), 2)
            cv2.putText(vis, f"f{i} t={i / fps:.2f}s", (20, 50), 0, 1.5, (0, 255, 255), 3)
            writer.write(cv2.resize(vis, (pw, ph)))
        if len(kpts) % 100 == 0:
            log(f"{len(kpts)} frames, {len(kpts) / (time.time() - t0):.1f} fps")
    if writer:
        writer.release()

    proc_fps = len(kpts) / max(time.time() - t0, 1e-6)
    kpts = np.stack(kpts) if kpts else np.empty((0, len(be.names), 3), np.float32)
    detect_rate = float(np.mean(~np.isnan(kpts[:, :, 0]).all(1))) if len(kpts) else 0.0
    names = list(be.names)
    wr = {}
    for k in WRIST_KEYS:
        if k in names:
            s = kpts[:, names.index(k), 2]
            wr[k] = round(float(np.mean(np.nan_to_num(s, nan=0) >= 0.3)), 3) if len(s) else 0.0
    meta = dict(model=mname, video=meta_src.get("video"), frames_dir=meta_src.get("frames_dir"), src_fps=fps,
                stride=stride, width=W, height=H, roi=roi, hand_crop=getattr(be, "hand_crop", None),
                proc_fps=round(proc_fps, 2), detect_rate=round(detect_rate, 3), valid_rate_score03=wr)
    np.savez_compressed(npz_path, kpts=kpts, t=np.array(ts, np.float32), frame=np.array(fidx, np.int32),
                        names=np.array(names), meta=json.dumps(meta))
    log(json.dumps(meta, indent=2))
    log(f"[extract] đã ghi {npz_path}")
    return npz_path

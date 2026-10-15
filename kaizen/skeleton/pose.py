"""Pose backends sau 1 interface `PoseBackend`: frame BGR → (K, 3) [x_px, y_px, score] (NaN nếu không thấy).

Thêm backend mới = thêm 1 entry vào POSE_BACKENDS (tên → factory(**opts) -> PoseBackend).
    mediapipe        : 33 pose + 21×2 tay = 75 điểm (tay ghép theo cổ tay pose gần nhất); cần models/*.task (setup.sh)
    mediapipe-heavy  : như trên, pose_landmarker_heavy
    rtmw             : RTMW wholebody 133 điểm qua rtmlib/onnxruntime
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class PoseBackend(Protocol):
    """frame BGR (ảnh gốc hoặc ROI) + timestamp ms → mảng (len(names), 3) float32 [x_px, y_px, score]."""
    names: list[str]

    def __call__(self, frame_bgr: np.ndarray, ts_ms: int) -> np.ndarray: ...


# ---------------- Tên keypoint ----------------
MP_POSE = ["nose", "l_eye_in", "l_eye", "l_eye_out", "r_eye_in", "r_eye", "r_eye_out",
           "l_ear", "r_ear", "mouth_l", "mouth_r", "l_shoulder", "r_shoulder",
           "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_pinky", "r_pinky",
           "l_index", "r_index", "l_thumb", "r_thumb", "l_hip", "r_hip",
           "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel",
           "l_foot", "r_foot"]
HAND = ["wrist"] + [f"{f}_{j}" for f in ["thumb", "index", "middle", "ring", "pinky"]
                    for j in range(1, 5)]
MP_NAMES = MP_POSE + [f"lh_{n}" for n in HAND] + [f"rh_{n}" for n in HAND]  # 33 + 42 = 75

COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
          "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
          "l_knee", "r_knee", "l_ankle", "r_ankle"]
RTMW_NAMES = (COCO17 + [f"foot_{i}" for i in range(6)] + [f"face_{i}" for i in range(68)]
              + [f"lh_{n}" for n in HAND] + [f"rh_{n}" for n in HAND])  # 133


# ---------------- Backend MediaPipe ----------------
class MediaPipeBackend:
    """Pose (VIDEO mode) + Hand.

    hand_crop=True: dò bàn tay trên vùng vuông quanh từng cổ tay pose (IMAGE mode) thay vì cả ảnh —
    cần cho ảnh lớn (HA4M 2048x1536) nơi bàn tay chỉ ~60px, và tránh bắt nhầm tay người khác.
    """

    def __init__(self, model_dir="models", heavy=False, hand_crop=False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.hand_crop = hand_crop
        pose_file = "pose_landmarker_heavy.task" if heavy else "pose_landmarker_full.task"
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(model_dir) / pose_file)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        mode = vision.RunningMode.IMAGE if hand_crop else vision.RunningMode.VIDEO
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(model_dir) / "hand_landmarker.task")),
            running_mode=mode, num_hands=2))
        self.names = MP_NAMES

    def _img(self, bgr):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    def __call__(self, frame_bgr, ts_ms):
        h, w = frame_bgr.shape[:2]
        out = np.full((len(self.names), 3), np.nan, np.float32)

        pr = self.pose.detect_for_video(self._img(frame_bgr), ts_ms)
        if pr.pose_landmarks:
            for i, lm in enumerate(pr.pose_landmarks[0]):
                out[i] = (lm.x * w, lm.y * h, lm.visibility)

        if self.hand_crop:
            self._hands_cropped(frame_bgr, out)
            return out

        hr = self.hand.detect_for_video(self._img(frame_bgr), ts_ms)
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            # Ghép tay vào cổ tay pose gần nhất -> tránh lỗi nhãn Left/Right
            # của MediaPipe (nó giả định ảnh bị lật kiểu selfie).
            wx, wy = lms[0].x * w, lms[0].y * h
            side = self._closest_side(out, wx, wy, hd[0].category_name)
            base = 33 if side == "l" else 54
            score = hd[0].score
            for j, lm in enumerate(lms):
                out[base + j] = (lm.x * w, lm.y * h, score)
        return out

    def _hands_cropped(self, frame, out):
        h, w = frame.shape[:2]
        for side, wi, ei, base in (("l", 15, 13, 33), ("r", 16, 14, 54)):
            wx, wy, ws = out[wi]
            if np.isnan(wx) or ws < 0.2:
                continue
            ex, ey = out[ei, :2]
            fore = np.hypot(wx - ex, wy - ey) if not np.isnan(ex) else 100.0
            half = int(np.clip(1.3 * fore, 96, 300))
            # dời tâm crop về phía bàn tay (hướng elbow->wrist)
            cx, cy = wx, wy
            if not np.isnan(ex) and fore > 1:
                cx, cy = wx + 0.5 * half * (wx - ex) / fore, wy + 0.5 * half * (wy - ey) / fore
            x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
            x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            crop = frame[y0:y1, x0:x1]
            scale = 256.0 / max(crop.shape[:2])
            if scale > 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
            hr = self.hand.detect(self._img(crop))
            best, bd = None, 1e9
            for lms, hd in zip(hr.hand_landmarks, hr.handedness):
                hx = x0 + lms[0].x * crop.shape[1] / scale
                hy = y0 + lms[0].y * crop.shape[0] / scale
                d = np.hypot(hx - wx, hy - wy)
                if d < bd:
                    best, bd = (lms, hd[0].score), d
            if best is None or bd > 0.6 * half:   # tay khác (vd tay còn lại) -> bỏ
                continue
            lms, score = best
            for j, lm in enumerate(lms):
                out[base + j] = (x0 + lm.x * crop.shape[1] / scale, y0 + lm.y * crop.shape[0] / scale, score)

    @staticmethod
    def _closest_side(out, x, y, fallback):
        lw, rw = out[15, :2], out[16, :2]
        if np.isnan(lw).any() or np.isnan(rw).any():
            return "r" if fallback == "Left" else "l"  # đảo vì camera không lật
        return "l" if np.hypot(*(lw - (x, y))) < np.hypot(*(rw - (x, y))) else "r"


# ---------------- Backend RTMW (rtmlib) ----------------
class RTMWBackend:
    """RTMW wholebody 133 điểm (rtmlib, ONNX); 1 công nhân / trạm → người có tổng score body lớn nhất."""

    def __init__(self, mode="balanced", device="cpu"):
        from rtmlib import Wholebody
        self.model = Wholebody(mode=mode, backend="onnxruntime", device=device)
        self.names = RTMW_NAMES

    def __call__(self, frame_bgr, ts_ms):
        kps, scores = self.model(frame_bgr)  # (N,133,2), (N,133)
        out = np.full((len(self.names), 3), np.nan, np.float32)
        if kps is None or len(kps) == 0:
            return out
        # Một công nhân / trạm: chọn người có tổng score body lớn nhất
        idx = int(np.argmax(scores[:, :17].sum(1)))
        out[:, :2], out[:, 2] = kps[idx], scores[idx]
        return out


def draw(frame: np.ndarray, kp: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """Vẽ chấm xanh cho keypoint có score ≥ thr (preview)."""
    for x, y, s in kp:
        if not np.isnan(x) and s >= thr:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    return frame


POSE_BACKENDS: dict[str, Callable[..., PoseBackend]] = {
    "mediapipe": lambda hand_crop=False, model_dir="models", **_: MediaPipeBackend(model_dir, False, hand_crop),
    "mediapipe-heavy": lambda hand_crop=False, model_dir="models", **_: MediaPipeBackend(model_dir, True, hand_crop),
    "rtmw": lambda rtmw_mode="balanced", device="cpu", **_: RTMWBackend(rtmw_mode, device),
}


def get_pose_backend(name: str, **opts) -> PoseBackend:
    """Khởi tạo backend theo tên trong POSE_BACKENDS (opts: hand_crop, model_dir, rtmw_mode, device)."""
    if name not in POSE_BACKENDS:
        raise ValueError(f"pose model không hỗ trợ: {name!r} (có: {sorted(POSE_BACKENDS)})")
    return POSE_BACKENDS[name](**opts)

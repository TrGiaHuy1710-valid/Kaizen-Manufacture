"""I/O cho dataset HA4M — schema đã xác nhận bằng src/00_inspect.py (xem outputs/inspect_report.md).

Cấu trúc thật (khác giả định ban đầu trong CLAUDE.md):

    <root>/ID<subject><trial>/            vd: IDU001V001  (subject=U001, trial=V001 = 1 video = 1 lần lắp)
        Color/<camera_serial>/FrameID000000_DeviceTimeStamp450817700us.png   (2048x1536 RGB)
        Skeletons/<camera_serial>/FrameID000000_DeviceTimeStamp450817733us.txt (TSV, 32 joint Azure Kinect)
        Depth/ Infrared/ ColorAligned/ PointClouds_DepthGeometry/   (không dùng)
        Labels.txt   # mỗi dòng 1 frame: "<frame_idx> <action_id> <repetition>" (cách nhau dấu cách, CRLF)
                     # action_id 0 = "don't care"; repetition ≠ 0 chỉ ở action 2, 3, 11

`root` có thể là thư mục chứa nhiều IDU*V* hoặc chính 1 thư mục IDU*V*.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

VIDEO_DIR_RE = re.compile(r"^ID(U\d+)(V\d+)$")
FRAME_RE = re.compile(r"FrameID(\d+)_DeviceTimeStamp(\d+)us", re.I)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_ROOT = os.environ.get("HA4M_ROOT", "data/ha4m")

# Tên action theo Table 3 của paper HA4M (Cicirelli et al., Sci Data 2022). Labels.txt chỉ có ID.
ACTION_NAMES = {
    0: "Don't care",
    1: "Pick up/Place Carrier",
    2: "Pick up/Place Gear Bearings (x3)",
    3: "Pick up/Place Planet Gears (x3)",
    4: "Pick up/Place Carrier Shaft",
    5: "Pick up/Place Sun Shaft",
    6: "Pick up/Place Sun Gear",
    7: "Pick up/Place Sun Gear Bearing",
    8: "Pick up/Place Ring Bear",
    9: "Pick up Block 2 and place it on Block 1",
    10: "Pick up/Place Cover",
    11: "Pick up/Place Screw (x2)",
    12: "Pick up Allen Key, turn both screws, return Allen Key and the EGT",
}


def _video_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if VIDEO_DIR_RE.match(root.name):
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and VIDEO_DIR_RE.match(p.name))


def video_dir(root: str | Path, subject: str, trial: str) -> Path:
    """Thư mục của 1 video (subject='U001', trial='V001')."""
    for d in _video_dirs(root):
        if d.name == f"ID{subject}{trial}":
            return d
    raise FileNotFoundError(f"Không thấy ID{subject}{trial} trong {root}")


def list_trials(root: str | Path = DEFAULT_ROOT) -> list[tuple[str, str]]:
    """[(subject, trial)] cho mọi thư mục IDU*V* trong root."""
    out = []
    for d in _video_dirs(root):
        m = VIDEO_DIR_RE.match(d.name)
        out.append((m.group(1), m.group(2)))
    return out


def modality_dir(root: str | Path, subject: str, trial: str, modality: str = "Color") -> Path:
    """Color/<camera_serial>/ — tự lấy thư mục con duy nhất (serial camera)."""
    base = video_dir(root, subject, trial) / modality
    subs = sorted(p for p in base.iterdir() if p.is_dir())
    return subs[0] if subs else base


def frame_files(root: str | Path, subject: str, trial: str, modality: str = "Color") -> list[tuple[int, int, Path]]:
    """[(frame_idx, device_timestamp_us, path)] sort theo SỐ frame (bỏ Thumbs.db v.v.)."""
    d = modality_dir(root, subject, trial, modality)
    out = []
    for p in d.iterdir():
        m = FRAME_RE.search(p.name)
        if m and (p.suffix.lower() in IMG_EXT or modality == "Skeletons"):
            out.append((int(m.group(1)), int(m.group(2)), p))
    out.sort(key=lambda x: x[0])
    return out


def frames_dir_files(frames_dir: str | Path) -> list[tuple[int, Path]]:
    """Ảnh trong 1 thư mục bất kỳ, sort theo số đầu tiên trong tên (fallback: số cuối)."""
    out = []
    for p in Path(frames_dir).iterdir():
        if p.suffix.lower() not in IMG_EXT:
            continue
        m = FRAME_RE.search(p.name)
        if m:
            idx = int(m.group(1))
        else:
            nums = re.findall(r"\d+", p.stem)
            if not nums:
                continue
            idx = int(nums[-1])
        out.append((idx, p))
    out.sort(key=lambda x: x[0])
    return out


def frames_iter(root: str | Path, subject: str, trial: str, stride: int = 1,
                max_frames: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """yield (frame_idx, bgr_image) theo thứ tự số frame."""
    files = frame_files(root, subject, trial)
    yield from _iter_paths([(i, p) for i, _, p in files], stride, max_frames)


def frames_dir_iter(frames_dir: str | Path, stride: int = 1,
                    max_frames: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """Như frames_iter nhưng nhận thẳng 1 thư mục ảnh."""
    yield from _iter_paths(frames_dir_files(frames_dir), stride, max_frames)


def _iter_paths(files, stride, max_frames):
    n = 0
    for k, (idx, p) in enumerate(files):
        if k % stride:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[io] cảnh báo: không đọc được {p}")
            continue
        yield idx, img
        n += 1
        if max_frames and n >= max_frames:
            break


def read_frame(root: str | Path, subject: str, trial: str, frame_idx: int) -> np.ndarray | None:
    """Đọc 1 frame theo index (None nếu không có)."""
    for i, _, p in frame_files(root, subject, trial):
        if i == frame_idx:
            return cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None


def load_frame_labels(root: str | Path, subject: str, trial: str) -> pd.DataFrame:
    """Labels.txt thô: DataFrame[frame, action_id, repetition] (1 dòng / frame)."""
    p = video_dir(root, subject, trial) / "Labels.txt"
    df = pd.read_csv(p, sep=r"\s+", header=None, names=["frame", "action_id", "repetition"],
                     engine="python")
    return df.astype(int).sort_values("frame").reset_index(drop=True)


def labels_to_segments(fl: pd.DataFrame, keep_dont_care: bool = False) -> pd.DataFrame:
    """Gộp các frame liên tiếp cùng (action_id, repetition) thành đoạn.
    end_frame là frame CUỐI (inclusive)."""
    key = fl["action_id"].astype(str) + "_" + fl["repetition"].astype(str)
    gap = fl["frame"].diff().fillna(1) != 1
    grp = ((key != key.shift()) | gap).cumsum()
    seg = fl.groupby(grp).agg(action_id=("action_id", "first"), repetition=("repetition", "first"),
                              start_frame=("frame", "min"), end_frame=("frame", "max")).reset_index(drop=True)
    if not keep_dont_care:
        seg = seg[seg["action_id"] != 0].reset_index(drop=True)
    seg["action_name"] = seg["action_id"].map(ACTION_NAMES).fillna("unknown")
    return seg


def load_labels(root: str | Path, subject: str, trial: str | None = None) -> pd.DataFrame:
    """DataFrame[trial, action_id, repetition, action_name, start_frame, end_frame] cho 1 subject
    (mọi trial nếu trial=None). Bỏ action 0 (don't care)."""
    trials = [t for s, t in list_trials(root) if s == subject and (trial is None or t == trial)]
    parts = []
    for t in trials:
        seg = labels_to_segments(load_frame_labels(root, subject, t))
        seg.insert(0, "trial", t)
        parts.append(seg)
    cols = ["trial", "action_id", "repetition", "action_name", "start_frame", "end_frame"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


# Azure Kinect Body Tracking joint index (32 joint)
KINECT_JOINTS = ["pelvis", "spine_navel", "spine_chest", "neck", "l_clavicle", "l_shoulder", "l_elbow",
                 "l_wrist", "l_hand", "l_handtip", "l_thumb", "r_clavicle", "r_shoulder", "r_elbow",
                 "r_wrist", "r_hand", "r_handtip", "r_thumb", "l_hip", "l_knee", "l_ankle", "l_foot",
                 "r_hip", "r_knee", "r_ankle", "r_foot", "head", "nose", "l_eye", "l_ear", "r_eye", "r_ear"]


def load_kinect_skeleton(root: str | Path, subject: str, trial: str,
                         joints: list[str] | None = None) -> pd.DataFrame:
    """Skeleton Azure Kinect dạng long: [frame, body_id, joint_id, joint, conf, X, Y, Z, x_color, y_color].
    Chỉ đọc body có id nhỏ nhất ở mỗi frame (1 người/trạm). joints=None → mọi joint."""
    want = None if joints is None else {KINECT_JOINTS.index(j) for j in joints}
    rows = []
    for idx, _, p in frame_files(root, subject, trial, "Skeletons"):
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        if df.empty:
            continue
        df = df[df.iloc[:, 0] == df.iloc[:, 0].min()]
        for r in df.itertuples(index=False):
            jid = int(r[1])
            if want is not None and jid not in want:
                continue
            rows.append((idx, int(r[0]), jid, KINECT_JOINTS[jid], float(r[2]), float(r[3]), float(r[4]),
                         float(r[5]), float(r[10]), float(r[11])))
    return pd.DataFrame(rows, columns=["frame", "body_id", "joint_id", "joint", "conf", "X", "Y", "Z",
                                       "x_color", "y_color"])

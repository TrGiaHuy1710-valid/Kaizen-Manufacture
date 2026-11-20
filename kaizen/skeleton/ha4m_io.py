"""I/O cho dataset HA4M — schema đã xác nhận bằng src/00_inspect.py (xem outputs/inspect_report.md).

Cấu trúc thật (khác giả định ban đầu trong CLAUDE.md):

    <root>/ID<subject><trial>/            vd: IDU001V001  (subject=U001, trial=V001 = 1 video = 1 lần lắp)
        Color/<camera_serial>/FrameID000000_DeviceTimeStamp450817700us.png   (2048x1536 RGB)
        Skeletons/<camera_serial>/FrameID000000_DeviceTimeStamp450817733us.txt (TSV, 32 joint Azure Kinect)
        Depth/ Infrared/ ColorAligned/ PointClouds_DepthGeometry/   (không dùng)
        Labels.txt   # mỗi dòng 1 frame: "<frame_idx> <action_id> <repetition>" (cách nhau dấu cách, CRLF)
                     # action_id 0 = "don't care"; repetition ≠ 0 chỉ ở action 2, 3, 11

`root` có thể là thư mục chứa nhiều IDU*V*, chính 1 thư mục IDU*V*, hoặc nhiều thư mục cách nhau dấu phẩy.
Quét phòng thủ (folder data mới có thể lệch nhẹ): mọi thư mục (độ sâu ≤ 2) có con `Color*` (không phân biệt
hoa thường, bỏ ColorAligned) là 1 video; tên IDU001V001 / U001_V001 / U001/V001 / U001 (→ V001) đều nhận;
nhãn là Labels*.txt (glob, không phân biệt hoa thường) — thiếu nhãn vẫn trích keypoints được.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

VIDEO_DIR_RE = re.compile(r"^ID(U\d+)(V\d+)$", re.I)
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


def roots_of(root: str | Path) -> list[Path]:
    """--data-root có thể là nhiều thư mục cách nhau dấu phẩy."""
    return [Path(x.strip()) for x in str(root).split(",") if x.strip()]


def _find_child(d: Path, name: str) -> Path | None:
    """Thư mục con tên `name` không phân biệt hoa thường; fallback tiền tố (bỏ *Aligned*)."""
    if not d.is_dir():
        return None
    kids = [p for p in d.iterdir() if p.is_dir()]
    low = name.lower()
    exact = [p for p in kids if p.name.lower() == low]
    if exact:
        return exact[0]
    pref = sorted(p for p in kids if p.name.lower().startswith(low) and "aligned" not in p.name.lower())
    return pref[0] if pref else None


def _ids_for(d: Path) -> tuple[str, str]:
    """(subject, trial) từ tên thư mục video, chấp nhận nhiều kiểu đặt tên lệch nhẹ."""
    n = d.name
    m = VIDEO_DIR_RE.match(n) or re.match(r"^(?:ID)?(U\d+)[_\-\s]*(V\d+)$", n, re.I)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    mp = re.match(r"^(?:ID)?(U\d+)$", d.parent.name, re.I)
    if mp and re.match(r"^V\d+$", n, re.I):
        return mp.group(1).upper(), n.upper()
    m = re.match(r"^(?:ID)?(U\d+)$", n, re.I)
    if m:
        return m.group(1).upper(), "V001"
    return re.sub(r"[^A-Za-z0-9]", "", n) or "S", "V001"


def _scan(root: str) -> dict[tuple[str, str], Path]:
    """Quét (độ sâu ≤ 2) mọi thư mục có con 'Color*' → {(subject, trial): video_dir}."""
    found: dict[tuple[str, str], Path] = {}
    for r in roots_of(root):
        if not r.exists():
            continue
        cands = [r] if _find_child(r, "Color") else []
        if not cands:
            for c in sorted(p for p in r.iterdir() if p.is_dir()):
                if _find_child(c, "Color"):
                    cands.append(c)
                else:
                    cands += [g for g in sorted(p for p in c.iterdir() if p.is_dir()) if _find_child(g, "Color")]
        for d in cands:
            key = _ids_for(d)
            if key in found and found[key].resolve() != d.resolve():
                print(f"[io] cảnh báo: trùng {key[0]}_{key[1]} ở {found[key]} và {d} — giữ cái đầu")
                continue
            found.setdefault(key, d)
    return found


_SCAN_CACHE: dict[str, dict] = {}


def _videos(root: str | Path) -> dict[tuple[str, str], Path]:
    k = str(root)
    if k not in _SCAN_CACHE:
        _SCAN_CACHE[k] = _scan(k)
    return _SCAN_CACHE[k]


def video_dir(root: str | Path, subject: str, trial: str) -> Path:
    """Thư mục của 1 video (subject='U001', trial='V001')."""
    d = _videos(root).get((subject, trial))
    if d is None:
        raise FileNotFoundError(f"Không thấy video {subject}_{trial} trong {root}")
    return d


def list_trials(root: str | Path = DEFAULT_ROOT) -> list[tuple[str, str]]:
    """[(subject, trial)] cho mọi video tìm được trong root (1 hoặc nhiều thư mục, cách nhau dấu phẩy)."""
    return sorted(_videos(root))


def modality_dir(root: str | Path, subject: str, trial: str, modality: str = "Color") -> Path:
    """<Modality>/<camera_serial>/ — tên modality không phân biệt hoa thường; tự lấy thư mục con (serial camera)."""
    vd = video_dir(root, subject, trial)
    base = _find_child(vd, modality)
    if base is None:
        raise FileNotFoundError(f"{vd} không có thư mục {modality}*")
    subs = sorted(p for p in base.iterdir() if p.is_dir())
    return subs[0] if subs else base


def frame_files(root: str | Path, subject: str, trial: str, modality: str = "Color") -> list[tuple[int, int, Path]]:
    """[(frame_idx, device_timestamp_us | -1, path)] sort theo SỐ frame (bỏ Thumbs.db v.v.).
    Tên chuẩn FrameID######_DeviceTimeStamp…us; tên khác → lấy số cuối trong tên làm frame_idx, timestamp = -1."""
    d = modality_dir(root, subject, trial, modality)
    want_img = modality.lower() != "skeletons"
    out = []
    for p in d.iterdir():
        if not p.is_file() or (want_img and p.suffix.lower() not in IMG_EXT):
            continue
        m = FRAME_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
        elif want_img or p.suffix.lower() == ".txt":
            nums = re.findall(r"\d+", p.stem)
            if nums:
                out.append((int(nums[-1]), -1, p))
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


def labels_path(root: str | Path, subject: str, trial: str) -> Path | None:
    """File nhãn của video: Labels*.txt (không phân biệt hoa thường); None nếu không có."""
    vd = video_dir(root, subject, trial)
    cands = sorted(p for p in vd.iterdir() if p.is_file() and p.suffix.lower() == ".txt"
                   and p.name.lower().startswith("label"))
    return cands[0] if cands else None


def load_frame_labels(root: str | Path, subject: str, trial: str) -> pd.DataFrame:
    """Labels.txt thô: DataFrame[frame, action_id, repetition] (1 dòng / frame). FileNotFoundError nếu thiếu."""
    p = labels_path(root, subject, trial)
    if p is None:
        raise FileNotFoundError(f"không có Labels*.txt cho {subject}_{trial}")
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

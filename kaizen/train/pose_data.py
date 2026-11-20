"""Chuẩn bị dữ liệu fine-tune pose (RTMPose, top-down COCO-17) từ HA4M: ảnh Color + skeleton Azure Kinect làm NHÃN GIẢ.

Kinect Body Tracking chiếu 32 khớp lên ảnh Color (cột X2DColor/Y2DColor); 17 khớp COCO có cùng tên trong
KINECT_JOINTS (nose, mắt, tai, vai, khuỷu, cổ tay, hông, gối, cổ chân) → ánh xạ trực tiếp theo tên.
- visibility COCO: confidence Kinect ≥ min_conf → 2, > 0 → 1, còn lại hoặc ngoài ảnh → 0 (x = y = 0)
- bbox = khung các khớp nhìn thấy, nới `margin` (thêm phía đầu); ảnh lưu là CROP quanh bbox (nới thêm
  `crop_margin`) dạng JPEG → nhỏ hơn ảnh gốc 2048×1536 PNG nhiều lần; toạ độ annotation theo ảnh crop.
- Chia train/val THEO SUBJECT. Ra: <out>/images/{train,val}/*.jpg, <out>/annotations/person_keypoints_{train,val}.json
  (COCO), <out>/stats.json.
Lưu ý: nhãn giả mang sai số của Kinect (nhất là cổ tay khi bị che) — model fine-tune sẽ học cả sai số đó.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from kaizen.skeleton import ha4m_io
from kaizen.skeleton.pose import COCO17

COCO_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle"]
COCO_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                 [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
KINECT_IDX = [ha4m_io.KINECT_JOINTS.index(n) for n in COCO17]


def read_kinect_file(path: Path) -> np.ndarray | None:
    """(32, 3) [x_color, y_color, conf] của body có id nhỏ nhất trong 1 file skeleton; None nếu rỗng."""
    try:
        rows = np.loadtxt(path, skiprows=1, ndmin=2)
    except (ValueError, OSError):
        return None
    if rows.size == 0:
        return None
    body = rows[rows[:, 0] == rows[:, 0].min()]
    out = np.zeros((len(ha4m_io.KINECT_JOINTS), 3))
    for r in body:
        j = int(r[1])
        if 0 <= j < len(out):
            out[j] = (r[10], r[11], r[2])
    return out


def coco_keypoints(kin: np.ndarray, w: int, h: int, min_conf: float) -> np.ndarray:
    """(17, 3) [x, y, v] theo ảnh gốc."""
    k = kin[KINECT_IDX].copy()
    v = np.where(k[:, 2] >= min_conf, 2, np.where(k[:, 2] > 0, 1, 0))
    inside = (k[:, 0] >= 0) & (k[:, 0] < w) & (k[:, 1] >= 0) & (k[:, 1] < h)
    v = np.where(inside, v, 0)
    k[:, 2] = v
    k[v == 0, :2] = 0
    return k


def person_bbox(k: np.ndarray, w: int, h: int, margin: float) -> tuple[float, float, float, float] | None:
    vis = k[k[:, 2] > 0]
    if len(vis) < 2:
        return None
    x0, y0 = vis[:, 0].min(), vis[:, 1].min()
    x1, y1 = vis[:, 0].max(), vis[:, 1].max()
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - bw * margin / 2, x1 + bw * margin / 2
    y0, y1 = y0 - bh * margin, y1 + bh * margin / 2          # nới nhiều hơn phía đầu
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def _split_subjects(subjects: list[str], val_subjects: list[str] | None, val_frac: float, seed: int):
    from kaizen.train.dataset import split_subjects
    return split_subjects(subjects, val_subjects, val_frac, seed)


def prepare(data_root: str, out: str | Path, subjects: set[str] | None = None, val_subjects: list[str] | None = None,
            val_frac: float = 0.15, stride: int = 15, min_conf: float = 0.5, min_visible: int = 6,
            margin: float = 0.15, crop_margin: float = 0.25, jpeg_quality: int = 90, max_per_trial: int = 0,
            seed: int = 0, log=print) -> dict:
    """Xuất COCO keypoints từ HA4M (xem docstring module). Trả stats."""
    out = Path(out)
    trials = [(s, t) for s, t in ha4m_io.list_trials(data_root) if not subjects or s in subjects]
    tr_s, va_s = _split_subjects([s for s, _ in trials], val_subjects, val_frac, seed)
    split_of = {s: "train" for s in tr_s} | {s: "val" for s in va_s}
    coco = {sp: {"images": [], "annotations": [], "categories": [{
        "id": 1, "name": "person", "supercategory": "person", "keypoints": COCO_NAMES, "skeleton": COCO_SKELETON}]}
        for sp in ("train", "val")}
    stats = {"trials": 0, "skipped_no_skeleton": [], "frames": {"train": 0, "val": 0}, "skipped_frames": 0,
             "vis2_rate": {}, "train_subjects": tr_s, "val_subjects": va_s, "stride": stride, "min_conf": min_conf}
    vis_count = np.zeros((2, 17))
    n_ok = 0
    for s, t in trials:
        try:
            sk = {i: p for i, _, p in ha4m_io.frame_files(data_root, s, t, "Skeletons")}
        except FileNotFoundError:
            stats["skipped_no_skeleton"].append(f"{s}_{t}")
            continue
        color = [(i, p) for i, _, p in ha4m_io.frame_files(data_root, s, t, "Color") if i in sk][::stride]
        if max_per_trial:
            color = color[:max_per_trial]
        sp = split_of.get(s, "train")
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        stats["trials"] += 1
        for fi, cp in color:
            kin = read_kinect_file(sk[fi])
            img = cv2.imread(str(cp)) if kin is not None else None
            if img is None:
                stats["skipped_frames"] += 1
                continue
            h, w = img.shape[:2]
            k = coco_keypoints(kin, w, h, min_conf)
            box = person_bbox(k, w, h, margin)
            if box is None or int((k[:, 2] == 2).sum()) < min_visible:
                stats["skipped_frames"] += 1
                continue
            bx0, by0, bx1, by1 = box
            cw, ch = (bx1 - bx0) * crop_margin, (by1 - by0) * crop_margin
            cx0, cy0 = int(max(0, bx0 - cw)), int(max(0, by0 - ch))
            cx1, cy1 = int(min(w, bx1 + cw)), int(min(h, by1 + ch))
            crop = img[cy0:cy1, cx0:cx1]
            name = f"{sp}/{s}_{t}_{fi:06d}.jpg"
            cv2.imwrite(str(out / "images" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            kk = k.copy()
            vis = kk[:, 2] > 0
            kk[vis, 0] -= cx0
            kk[vis, 1] -= cy0
            img_id = len(coco[sp]["images"]) + 1
            coco[sp]["images"].append({"id": img_id, "file_name": name, "width": cx1 - cx0, "height": cy1 - cy0,
                                       "subject": s, "trial": t, "frame": int(fi)})
            bbox = [bx0 - cx0, by0 - cy0, bx1 - bx0, by1 - by0]
            coco[sp]["annotations"].append({
                "id": img_id, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                "keypoints": [round(float(v), 2) for v in kk.reshape(-1)], "num_keypoints": int(vis.sum()),
                "bbox": [round(float(v), 2) for v in bbox], "area": round(float(bbox[2] * bbox[3]), 1)})
            vis_count[0] += (k[:, 2] == 2)
            vis_count[1] += 1
            stats["frames"][sp] += 1
            n_ok += 1
        log(f"[pose] {s}_{t} ({sp}): {stats['frames']['train'] + stats['frames']['val']} ảnh tích luỹ")
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val"):
        (out / "annotations" / f"person_keypoints_{sp}.json").write_text(json.dumps(coco[sp]), encoding="utf-8")
    stats["vis2_rate"] = {n: round(float(vis_count[0][i] / max(vis_count[1][i], 1)), 3) for i, n in enumerate(COCO17)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[pose] xong: train {stats['frames']['train']} · val {stats['frames']['val']} ảnh → {out}")
    return stats

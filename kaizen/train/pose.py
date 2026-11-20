"""Fine-tune pose (RTMPose) trên HA4M — phần chạy được trong env chính (không cần mmpose):

    python -m kaizen.train.pose prepare --out data/pose_ft --stride 15 --val-subjects U040,U041
        → ảnh crop + COCO keypoints từ skeleton Kinect (nhãn giả), chia theo subject
    python -m kaizen.train.pose eval --pose-model mediapipe --subjects U040,U041
        → so .npz đã trích (data/keypoints/<stem>_<pose>.npz) với Kinect: PCK@0.05/0.1, sai số px từng khớp

Phần train thật chạy bằng mmpose trong env RIÊNG (config: kaizen/train/pose_configs/), export ONNX bằng mmdeploy,
rồi dùng lại trong pipeline qua backend `rtmpose-custom` (KAIZEN_POSE_ONNX=<file.onnx>). Lệnh đầy đủ: docs/training.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from kaizen.core.config import DEFAULT_DATA_ROOT
from kaizen.skeleton import ha4m_io
from kaizen.skeleton.pose import COCO17
from kaizen.train.pose_data import coco_keypoints, prepare, read_kinect_file


def _subset(s: str) -> set[str] | None:
    v = {x.strip() for x in s.split(",") if x.strip()}
    return v or None


def eval_npz(data_root: str, npz: Path, stride: int = 5, min_conf: float = 0.5) -> dict | None:
    """Sai số từng khớp COCO-17 của 1 .npz so với Kinect (chỉ frame có cả hai). None nếu thiếu skeleton."""
    from kaizen.train.features import load_npz
    s, t = npz.name.split("_")[:2]
    try:
        sk = {i: p for i, _, p in ha4m_io.frame_files(data_root, s, t, "Skeletons")}
    except FileNotFoundError:
        return None
    kp = load_npz(npz)
    names, W, H = kp["names"], int(kp["meta"].get("width", 2048)), int(kp["meta"].get("height", 1536))
    cols = [names.index(n) if n in names else None for n in COCO17]
    errs = [[] for _ in COCO17]
    hits = {a: [[] for _ in COCO17] for a in (0.05, 0.1)}
    n_frames = 0
    for row, fi in list(enumerate(kp["frame"]))[::stride]:
        if int(fi) not in sk:
            continue
        kin = read_kinect_file(sk[int(fi)])
        if kin is None:
            continue
        gt = coco_keypoints(kin, W, H, min_conf)
        vis = gt[gt[:, 2] == 2]
        if len(vis) < 4:
            continue
        diag = float(np.hypot(*(vis[:, :2].max(0) - vis[:, :2].min(0))))
        n_frames += 1
        for j, c in enumerate(cols):
            if gt[j, 2] != 2:
                continue
            p = kp["kpts"][row, c] if c is not None else np.array([np.nan, np.nan, 0.0])
            ok = c is not None and not np.isnan(p[0]) and p[2] >= 0.3
            e = float(np.hypot(p[0] - gt[j, 0], p[1] - gt[j, 1])) if ok else np.inf
            if ok:
                errs[j].append(e)
            for a in hits:
                hits[a][j].append(e <= a * diag)
    per = {}
    for j, n in enumerate(COCO17):
        per[n] = {"median_px": round(float(np.median(errs[j])), 1) if errs[j] else None,
                  "pck@0.05": round(float(np.mean(hits[0.05][j]) * 100), 1) if hits[0.05][j] else None,
                  "pck@0.1": round(float(np.mean(hits[0.1][j]) * 100), 1) if hits[0.1][j] else None,
                  "n": len(hits[0.1][j])}
    allh = {a: [h for j in range(17) for h in hits[a][j]] for a in hits}
    return {"trial": f"{s}_{t}", "frames": n_frames, "per_joint": per,
            "pck@0.05": round(float(np.mean(allh[0.05]) * 100), 1) if allh[0.05] else None,
            "pck@0.1": round(float(np.mean(allh[0.1]) * 100), 1) if allh[0.1] else None}


def cmd_prepare(a) -> int:
    stats = prepare(a.data_root, a.out, _subset(a.subjects), list(_subset(a.val_subjects) or []), a.val_frac,
                    a.stride, a.min_conf, a.min_visible, a.margin, a.crop_margin, a.jpeg_quality, a.max_per_trial,
                    a.seed)
    if stats["frames"]["train"] == 0:
        print("[pose] ⚠ không có ảnh train nào — kiểm tra thư mục Skeletons/ trong dataset", file=sys.stderr)
        return 1
    print(f"[pose] tỷ lệ khớp Kinect tin cậy (v=2): {json.dumps(stats['vis2_rate'])}")
    return 0


def cmd_eval(a) -> int:
    kp_dir = Path(a.keypoints_dir)
    rows = []
    for s, t in ha4m_io.list_trials(a.data_root):
        if _subset(a.subjects) and s not in _subset(a.subjects):
            continue
        npz = kp_dir / f"{s}_{t}_{a.pose_model}.npz"
        if not npz.exists():
            continue
        r = eval_npz(a.data_root, npz, a.stride, a.min_conf)
        if r:
            rows.append(r)
            print(f"[pose] {r['trial']}: PCK@0.05 {r['pck@0.05']} · PCK@0.1 {r['pck@0.1']} · {r['frames']} frame · "
                  f"cổ tay phải {r['per_joint']['r_wrist']['median_px']} px · trái {r['per_joint']['l_wrist']['median_px']} px")
    if not rows:
        print(f"[pose] không có .npz {a.pose_model} nào kèm Skeletons — trích trước: "
              f"python src/01_extract.py --model {a.pose_model} --subjects ...", file=sys.stderr)
        return 1
    summ = {"pose_model": a.pose_model, "trials": len(rows),
            "pck@0.05": round(float(np.mean([r["pck@0.05"] for r in rows])), 1),
            "pck@0.1": round(float(np.mean([r["pck@0.1"] for r in rows])), 1),
            "median_px": {n: (round(float(np.median([r["per_joint"][n]["median_px"] for r in rows
                                                    if r["per_joint"][n]["median_px"] is not None])), 1)
                              if any(r["per_joint"][n]["median_px"] is not None for r in rows) else None)
                          for n in COCO17}}
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summ, "per_trial": rows}, ensure_ascii=False, indent=2), "utf-8")
        print(f"[pose] đã ghi {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="xuất COCO keypoints (nhãn giả từ Kinect) để fine-tune RTMPose")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", default="data/pose_ft")
    p.add_argument("--subjects", default="")
    p.add_argument("--val-subjects", default="")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--stride", type=int, default=15, help="lấy 1/N frame (frame liền nhau gần như trùng nhau)")
    p.add_argument("--min-conf", type=float, default=0.5, help="confidence Kinect tối thiểu để coi khớp là tin cậy")
    p.add_argument("--min-visible", type=int, default=6, help="bỏ frame có ít hơn N khớp tin cậy")
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--crop-margin", type=float, default=0.25)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--max-per-trial", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("eval", help="so keypoints đã trích với skeleton Kinect (PCK, sai số px)")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--keypoints-dir", default="data/keypoints")
    p.add_argument("--pose-model", default="mediapipe", help="mediapipe | rtmw | rtmpose-custom (hậu tố .npz)")
    p.add_argument("--subjects", default="", help="nên là subject KHÔNG dùng để fine-tune")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--out", default="")
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

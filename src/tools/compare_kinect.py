"""Đối chứng cổ tay MediaPipe/RTMW với skeleton Azure Kinect (toạ độ 2D trên ảnh Color).

    python src/tools/compare_kinect.py --subject U001 --trial V001 [--model mediapipe]
Ra outputs/kinect_compare_<subject>_<trial>.json: sai số pixel (median/P90) và % frame lệch < 50px cho mỗi cổ tay,
tính trên frame cả hai nguồn đều có điểm (Kinect conf > 0, keypoint score ≥ 0.3).
Lưu ý: Kinect body tracking cũng là ước lượng (từ depth), không phải ground truth tuyệt đối.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ha4m_io  # noqa: E402

PAIRS = {"left": ("l_wrist", "l_wrist"), "right": ("r_wrist", "r_wrist")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=ha4m_io.DEFAULT_ROOT)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--model", default="mediapipe")
    ap.add_argument("--kp-dir", default="data/keypoints")
    a = ap.parse_args()

    d = np.load(Path(a.kp_dir) / f"{a.subject}_{a.trial}_{a.model}.npz")
    names = [str(n) for n in d["names"]]
    frame = d["frame"] if "frame" in d.files else np.arange(len(d["t"]))
    sk = ha4m_io.load_kinect_skeleton(a.data_root, a.subject, a.trial, joints=["l_wrist", "r_wrist"])
    res = {"subject": a.subject, "trial": a.trial, "model": a.model,
           "note": "Kinect body tracking cũng là ước lượng; dùng để đối chứng, không phải ground truth"}
    for hand, (kn, mn) in PAIRS.items():
        ks = sk[(sk["joint"] == kn) & (sk["conf"] > 0)].set_index("frame")
        k = d["kpts"][:, names.index(mn)]
        err = []
        for i, f in enumerate(frame):
            if f not in ks.index or np.isnan(k[i, 0]) or k[i, 2] < 0.3:
                continue
            row = ks.loc[f]
            err.append(float(np.hypot(k[i, 0] - row["x_color"], k[i, 1] - row["y_color"])))
        e = np.array(err)
        res[hand] = {"n_frames": int(len(e)),
                     "median_px": round(float(np.median(e)), 1) if len(e) else None,
                     "p90_px": round(float(np.percentile(e, 90)), 1) if len(e) else None,
                     "pct_within_50px": round(float(100 * np.mean(e < 50)), 1) if len(e) else None}
        print(f"[kinect] {hand}: {res[hand]}")
    out = Path("outputs") / f"kinect_compare_{a.subject}_{a.trial}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kinect] đã ghi {out}")


if __name__ == "__main__":
    main()

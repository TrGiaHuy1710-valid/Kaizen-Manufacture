"""[02] keypoints (.npz) → outputs/features/<subject>_<trial>.parquet (1 dòng / frame).

Cột: t, frame, lw_x, lw_y, rw_x, rw_y, lw_speed, rw_speed (px/s), lw_zone, rw_zone, lw_score, rw_score,
lw_interp, rw_interp, interp, cùng các điểm phụ cho Muri (l_elbow_*, r_elbow_*, l_mcp_*, r_mcp_*).

Xử lý cổ tay: score < score_min → NaN; điểm gai (nhảy vào và ra > max_jump_px_s) → NaN; nội suy tuyến tính đoạn NaN ≤ max_gap_frames (cờ *_interp);
đoạn dài hơn giữ NaN (không tính quãng đường); lọc One Euro trên từng đoạn liên tục
(fallback Savitzky–Golay); speed = |Δp|·fps sau lọc; zone = point-in-polygon (shapely), "none" nếu ngoài.

    python src/02_features.py --model mediapipe [--subjects U001] [--force]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.core.cli import base_parser, setup, stage_dir  # noqa: E402
from kaizen.core.config import load_config  # noqa: E402
from kaizen.core.io import can_write, select_stems, stem  # noqa: E402
from kaizen.skeleton import ha4m_io  # noqa: E402
from kaizen.skeleton.features import (AUX, WRISTS, assign_zones, build_features, compute_features,  # noqa: E402,F401
                                      interp_short_gaps, plot_speed, reject_spikes, runs, smooth)

TAG = "02"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--model", default="mediapipe")
    ap.add_argument("--kp-dir", default="data/keypoints")
    ap.add_argument("--out-dir", default=None, help="mặc định <run>/features")
    ap.add_argument("--any-root", action="store_true",
                    help="xử lý mọi .npz trong --kp-dir, kể cả trial không thuộc --data-root")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    out_dir = stage_dir(ctx, a.out_dir, "features", "outputs/features")
    cfg = load_config(a.config)
    suffix = f"_{a.model}.npz"
    stems = [p.name[: -len(suffix)] for p in Path(a.kp_dir).glob(f"*{suffix}") if p.name.count("_") >= 2]
    in_root = {stem(s, t) for s, t in ha4m_io.list_trials(a.data_root)}
    if in_root and not a.any_root:
        skipped = sorted(set(stems) - in_root)
        stems = [s for s in stems if s in in_root]
        if skipped:
            print(f"[{TAG}] bỏ qua {len(skipped)} .npz không thuộc data_root {a.data_root}: {skipped[:5]}"
                  f"{' ...' if len(skipped) > 5 else ''} (--any-root để lấy hết)")
    stems = select_stems(stems, a.subjects, a.trials)
    if not stems:
        print(f"[{TAG}] không có file *{suffix} nào khớp trong {a.kp_dir}", file=sys.stderr)
        sys.exit(1)
    done = {}
    for st in stems:
        out = out_dir / f"{st}.parquet"
        if not can_write(out, a.force, TAG):
            continue
        df, stats = build_features(Path(a.kp_dir) / f"{st}{suffix}", cfg)
        df.to_parquet(out, index=False)
        plot_speed(df, out_dir / f"{st}_speed.png", cfg["moving_speed_px_s"], f"{st} — tốc độ cổ tay")
        for p in WRISTS:
            v = stats[f"{p}_valid"] * 100
            warn = "  ⚠ CẢNH BÁO < 85%" if v < 85 else ""
            print(f"[{TAG}] {st} {p}: hợp lệ {stats[f'{p}_valid_raw'] * 100:.1f}% (score≥ngưỡng), loại {stats[f'{p}_spikes']} điểm gai "
                  f"→ {v:.1f}% sau nội suy{warn}")
        zc = pd.concat([df["lw_zone"], df["rw_zone"]]).value_counts()
        print(f"[{TAG}] {st}: {len(df)} frame, nội suy {df['interp'].mean() * 100:.1f}%, zone (frame·tay): {zc.to_dict()}")
        print(f"[{TAG}] đã ghi {out}")
        meta = json.loads(stats["model_meta"]) if stats["model_meta"].startswith("{") else {}
        done[st] = {"keypoints": str(Path(a.kp_dir) / f"{st}{suffix}"), "n_frames": len(df),
                    "pose_meta": {k: meta.get(k) for k in ("model", "proc_fps", "detect_rate", "roi", "hand_crop")},
                    "valid_left_pct": round(stats["lw_valid"] * 100, 1), "valid_right_pct": round(stats["rw_valid"] * 100, 1)}
    if ctx:
        ctx.set_trials(**done)
        ctx.record_stage("02_features", n_trials=len(done), model=a.model)


if __name__ == "__main__":
    main()

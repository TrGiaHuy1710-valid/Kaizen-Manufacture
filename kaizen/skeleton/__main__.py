"""CLI Skeleton Engine.

    python -m kaizen.skeleton extract "video://data/raw/test.mp4" --model mediapipe [--stride 2] [--max-frames 300]
    python -m kaizen.skeleton extract "ha4m://data/ha4m?subject=U001&trial=V001" --roi 350,250,1500,1150
    python -m kaizen.skeleton extract "frames://path/to/dir?fps=30" --model rtmw --out data/keypoints/x_rtmw.npz
    python -m kaizen.skeleton info "ha4m://data/ha4m?subject=U001&trial=V001"
    python -m kaizen.skeleton features data/keypoints/U001_V001_mediapipe.npz --out /tmp/f.parquet [--events /tmp/e.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kaizen.core.config import DEFAULT_CONFIG, load_config
from kaizen.skeleton.events import detect_events
from kaizen.skeleton.extract import extract, npz_path_for, parse_roi
from kaizen.skeleton.features import compute_features
from kaizen.skeleton.pose import POSE_BACKENDS
from kaizen.skeleton.sources import open_source


def main(argv: list[str] | None = None) -> int:
    """Entry point `python -m kaizen.skeleton`."""
    ap = argparse.ArgumentParser(prog="python -m kaizen.skeleton", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="URI nguồn → .npz keypoints (schema cũ)")
    e.add_argument("uri")
    e.add_argument("--model", choices=sorted(POSE_BACKENDS), default="mediapipe")
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--max-frames", type=int, default=0)
    e.add_argument("--out", default="data/keypoints", help="thư mục (→ <stem>_<model>.npz) hoặc đường dẫn .npz")
    e.add_argument("--name", default="", help="stem output (mặc định suy từ nguồn)")
    e.add_argument("--roi", default="", help="x,y,w,h")
    e.add_argument("--hand-crop", choices=["auto", "on", "off"], default="auto")
    e.add_argument("--rtmw-mode", choices=["lightweight", "balanced", "performance"], default="balanced")
    e.add_argument("--device", default="cpu")
    e.add_argument("--preview", action="store_true")
    e.add_argument("--force", action="store_true")
    i = sub.add_parser("info", help="in meta + fps của nguồn")
    i.add_argument("uri")
    f = sub.add_parser("features", help=".npz → features.parquet (+ events.csv)")
    f.add_argument("npz")
    f.add_argument("--config", default=DEFAULT_CONFIG)
    f.add_argument("--out", required=True)
    f.add_argument("--events", default="")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        s = open_source(a.uri)
        print(json.dumps({**s.meta(), "fps": s.fps()}, ensure_ascii=False, indent=2))
        return 0
    if a.cmd == "features":
        cfg = load_config(a.config)
        df = compute_features(a.npz, cfg)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(a.out, index=False)
        print(f"[skeleton] {len(df)} frame → {a.out}")
        if a.events:
            ev = detect_events(df, cfg)
            ev.to_csv(a.events, index=False)
            print(f"[skeleton] {len(ev)} event → {a.events}")
        return 0
    try:
        s = open_source(a.uri)
    except (FileNotFoundError, ValueError) as ex:
        print(f"[skeleton] {ex}", file=sys.stderr)
        return 1
    out = npz_path_for(s, a.model, a.out, a.name)
    if out.exists() and not a.force:
        print(f"[skeleton] bỏ qua {out} (đã có, dùng --force)")
        return 0
    hc = None if a.hand_crop == "auto" else a.hand_crop == "on"
    extract(s, a.model, a.stride, out, max_frames=a.max_frames, roi=parse_roi(a.roi), hand_crop=hc,
            preview=a.preview, rtmw_mode=a.rtmw_mode, device=a.device, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

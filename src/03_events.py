"""[03] features → outputs/events/<subject>_<trial>.csv: start_s, end_s, hand, zone, motion, dur_s, event_type.

Mỗi tay là chuỗi trạng thái (zone, motion), motion ∈ {moving, still, lost}:
  moving nếu speed ≥ moving_speed_px_s, still nếu nhỏ hơn, lost nếu cổ tay NaN (mất dấu dài).
Gộp frame liên tiếp cùng trạng thái → event; event < min_event_s bị nhập vào hàng xóm dài hơn
(lặp tới khi hết), rồi gộp lại các event kề nhau cùng trạng thái.

Rule suy diễn event_type (đơn giản, có chủ đích):
  zone khay (không phải assembly*/done*) + still ≤ pick_max_still_s  → "pick"
  zone khay + still dài hơn                                           → "search"
  zone khay + moving                                                  → "reach"
  zone assembly*  (moving hoặc still)                                 → "assemble"
  zone done*                                                          → "place"
  none + still > idle_still_s                                         → "idle"
  none + still ngắn hơn                                               → "pause"
  none + moving                                                       → "transfer"
  lost                                                                → "lost"

    python src/03_events.py [--subjects U001] [--print-timeline] [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.core.cli import base_parser, setup, stage_dir  # noqa: E402
from kaizen.core.config import load_config  # noqa: E402
from kaizen.core.io import can_write, parse_stem, select_stems  # noqa: E402
from kaizen.skeleton import ha4m_io  # noqa: E402
from kaizen.skeleton.events import (HANDS, TL_CHAR, absorb_short, ascii_timeline, build_events,  # noqa: E402,F401
                                    classify, detect_events, rle, zone_kind)

TAG = "03"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--in-dir", default=None, help="mặc định <run>/features")
    ap.add_argument("--out-dir", default=None, help="mặc định <run>/events")
    ap.add_argument("--print-timeline", action="store_true")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    in_dir = stage_dir(ctx, a.in_dir, "features", "outputs/features")
    out_dir = stage_dir(ctx, a.out_dir, "events", "outputs/events")
    cfg = load_config(a.config)
    stems = select_stems([p.stem for p in in_dir.glob("*.parquet")], a.subjects, a.trials)
    if not stems:
        print(f"[{TAG}] không có features trong {in_dir}", file=sys.stderr)
        sys.exit(1)
    for st in stems:
        out = out_dir / f"{st}.csv"
        if can_write(out, a.force, TAG):
            df = pd.read_parquet(in_dir / f"{st}.parquet")
            ev = build_events(df, cfg)
            ev.to_csv(out, index=False)
            cnt = ev.groupby(["hand", "event_type"]).size().unstack(fill_value=0)
            print(f"[{TAG}] {st}: {len(ev)} event\n{cnt.to_string()}")
            print(f"[{TAG}] đã ghi {out}")
        if a.print_timeline:
            ev = pd.read_csv(out)
            oracle = None
            try:
                sub, tri = parse_stem(st)
                seg = ha4m_io.load_labels(a.data_root, sub, tri)
                oracle = pd.DataFrame({"step_id": seg["action_id"], "start_s": seg["start_frame"] / cfg["fps"],
                                       "end_s": (seg["end_frame"] + 1) / cfg["fps"]})
            except Exception:
                pass
            print(ascii_timeline(ev, float(ev["end_s"].max()), 80, oracle))
    if ctx:
        ctx.record_stage("03_events", n_trials=len(stems))


if __name__ == "__main__":
    main()

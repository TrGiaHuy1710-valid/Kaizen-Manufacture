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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, load_config, parse_stem, select_stems, setup, stage_dir  # noqa: E402
import ha4m_io  # noqa: E402

TAG = "03"
HANDS = {"lw": "left", "rw": "right"}
TL_CHAR = {"pick": "P", "search": "S", "reach": "r", "assemble": "A", "place": "D", "idle": "_",
           "pause": ",", "transfer": "-", "lost": " "}


def zone_kind(zone: str) -> str:
    if zone == "none":
        return "none"
    if zone.startswith("assembly"):
        return "assembly"
    if zone.startswith("done"):
        return "done"
    return "parts"


def classify(zone: str, motion: str, dur: float, cfg: dict) -> str:
    """event_type theo rule trong docstring."""
    if motion == "lost":
        return "lost"
    k = zone_kind(zone)
    if k == "parts":
        if motion == "moving":
            return "reach"
        return "pick" if dur <= cfg.get("pick_max_still_s", 2.0) else "search"
    if k == "assembly":
        return "assemble"
    if k == "done":
        return "place"
    if motion == "moving":
        return "transfer"
    return "idle" if dur > cfg.get("idle_still_s", 1.5) else "pause"


def rle(states: list, t: np.ndarray, dt: float) -> list[list]:
    """[[state, start_s, end_s], ...] từ chuỗi trạng thái theo frame."""
    out = []
    for i, s in enumerate(states):
        if out and out[-1][0] == s:
            out[-1][2] = t[i] + dt
        else:
            out.append([s, t[i], t[i] + dt])
    return out


def absorb_short(ev: list[list], min_s: float) -> list[list]:
    """Nhập event ngắn nhất (< min_s) vào hàng xóm dài hơn, lặp lại; rồi gộp kề nhau cùng trạng thái."""
    ev = [e[:] for e in ev]
    while len(ev) > 1:
        durs = [e[2] - e[1] for e in ev]
        i = int(np.argmin(durs))
        if durs[i] >= min_s:
            break
        left = durs[i - 1] if i > 0 else -1
        right = durs[i + 1] if i + 1 < len(ev) else -1
        j = i - 1 if left >= right else i + 1
        if j < i:
            ev[j][2] = ev[i][2]
        else:
            ev[j][1] = ev[i][1]
        del ev[i]
        # gộp lại nếu hai bên giống nhau
        merged = [ev[0]]
        for e in ev[1:]:
            if e[0] == merged[-1][0]:
                merged[-1][2] = e[2]
            else:
                merged.append(e)
        ev = merged
    return ev


def build_events(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fps = float(cfg["fps"])
    thr = float(cfg["moving_speed_px_s"])
    t = df["t"].to_numpy(float)
    rows = []
    for p, hand in HANDS.items():
        sp = df[f"{p}_speed"].to_numpy(float)
        zone = df[f"{p}_zone"].astype(str).to_numpy()
        lost = np.isnan(df[f"{p}_x"].to_numpy(float))
        motion = np.where(lost, "lost", np.where(sp >= thr, "moving", "still"))
        zone = np.where(lost, "none", zone)
        states = list(zip(zone, motion))
        for (z, m), s, e in absorb_short(rle(states, t, 1.0 / fps), cfg["min_event_s"]):
            rows.append(dict(start_s=round(s, 3), end_s=round(e, 3), hand=hand, zone=z, motion=m,
                             dur_s=round(e - s, 3)))
    ev = pd.DataFrame(rows, columns=["start_s", "end_s", "hand", "zone", "motion", "dur_s"])
    ev["event_type"] = [classify(z, m, d, cfg) for z, m, d in zip(ev["zone"], ev["motion"], ev["dur_s"])]
    return ev.sort_values(["hand", "start_s"]).reset_index(drop=True)


def ascii_timeline(ev: pd.DataFrame, t_end: float, width: int = 80, oracle: pd.DataFrame | None = None) -> str:
    """Timeline ASCII: mỗi cột = t_end/width giây, ký tự = event_type chiếm nhiều nhất."""
    bins = np.linspace(0, t_end, width + 1)
    lines = [f"0s{' ' * (width - len(f'{t_end:.1f}s') - 2)}{t_end:.1f}s  (1 cột = {t_end / width:.2f}s)"]
    if oracle is not None and len(oracle):
        row = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (oracle["start_s"] < b) & (oracle["end_s"] > a)
            if not m.any():
                row.append(" ")
            else:
                ov = np.minimum(oracle.loc[m, "end_s"], b) - np.maximum(oracle.loc[m, "start_s"], a)
                sid = int(oracle.loc[m, "step_id"].to_numpy()[int(np.argmax(ov.to_numpy()))])
                row.append("0123456789ABC"[sid] if sid < 13 else "?")
        lines.append("oracle |" + "".join(row) + "|")
    for hand in ("left", "right"):
        h = ev[ev["hand"] == hand]
        row = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (h["start_s"] < b) & (h["end_s"] > a)
            if not m.any():
                row.append(" ")
                continue
            ov = (np.minimum(h.loc[m, "end_s"], b) - np.maximum(h.loc[m, "start_s"], a)).to_numpy()
            row.append(TL_CHAR.get(h.loc[m, "event_type"].to_numpy()[int(np.argmax(ov))], "?"))
        lines.append(f"{hand:>6} |" + "".join(row) + "|")
    lines.append("chú thích: P=pick S=search r=reach A=assemble D=place _=idle ,=pause -=transfer ' '=lost; "
                 "oracle: số bước (A=10,B=11,C=12)")
    return "\n".join(lines)


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

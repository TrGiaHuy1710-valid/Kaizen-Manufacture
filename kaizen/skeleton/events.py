"""Events từ features (stage 03): start_s, end_s, hand, zone, motion, dur_s, event_type.

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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kaizen.core.config import StationLike, as_cfg

TAG = "03"
HANDS = {"lw": "left", "rw": "right"}
TL_CHAR = {"pick": "P", "search": "S", "reach": "r", "assemble": "A", "place": "D", "idle": "_",
           "pause": ",", "transfer": "-", "lost": " "}


def zone_kind(zone: str) -> str:
    """Tên zone → none | assembly | done | parts (khay)."""
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
    """features → events (2 tay), gộp + lọc event ngắn + gán event_type."""
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


def detect_events(features: pd.DataFrame, station: StationLike) -> pd.DataFrame:
    """features (1 dòng / frame) → events.csv DataFrame (core.schemas.EVENT_COLUMNS)."""
    return build_events(features, as_cfg(station))

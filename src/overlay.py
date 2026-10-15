"""Vẽ overlay lên frame (dùng chung cho tools/render_demo.py và UI) + bảng màu bước cố định.

Màu bước cố định theo step_id (Gantt, Yamazumi, banner video, zone khay dùng cùng một bảng):
6 sắc độ của bảng màu tham chiếu × 2 độ đậm (bước lẻ đậm, bước chẵn nhạt); bước 0/unclassified xám.
Zone khay lấy màu của bước đầu tiên dùng zone đó trong station.yaml → nhìn video thấy ngay khay của bước nào.
Overlay chỉ HIỂN THỊ số đã có trong artifact (features/steps/events); đồng hồ bước/chu kỳ = t − mốc bắt đầu.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_BASE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
GRAY = "#8d8c86"
ASSEMBLY = "#52514e"
HAND_STATE = {"active": ("hoạt động", "#1baf7a"), "idle": ("chờ", "#eda100"), "lost": ("mất dấu", "#8d8c86")}


def _mix(hex_c: str, w: float) -> str:
    r, g, b = (int(hex_c[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c + (255 - c) * w) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def step_color(step_id: int) -> str:
    """Hex màu cố định của 1 bước SOP (0 = unclassified → xám)."""
    sid = int(step_id)
    if sid <= 0:
        return GRAY
    base = _BASE[((sid - 1) // 2) % len(_BASE)]
    return base if sid % 2 == 1 else _mix(base, 0.45)


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


def zone_colors(cfg: dict) -> dict[str, str]:
    """Màu zone: zone khay = màu bước đầu tiên dùng nó; assembly = xám đậm; zone không gán bước = xám."""
    first = {}
    for s in cfg.get("steps", []):
        z = s.get("zone")
        if z and z not in first:
            first[z] = int(s["id"])
    out = {}
    for z in cfg.get("zones", {}):
        if z.startswith("assembly"):
            out[z] = ASSEMBLY
        else:
            out[z] = step_color(first[z]) if z in first else GRAY
    return out


@lru_cache(maxsize=4)
def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    import matplotlib
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    try:
        return ImageFont.truetype(str(p), size)
    except OSError:
        return ImageFont.load_default()


@dataclass
class OverlayData:
    cfg: dict
    feat: pd.DataFrame | None
    steps: pd.DataFrame | None
    events: pd.DataFrame | None
    source: str = ""

    @property
    def cycle(self) -> tuple[float, float] | None:
        if self.steps is None or self.steps.empty:
            return None
        v = self.steps[self.steps["step_id"] > 0]
        v = v if len(v) else self.steps
        return float(v["start_s"].min()), float(v["end_s"].max())


def load_overlay_data(run_root: Path, trial: str, cfg: dict, source: str) -> OverlayData:
    """Đọc features/steps/events của 1 trial trong run (steps ưu tiên nguồn của run, fallback nguồn còn lại)."""
    run_root = Path(run_root)
    fp = run_root / "features" / f"{trial}.parquet"
    feat = pd.read_parquet(fp) if fp.exists() else None
    steps, used = None, ""
    for src in ([source] if source else []) + ["oracle", "pred"]:
        p = run_root / "steps" / f"{trial}_{src}.csv"
        if p.exists():
            steps, used = pd.read_csv(p), src
            break
    ep = run_root / "events" / f"{trial}.csv"
    events = pd.read_csv(ep) if ep.exists() else None
    return OverlayData(cfg, feat, steps, events, used)


def _row_at(df: pd.DataFrame | None, t: float, extra: pd.Series | None = None):
    if df is None or df.empty:
        return None
    m = (df["start_s"] <= t) & (df["end_s"] > t)
    if extra is not None:
        m &= extra
    sub = df[m]
    return None if sub.empty else sub.iloc[0]


def hand_state(events: pd.DataFrame | None, hand: str, t: float) -> str:
    if events is None:
        return "lost"
    r = _row_at(events, t, events["hand"] == hand)
    if r is None or r["event_type"] == "lost":
        return "lost"
    return "idle" if r["event_type"] in ("idle", "pause") else "active"


def draw(img: np.ndarray, t: float, od: OverlayData, scale: float = 1.0, trail_s: float = 1.0) -> np.ndarray:
    """Vẽ zone, vệt cổ tay ~trail_s giây, banner bước + đồng hồ, trạng thái tay. img đã resize theo `scale`."""
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    S = lambda pts: (np.asarray(pts, np.float64) * scale).astype(np.int32)  # noqa: E731
    th = max(1, round(2 * w / 960))

    # Zone
    zc = zone_colors(od.cfg)
    overlay = img.copy()
    for name, pts in od.cfg.get("zones", {}).items():
        if pts and len(pts) >= 3:
            cv2.fillPoly(overlay, [S(pts)], hex_to_bgr(zc.get(name, GRAY)))
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    for name, pts in od.cfg.get("zones", {}).items():
        if pts and len(pts) >= 3:
            cv2.polylines(img, [S(pts)], True, hex_to_bgr(zc.get(name, GRAY)), th)

    # Cổ tay: vệt + điểm hiện tại
    if od.feat is not None and len(od.feat):
        f = od.feat
        w_ = f[(f["t"] >= t - trail_s) & (f["t"] <= t)]
        i_now = int(np.clip(np.searchsorted(f["t"].to_numpy(), t), 0, len(f) - 1))
        for p, col, lab in (("lw", "#d1495b", "L"), ("rw", "#2e86ab", "R")):
            pts = w_[[f"{p}_x", f"{p}_y"]].dropna().to_numpy()
            if len(pts) > 1:
                cv2.polylines(img, [S(pts)], False, hex_to_bgr(col), th + 1, cv2.LINE_AA)
            x, y = f[f"{p}_x"].iloc[i_now], f[f"{p}_y"].iloc[i_now]
            if not (np.isnan(x) or np.isnan(y)):
                c = tuple(int(v) for v in S([x, y]))
                cv2.circle(img, c, 4 * th + 2, hex_to_bgr(col), -1, cv2.LINE_AA)
                cv2.circle(img, c, 4 * th + 2, (255, 255, 255), th, cv2.LINE_AA)
                cv2.putText(img, lab, (c[0] + 6 * th, c[1] - 4 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * th + 0.2,
                            (255, 255, 255), th, cv2.LINE_AA)

    # Banner (PIL để có chữ tiếng Việt)
    bh = max(34, int(h * 0.085))
    cv2.rectangle(img, (0, 0), (w, bh), (24, 24, 24), -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f_big, f_small = _font(max(12, int(bh * 0.40)), True), _font(max(10, int(bh * 0.30)))
    st = _row_at(od.steps, t)
    cyc = od.cycle
    if st is not None and int(st["step_id"]) > 0:
        sid, rep = int(st["step_id"]), int(st.get("repetition", 0))
        col = step_color(sid)
        title = f"Bước {sid}{f'.{rep}' if rep else ''} · {st['name']}"
        clock = f"bước {t - float(st['start_s']):.1f} s / {float(st['dur_s']):.1f} s"
    elif st is not None:
        col, title, clock = GRAY, "Chưa gán bước (unclassified)", f"đoạn {t - float(st['start_s']):.1f} s"
    else:
        col = GRAY
        title = "Chưa vào chu kỳ" if cyc and t < cyc[0] else ("Đã hết chu kỳ" if cyc else "Không có dữ liệu bước")
        clock = ""
    parts = [clock] if clock else []
    if cyc and cyc[0] <= t <= cyc[1]:
        parts.append(f"chu kỳ {t - cyc[0]:.1f} s / {cyc[1] - cyc[0]:.1f} s")
    parts.append(f"t = {t:.1f} s")
    if od.source:
        parts.append(f"nguồn: {od.source}")
    clock = "  ·  ".join(parts)
    pad = int(bh * 0.16)
    d.rectangle([pad, pad, pad + int(bh * 0.22), bh - pad], fill=col)
    x0 = pad * 2 + int(bh * 0.22)
    d.text((x0, pad - 2), title[:90], font=f_big, fill=(255, 255, 255))
    d.text((x0, int(bh * 0.55)), clock, font=f_small, fill=(205, 205, 200))

    # Trạng thái tay (góc phải dưới banner)
    y = bh + pad
    for hand, lab in (("left", "Tay trái"), ("right", "Tay phải")):
        txt, c = HAND_STATE[hand_state(od.events, hand, t)]
        s = f"{lab}: {txt}"
        tw = d.textlength(s, font=f_small)
        dot, ph = bh * 0.16, int(bh * 0.42)
        x0 = w - pad - tw - dot - 3 * pad
        d.rounded_rectangle([x0, y, w - pad, y + ph], radius=6, fill=(24, 24, 24))
        d.ellipse([x0 + pad, y + (ph - dot) / 2, x0 + pad + dot, y + (ph + dot) / 2], fill=c)
        d.text((x0 + 2 * pad + dot, y + bh * 0.05), s, font=f_small, fill=(255, 255, 255))
        y += int(bh * 0.5)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_frame(img_full: np.ndarray, t: float, od: OverlayData, width: int = 960) -> np.ndarray:
    """Resize frame gốc về `width` px rồi vẽ overlay (nhanh hơn vẽ ở 2048px)."""
    scale = width / img_full.shape[1]
    img = cv2.resize(img_full, (width, int(round(img_full.shape[0] * scale / 2)) * 2), interpolation=cv2.INTER_AREA)
    return draw(img, t, od, scale=scale)

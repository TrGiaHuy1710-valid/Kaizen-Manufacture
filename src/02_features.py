"""[02] keypoints (.npz) → outputs/features/<subject>_<trial>.parquet (1 dòng / frame).

Cột: t, frame, lw_x, lw_y, rw_x, rw_y, lw_speed, rw_speed (px/s), lw_zone, rw_zone, lw_score, rw_score,
lw_interp, rw_interp, interp, cùng các điểm phụ cho Muri (l_elbow_*, r_elbow_*, l_mcp_*, r_mcp_*).

Xử lý cổ tay: score < score_min → NaN; điểm gai (nhảy vào và ra > max_jump_px_s) → NaN; nội suy tuyến tính đoạn NaN ≤ max_gap_frames (cờ *_interp);
đoạn dài hơn giữ NaN (không tính quãng đường); lọc One Euro trên từng đoạn liên tục
(fallback Savitzky–Golay); speed = |Δp|·fps sau lọc; zone = point-in-polygon (shapely), "none" nếu ngoài.

    python src/02_features.py --model mediapipe [--subjects U001] [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, load_config, select_stems  # noqa: E402

TAG = "02"
WRISTS = {"lw": "l_wrist", "rw": "r_wrist"}
AUX = {"l_elbow": "l_elbow", "r_elbow": "r_elbow", "l_mcp": "lh_middle_1", "r_mcp": "rh_middle_1"}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Các đoạn [s, e) liên tiếp mask=True."""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def reject_spikes(x: np.ndarray, y: np.ndarray, fps: float, max_jump: float) -> np.ndarray:
    """Mask điểm gai: tốc độ vào VÀ ra khỏi điểm đều > max_jump (px/s) — pose nhảy khi bị che. Lặp tới khi hết."""
    bad = np.zeros(len(x), bool)
    for _ in range(10):
        xs, ys = np.where(bad, np.nan, x), np.where(bad, np.nan, y)
        idx = np.flatnonzero(~np.isnan(xs))
        if len(idx) < 3:
            break
        px, py = xs[idx], ys[idx]
        gap = np.diff(idx)
        v = np.hypot(np.diff(px), np.diff(py)) * fps / gap
        spike = np.zeros(len(idx), bool)
        spike[1:-1] = (v[:-1] > max_jump) & (v[1:] > max_jump)
        if not spike.any():
            break
        bad[idx[spike]] = True
    return bad


def interp_short_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Nội suy tuyến tính các đoạn NaN ≤ max_gap nằm giữa 2 điểm hợp lệ. Trả (x_mới, cờ_nội_suy)."""
    x = x.copy()
    flag = np.zeros(len(x), bool)
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return x, flag
    for s, e in runs(~ok):
        if s == 0 or e == len(x) or e - s > max_gap:
            continue
        x[s:e] = np.interp(idx[s:e], [s - 1, e], [x[s - 1], x[e]])
        flag[s:e] = True
    return x, flag


def smooth(x: np.ndarray, t: np.ndarray, fps: float, mincutoff: float, beta: float) -> np.ndarray:
    """One Euro trên từng đoạn liên tục không-NaN; fallback savgol nếu thiếu lib."""
    y = x.copy()
    try:
        from OneEuroFilter import OneEuroFilter
        for s, e in runs(~np.isnan(x)):
            f = OneEuroFilter(fps, mincutoff=mincutoff, beta=beta)
            y[s:e] = [f(float(v), float(tt)) for v, tt in zip(x[s:e], t[s:e])]
    except ImportError:
        from scipy.signal import savgol_filter
        win = max(5, int(round(0.3 * fps)) | 1)
        for s, e in runs(~np.isnan(x)):
            if e - s >= win:
                y[s:e] = savgol_filter(x[s:e], win, 2)
    return y


def assign_zones(xs: np.ndarray, ys: np.ndarray, zones: dict) -> np.ndarray:
    """Tên zone đầu tiên (theo thứ tự trong yaml) chứa điểm; 'none' nếu không có/NaN."""
    from shapely import contains_xy
    from shapely.geometry import Polygon
    out = np.full(len(xs), "none", dtype=object)
    free = ~(np.isnan(xs) | np.isnan(ys))
    for name, pts in zones.items():
        if not pts or len(pts) < 3:
            continue
        poly = Polygon(pts)
        hit = free.copy()
        hit[free] = contains_xy(poly, xs[free], ys[free])
        out[hit] = name
        free &= ~hit
    return out


def build_features(npz_path: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Đọc .npz → DataFrame features + thống kê kiểm tra."""
    d = np.load(npz_path, allow_pickle=False)
    kpts, names = d["kpts"].astype(np.float64), [str(n) for n in d["names"]]
    t = d["t"].astype(np.float64)
    frame = d["frame"] if "frame" in d.files else np.round(t * cfg["fps"]).astype(int)
    fps = float(cfg["fps"])
    fcfg = cfg.get("features", {})
    smin = fcfg.get("score_min", 0.3)
    max_gap = int(fcfg.get("max_gap_frames", 15))
    mc, beta = fcfg.get("oneeuro_mincutoff", 1.0), fcfg.get("oneeuro_beta", 0.01)

    df = pd.DataFrame({"t": t, "frame": frame})
    stats = {}
    for p, nm in WRISTS.items():
        k = kpts[:, names.index(nm)]
        score = k[:, 2]
        bad = np.isnan(score) | (score < smin)
        x, y = np.where(bad, np.nan, k[:, 0]), np.where(bad, np.nan, k[:, 1])
        stats[f"{p}_valid_raw"] = float(np.mean(~bad)) if len(bad) else 0.0
        spk = reject_spikes(x, y, fps, float(fcfg.get("max_jump_px_s", 2500)))
        x, y = np.where(spk, np.nan, x), np.where(spk, np.nan, y)
        stats[f"{p}_spikes"] = int(spk.sum())
        x, fx = interp_short_gaps(x, max_gap)
        y, _ = interp_short_gaps(y, max_gap)
        x, y = smooth(x, t, fps, mc, beta), smooth(y, t, fps, mc, beta)
        sp = np.full(len(x), np.nan)
        if len(x) > 1:
            sp[1:] = np.hypot(np.diff(x), np.diff(y)) * fps
            sp[0] = sp[1]
        df[f"{p}_x"], df[f"{p}_y"], df[f"{p}_speed"] = x, y, sp
        df[f"{p}_score"] = score
        df[f"{p}_interp"] = fx
        df[f"{p}_zone"] = assign_zones(x, y, cfg["zones"])
        stats[f"{p}_valid"] = float(np.mean(~np.isnan(x))) if len(x) else 0.0
    df["interp"] = df["lw_interp"] | df["rw_interp"]
    for col, nm in AUX.items():
        if nm in names:
            k = kpts[:, names.index(nm)]
            bad = np.isnan(k[:, 2]) | (k[:, 2] < smin)
            df[f"{col}_x"] = np.where(bad, np.nan, k[:, 0])
            df[f"{col}_y"] = np.where(bad, np.nan, k[:, 1])
    meta_raw = str(d["meta"]) if "meta" in d.files else "{}"
    stats["model_meta"] = meta_raw
    return df, stats


def plot_speed(df: pd.DataFrame, out: Path, thr: float, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(df["t"], df["lw_speed"], lw=0.8, color="#d1495b", label="cổ tay trái (lw)")
    ax.plot(df["t"], df["rw_speed"], lw=0.8, color="#2e86ab", label="cổ tay phải (rw)")
    ax.axhline(thr, color="#555", ls="--", lw=0.8, label=f"ngưỡng moving {thr:g} px/s")
    ax.set_xlabel("t (s)"); ax.set_ylabel("speed (px/s)"); ax.set_title(title)
    ax.set_ylim(0, np.nanpercentile(df[["lw_speed", "rw_speed"]].values, 99.5) * 1.1 if df["rw_speed"].notna().any() else 1)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--model", default="mediapipe")
    ap.add_argument("--kp-dir", default="data/keypoints")
    ap.add_argument("--out-dir", default="outputs/features")
    a = ap.parse_args()
    cfg = load_config(a.config)
    suffix = f"_{a.model}.npz"
    stems = select_stems([p.name[: -len(suffix)] for p in Path(a.kp_dir).glob(f"*{suffix}")
                          if p.name.count("_") >= 2], a.subjects, a.trials)
    if not stems:
        sys.exit(f"[{TAG}] không có file *{suffix} trong {a.kp_dir}")
    for st in stems:
        out = Path(a.out_dir) / f"{st}.parquet"
        if not can_write(out, a.force, TAG):
            continue
        df, stats = build_features(Path(a.kp_dir) / f"{st}{suffix}", cfg)
        df.to_parquet(out, index=False)
        plot_speed(df, Path(a.out_dir) / f"{st}_speed.png", cfg["moving_speed_px_s"], f"{st} — tốc độ cổ tay")
        for p in WRISTS:
            v = stats[f"{p}_valid"] * 100
            warn = "  ⚠ CẢNH BÁO < 85%" if v < 85 else ""
            print(f"[{TAG}] {st} {p}: hợp lệ {stats[f'{p}_valid_raw'] * 100:.1f}% (score≥ngưỡng), loại {stats[f'{p}_spikes']} điểm gai "
                  f"→ {v:.1f}% sau nội suy{warn}")
        zc = pd.concat([df["lw_zone"], df["rw_zone"]]).value_counts()
        print(f"[{TAG}] {st}: {len(df)} frame, nội suy {df['interp'].mean() * 100:.1f}%, zone (frame·tay): {zc.to_dict()}")
        print(f"[{TAG}] đã ghi {out}")


if __name__ == "__main__":
    main()

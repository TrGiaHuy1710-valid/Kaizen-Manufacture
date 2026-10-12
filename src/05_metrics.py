"""[05] steps + events + features → outputs/metrics/metrics.json (+ <stem>_steps.csv, trials.csv).

Mỗi metric là {value, unit, definition} — 07 đưa nguyên văn cho LLM.
Muda: cycle_time, dur từng bước, idle/transfer từng tay, pick theo zone, quãng đường cổ tay, %VA.
Mura: mean/std/CV cycle time + từng bước giữa trial (trong subject) và giữa subject (cần ≥ 2 mẫu).
Muri: góc cổ tay 2D (elbow→wrist vs wrist→middle_mcp), % thời gian lệch > muri_dev_deg so với trung vị cá nhân.
Eval: copy từ outputs/steps/<stem>_eval.json nếu có.

    python src/05_metrics.py --source oracle|pred [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, load_config, parse_stem, r, read_json, select_stems, write_json  # noqa: E402

TAG = "05"
HANDS = {"left": "lw", "right": "rw"}
MURI_NOTE = "ước lượng 2D, chỉ mang tính sàng lọc"


def M(value, unit: str, definition: str) -> dict:
    return {"value": value, "unit": unit, "definition": definition}


def clip_events(ev: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    """Cắt event vào cửa sổ [t0, t1]."""
    e = ev[(ev["end_s"] > t0) & (ev["start_s"] < t1)].copy()
    e["start_s"] = e["start_s"].clip(lower=t0)
    e["end_s"] = e["end_s"].clip(upper=t1)
    e["dur_s"] = e["end_s"] - e["start_s"]
    return e


def wrist_angle_dev(feat: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Góc (độ) giữa (elbow→wrist) và (wrist→middle_mcp) theo frame; NaN nếu thiếu điểm."""
    p = "lw" if side == "l" else "rw"
    need = [f"{side}_elbow_x", f"{side}_mcp_x"]
    if any(c not in feat for c in need):
        return np.full(len(feat), np.nan), feat["t"].to_numpy()
    wx, wy = feat[f"{p}_x"].to_numpy(), feat[f"{p}_y"].to_numpy()
    v1 = np.stack([wx - feat[f"{side}_elbow_x"], wy - feat[f"{side}_elbow_y"]], 1)
    v2 = np.stack([feat[f"{side}_mcp_x"] - wx, feat[f"{side}_mcp_y"] - wy], 1)
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos = np.einsum("ij,ij->i", v1, v2) / np.where(n > 0, n, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1, 1))), feat["t"].to_numpy()


def trial_metrics(st: str, steps: pd.DataFrame, ev: pd.DataFrame | None, feat: pd.DataFrame | None,
                  cfg: dict) -> tuple[dict, pd.DataFrame]:
    fps = float(cfg["fps"])
    pxcm = cfg.get("px_per_cm")
    dist_unit = "cm" if pxcm else "px"
    sd = {int(s["id"]): s for s in cfg.get("steps", [])}
    valid = steps[steps["step_id"] > 0]
    t0, t1 = float(valid["start_s"].min()), float(valid["end_s"].max())
    ct = t1 - t0
    out: dict = {"window_s": [r(t0), r(t1)]}
    out["cycle_time_s"] = M(r(ct), "s", "end bước cuối − start bước đầu (bỏ đoạn don't care/unclassified)")
    out["n_steps"] = M(int(valid["step_id"].nunique()), "count", "số bước SOP khác nhau có mặt trong chu kỳ")
    out["unclassified_s"] = M(r(steps.loc[steps["step_id"] == 0, "dur_s"].sum()), "s",
                              "thời gian không gán được bước (chỉ ở chế độ predicted)")

    # Bảng theo bước
    rows = []
    for sid, g in valid.groupby("step_id"):
        info = sd.get(int(sid), {})
        row = dict(step_id=int(sid), name=g["name"].iloc[0], va=bool(info.get("va", False)),
                   n_rep=len(g), dur_s=r(g["dur_s"].sum()), start_s=r(g["start_s"].min()), end_s=r(g["end_s"].max()),
                   pct_cycle=r(100 * g["dur_s"].sum() / ct, 1) if ct > 0 else None)
        if ev is not None:
            for hand in HANDS:
                he = ev[ev["hand"] == hand]
                idle = act = 0.0
                for s in g.itertuples():
                    e = clip_events(he, s.start_s, s.end_s)
                    idle += e.loc[e["event_type"] == "idle", "dur_s"].sum()
                    act += e.loc[~e["event_type"].isin(["idle", "lost", "pause"]), "dur_s"].sum()
                d = g["dur_s"].sum()
                row[f"{hand}_idle_pct"] = r(100 * idle / d, 1) if d > 0 else None
                row[f"{hand}_active_pct"] = r(100 * act / d, 1) if d > 0 else None
        rows.append(row)
    step_df = pd.DataFrame(rows)
    out["steps"] = {
        "unit": {"dur_s": "s", "pct_cycle": "%", "left_idle_pct": "% thời lượng bước", "right_idle_pct": "% thời lượng bước",
                 "left_active_pct": "% thời lượng bước", "right_active_pct": "% thời lượng bước"},
        "definition": {"dur_s": "tổng thời lượng các lần lặp của bước", "pct_cycle": "dur_s / cycle_time",
                       "*_idle_pct": "% thời lượng bước tay đó ở trạng thái idle (none + still > idle_still_s)",
                       "*_active_pct": "% thời lượng bước tay đó ở pick/reach/search/assemble/place/transfer",
                       "va": "bước tạo giá trị (station.yaml)"},
        "rows": step_df.to_dict("records")}
    va = step_df.loc[step_df["va"], "dur_s"].sum() if len(step_df) else 0.0
    out["va_pct"] = M(r(100 * va / ct, 1) if ct > 0 else None, "%", "Σ dur bước va=true / cycle_time")

    if ev is not None:
        e = clip_events(ev, t0, t1)
        for hand in HANDS:
            he = e[e["hand"] == hand]
            out[f"idle_{hand}_s"] = M(r(he.loc[he["event_type"] == "idle", "dur_s"].sum()), "s",
                                      f"tổng thời gian tay {hand} idle (ngoài zone, đứng yên > idle_still_s) trong chu kỳ")
            out[f"transfer_{hand}_s"] = M(r(he.loc[he["event_type"] == "transfer", "dur_s"].sum()), "s",
                                          f"tổng thời gian tay {hand} di chuyển ngoài mọi zone trong chu kỳ")
            out[f"lost_{hand}_pct"] = M(r(100 * he.loc[he["event_type"] == "lost", "dur_s"].sum() / ct, 1), "%",
                                        f"% chu kỳ mất dấu cổ tay {hand} (che khuất/ra khỏi khung)")
        both_idle = 0.0
        L, R_ = e[(e["hand"] == "left") & (e["event_type"] == "idle")], e[(e["hand"] == "right") & (e["event_type"] == "idle")]
        for a in L.itertuples():
            for b in R_.itertuples():
                both_idle += max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
        out["idle_both_s"] = M(r(both_idle), "s", "thời gian CẢ HAI tay cùng idle trong chu kỳ")
        pk = e[e["event_type"] == "pick"]
        out["picks_by_zone"] = M({z: int(n) for z, n in pk.groupby("zone").size().items()}, "count/cycle",
                                 "số event pick (khay + still ngắn) theo zone, cả hai tay, trong chu kỳ")
        out["picks_total"] = M(int(len(pk)), "count/cycle", "tổng số pick trong chu kỳ")
        out["pick_mean_dur_s"] = M({z: r(g["dur_s"].mean()) for z, g in pk.groupby("zone")}, "s",
                                   "thời lượng trung bình 1 event pick theo zone")

    if feat is not None:
        w = (feat["t"] >= t0) & (feat["t"] <= t1)
        f = feat[w]
        for hand, p in HANDS.items():
            sp = f[f"{p}_speed"].to_numpy(float)
            d = np.nansum(sp) / fps
            out[f"distance_{hand}"] = M(r(d / pxcm if pxcm else d, 1), dist_unit,
                                        f"quãng đường cổ tay {hand} trong chu kỳ (bỏ đoạn mất dấu dài)")
            out[f"valid_{hand}_pct"] = M(r(100 * np.mean(~np.isnan(f[f'{p}_x'])), 1), "%",
                                         f"% frame có cổ tay {hand} hợp lệ (sau nội suy) trong chu kỳ")
        muri = {"note": MURI_NOTE}
        for side, hand in (("l", "left"), ("r", "right")):
            ang, _ = wrist_angle_dev(f, side)
            ok = ~np.isnan(ang)
            if ok.sum() < 30:
                muri[hand] = {"available": False, "coverage_pct": r(100 * ok.mean(), 1)}
                continue
            med = float(np.median(ang[ok]))
            dev = np.abs(ang[ok] - med)
            thr = float(cfg.get("muri_dev_deg", 30))
            muri[hand] = {"available": True,
                          "coverage_pct": M(r(100 * ok.mean(), 1), "%", "% frame đủ elbow+wrist+middle_mcp để tính góc"),
                          "median_angle_deg": M(r(med, 1), "deg", "trung vị cá nhân của góc cổ tay 2D"),
                          "dev_gt_thr_pct": M(r(100 * np.mean(dev > thr), 1), "%",
                                              f"% thời gian (trong frame đo được) góc lệch > {thr:g}° so với trung vị"),
                          "dev_p90_deg": M(r(np.percentile(dev, 90), 1), "deg", "phân vị 90 độ lệch góc")}
        out["muri_wrist"] = muri
    return out, step_df


def mura(values: dict[str, float]) -> dict:
    v = np.array([x for x in values.values() if x is not None], float)
    if len(v) < 2:
        return {"n": int(len(v)), "mean": r(v.mean()) if len(v) else None, "std": None, "cv": None,
                "note": "cần ≥ 2 mẫu"}
    m, s = v.mean(), v.std(ddof=1)
    return {"n": int(len(v)), "mean": r(m), "std": r(s), "cv": r(s / m, 3) if m else None}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--source", choices=["oracle", "pred"], default="oracle")
    ap.add_argument("--steps-dir", default="outputs/steps")
    ap.add_argument("--events-dir", default="outputs/events")
    ap.add_argument("--features-dir", default="outputs/features")
    ap.add_argument("--out-dir", default="outputs/metrics")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out_dir = Path(a.out_dir)
    out_json = out_dir / "metrics.json"
    if not can_write(out_json, a.force, TAG):
        return
    suf = f"_{a.source}.csv"
    stems = select_stems([p.name[: -len(suf)] for p in Path(a.steps_dir).glob(f"*{suf}")], a.subjects, a.trials)
    if not stems:
        sys.exit(f"[{TAG}] không có steps *{suf} trong {a.steps_dir}")

    trials, summary = {}, []
    for st in stems:
        steps = pd.read_csv(Path(a.steps_dir) / f"{st}{suf}")
        if not (steps["step_id"] > 0).any():
            print(f"[{TAG}] {st}: không có bước hợp lệ, bỏ qua")
            continue
        evp, fp = Path(a.events_dir) / f"{st}.csv", Path(a.features_dir) / f"{st}.parquet"
        ev = pd.read_csv(evp) if evp.exists() else None
        feat = pd.read_parquet(fp) if fp.exists() else None
        m, step_df = trial_metrics(st, steps, ev, feat, cfg)
        evalp = Path(a.steps_dir) / f"{st}_eval.json"
        if evalp.exists():
            e = read_json(evalp)
            m["eval_vs_oracle"] = {k: v for k, v in e.items() if k != "per_step"}
            m["eval_vs_oracle"]["definition"] = "predicted vs nhãn HA4M: MAE (s), F1@IoU0.5 theo bước"
        trials[st] = m
        step_df.to_csv(out_dir / f"{st}_steps.csv", index=False)
        summary.append(dict(trial=st, cycle_time_s=m["cycle_time_s"]["value"], va_pct=m["va_pct"]["value"],
                            idle_left_s=m.get("idle_left_s", {}).get("value"),
                            idle_right_s=m.get("idle_right_s", {}).get("value"),
                            picks_total=m.get("picks_total", {}).get("value"),
                            distance_right=m.get("distance_right", {}).get("value")))
        print(f"[{TAG}] {st}: cycle {m['cycle_time_s']['value']}s, VA {m['va_pct']['value']}%, "
              f"idle L/R {m.get('idle_left_s', {}).get('value')}/{m.get('idle_right_s', {}).get('value')}s")
    pd.DataFrame(summary).to_csv(out_dir / "trials.csv", index=False)

    # Mura
    by_sub: dict[str, list[str]] = {}
    for st in trials:
        by_sub.setdefault(parse_stem(st)[0], []).append(st)
    step_ids = sorted({row["step_id"] for m in trials.values() for row in m["steps"]["rows"]})

    def step_dur(st, sid):
        return next((row["dur_s"] for row in trials[st]["steps"]["rows"] if row["step_id"] == sid), None)

    mura_out = {"definition": "mean/std (s) và CV = std/mean giữa các trial; cần ≥ 2 trial", "unit": "s (cv không đơn vị)",
                "within_subject": {}, "between_subjects": {}}
    for sub, sts in by_sub.items():
        mura_out["within_subject"][sub] = {
            "cycle_time": mura({st: trials[st]["cycle_time_s"]["value"] for st in sts}),
            "steps": {str(sid): mura({st: step_dur(st, sid) for st in sts}) for sid in step_ids}}
    sub_mean = {sub: float(np.mean([trials[st]["cycle_time_s"]["value"] for st in sts])) for sub, sts in by_sub.items()}
    mura_out["between_subjects"]["cycle_time"] = mura(sub_mean)
    all_steps = {str(sid): mura({st: step_dur(st, sid) for st in trials}) for sid in step_ids}
    mura_out["all_trials"] = {"cycle_time": mura({st: m["cycle_time_s"]["value"] for st, m in trials.items()}),
                              "steps": all_steps}
    cvs = [(sid, v["cv"]) for sid, v in all_steps.items() if v.get("cv") is not None]
    mura_out["top_cv_steps"] = [{"step_id": int(s), "cv": c} for s, c in sorted(cvs, key=lambda x: -x[1])[:5]]

    res = {"meta": {"source": a.source, "fps": cfg["fps"], "px_per_cm": cfg.get("px_per_cm"),
                    "moving_speed_px_s": cfg["moving_speed_px_s"], "n_trials": len(trials),
                    "subjects": sorted(by_sub), "muri_note": MURI_NOTE},
           "trials": trials, "mura": mura_out}
    write_json(out_json, res)
    print(f"[{TAG}] đã ghi {out_json} ({len(trials)} trial)")


if __name__ == "__main__":
    main()

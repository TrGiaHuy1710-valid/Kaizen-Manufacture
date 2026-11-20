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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.analyzer.metrics import (HANDS, MURI_NOTE, M, build_metrics_report, clip_events,  # noqa: E402,F401
                                     compute_metrics, compute_mura, mura, trial_metrics, wrist_angle_dev)
from kaizen.core.cli import base_parser, setup, stage_dir  # noqa: E402
from kaizen.core.config import load_config  # noqa: E402
from kaizen.core.io import can_write, read_json, select_stems, write_json  # noqa: E402

TAG = "05"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--source", choices=["oracle", "pred"], default="oracle")
    ap.add_argument("--steps-dir", default=None, help="mặc định <run>/steps")
    ap.add_argument("--events-dir", default=None, help="mặc định <run>/events")
    ap.add_argument("--features-dir", default=None, help="mặc định <run>/features")
    ap.add_argument("--out-dir", default=None, help="mặc định <run>/metrics")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    steps_dir = stage_dir(ctx, a.steps_dir, "steps", "outputs/steps")
    events_dir = stage_dir(ctx, a.events_dir, "events", "outputs/events")
    features_dir = stage_dir(ctx, a.features_dir, "features", "outputs/features")
    out_dir = stage_dir(ctx, a.out_dir, "metrics", "outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    out_json = out_dir / "metrics.json"
    if not can_write(out_json, a.force, TAG):
        return
    suf = f"_{a.source}.csv"
    stems = select_stems([p.name[: -len(suf)] for p in steps_dir.glob(f"*{suf}")], a.subjects, a.trials)
    if not stems:
        print(f"[{TAG}] không có steps *{suf} trong {steps_dir}", file=sys.stderr)
        sys.exit(1)

    trials, summary = {}, []
    for st in stems:
        steps = pd.read_csv(steps_dir / f"{st}{suf}")
        if not (steps["step_id"] > 0).any():
            print(f"[{TAG}] {st}: không có bước hợp lệ, bỏ qua")
            continue
        evp, fp = events_dir / f"{st}.csv", features_dir / f"{st}.parquet"
        ev = pd.read_csv(evp) if evp.exists() else None
        feat = pd.read_parquet(fp) if fp.exists() else None
        m = compute_metrics(steps, ev, feat, cfg, trial=st)
        evalp = steps_dir / f"{st}_eval.json"
        if evalp.exists():
            e = read_json(evalp)
            ev_d = {k: v for k, v in e.items() if k != "per_step"}
            ev_d["definition"] = "predicted vs nhãn HA4M: MAE (s), F1@IoU0.5 theo bước"
            m.eval_vs_oracle = ev_d
        trials[st] = m
        m.steps_table.to_csv(out_dir / f"{st}_steps.csv", index=False)
        summary.append(dict(trial=st, cycle_time_s=m.value("cycle_time_s"), va_pct=m.value("va_pct"),
                            idle_left_s=m.value("idle_left_s"), idle_right_s=m.value("idle_right_s"),
                            picks_total=m.value("picks_total"), distance_right=m.value("distance_right")))
        print(f"[{TAG}] {st}: cycle {m.value('cycle_time_s')}s, VA {m.value('va_pct')}%, "
              f"idle L/R {m.value('idle_left_s')}/{m.value('idle_right_s')}s")
    pd.DataFrame(summary).to_csv(out_dir / "trials.csv", index=False)
    res = build_metrics_report(trials, cfg, a.source).to_json_dict()
    write_json(out_json, res)
    print(f"[{TAG}] đã ghi {out_json} ({len(trials)} trial)")
    if ctx:
        ctx.update_manifest(source=a.source)
        ctx.record_stage("05_metrics", source=a.source, n_trials=len(trials), trials=list(trials))


if __name__ == "__main__":
    main()

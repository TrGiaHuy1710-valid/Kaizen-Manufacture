"""[08] Tổng hợp → outputs/report.md + ảnh PNG trong outputs/report_assets/.

Gồm: bảng tổng theo trial, sai số vs oracle, Gantt timeline (bước oracle/predicted + event 2 tay),
Yamazumi (màu VA/NVA, nhãn số bước), phân bố cycle time + bảng CV (Mura), Muri, từng finding kèm
giải thích LLM + trạng thái validator + frame bằng chứng (vẽ zone + cổ tay), mục "Giả định & giới hạn".

    python src/08_report.py [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.analyzer.report import (cycle_dist, evidence_frame, gantt, md_table, write_report,  # noqa: E402,F401
                                    yamazumi)
from kaizen.core.cli import base_parser, setup, stage_dir, stage_file  # noqa: E402
from kaizen.core.config import load_config  # noqa: E402
from kaizen.core.io import can_write, read_json  # noqa: E402

TAG = "08"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--metrics", default=None, help="mặc định <run>/metrics/metrics.json")
    ap.add_argument("--findings", default=None, help="mặc định <run>/findings/findings.json")
    ap.add_argument("--explained", default=None, help="mặc định <run>/explained/explained.json")
    ap.add_argument("--steps-dir", default=None)
    ap.add_argument("--events-dir", default=None)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--inspect", default="outputs/inspect_report.md")
    ap.add_argument("--out", default=None, help="mặc định <run>/report/report.md")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    steps_dir = stage_dir(ctx, a.steps_dir, "steps", "outputs/steps")
    events_dir = stage_dir(ctx, a.events_dir, "events", "outputs/events")
    features_dir = stage_dir(ctx, a.features_dir, "features", "outputs/features")
    f_metrics = stage_file(ctx, a.metrics, "metrics", "metrics.json", "outputs/metrics/metrics.json")
    f_findings = stage_file(ctx, a.findings, "findings", "findings.json", "outputs/findings/findings.json")
    f_explained = stage_file(ctx, a.explained, "explained", "explained.json", "outputs/findings/explained.json")
    out = stage_file(ctx, a.out, "report", "report.md", "outputs/report.md")
    man = ctx.manifest if ctx else {}
    if not can_write(out, a.force, TAG):
        return
    assets, n_trials = write_report(out, load_config(a.config), read_json(f_metrics),
                                    read_json(f_findings) if f_findings.exists() else None,
                                    read_json(f_explained) if f_explained.exists() else None,
                                    steps_dir, events_dir, features_dir, a.data_root, a.inspect, a.subjects, a.trials,
                                    ctx.run_id if ctx else None, man)
    print(f"[{TAG}] đã ghi {out} + {len(list(assets.glob('*.png')))} ảnh trong {assets}")
    if ctx:
        ctx.record_stage("08_report", report=str(out), n_trials=n_trials)


if __name__ == "__main__":
    main()

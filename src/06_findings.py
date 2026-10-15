"""[06] metrics → outputs/findings/findings.json (rule-based, KHÔNG LLM).

Finding: {id, type, severity, evidence: {metric, value, unit, trials, timestamps[], ...ngữ cảnh},
          impact_s_per_cycle, confidence (high|med|low), rule, rule_text}
Rule:
  R1 bước chậm bất thường: dur bước > median + 2·MAD giữa các trial cùng subject (cần ≥ 3 trial;
     nếu subject < 3 trial thì so với mọi trial trong dữ liệu nếu ≥ 3; nếu không → bỏ qua).
     impact = dur − median.
  R2 pick dư: số pick vào 1 zone trong chu kỳ > số lần SOP cần zone đó (Σ reps các bước có zone đó).
     impact = số pick dư × thời lượng pick trung bình của zone.
  R3 idle dài: tổng idle 1 tay > idle_threshold_s trong chu kỳ. impact = idle − ngưỡng.
  R4 sai thứ tự / thiếu bước: predicted → bước có zone không tìm thấy hoặc thứ tự mốc ≠ SOP;
     oracle → sanity check chuỗi nhãn so với SOP. impact = 0 (lỗi chất lượng, không đo được thời gian).
  R5 mất cân bằng tay: trong 1 bước, 1 tay idle > 60% thời lượng trong khi tay kia active > 60%.
     impact = Σ thời gian idle của tay rảnh ở các bước bị gắn cờ.
  R6 biến động lớn: CV cycle time giữa các trial của 1 subject > 0.25 (cần ≥ 2 trial). impact = std.
  R7 Muri cổ tay (sàng lọc 2D): % thời gian lệch góc > muri_dev_deg vượt muri_pct_threshold. impact = 0.
Severity: impact/cycle ≥ 5% → high, ≥ 2% → med, còn lại low (R4 oracle-mismatch & R7 theo mức riêng).
Xếp hạng: impact_s_per_cycle × {high:1, med:0.6, low:0.3} giảm dần.

    python src/06_findings.py [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.analyzer.findings import (CONF_W, RULES, detect_findings, load_events, mad, run_rules,  # noqa: E402,F401
                                     severity, step_rows)
from kaizen.core.cli import base_parser, setup, stage_dir, stage_file  # noqa: E402
from kaizen.core.config import load_config  # noqa: E402
from kaizen.core.io import can_write, read_json, write_json  # noqa: E402

TAG = "06"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--metrics", default=None, help="mặc định <run>/metrics/metrics.json")
    ap.add_argument("--steps-dir", default=None, help="mặc định <run>/steps")
    ap.add_argument("--events-dir", default=None, help="mặc định <run>/events")
    ap.add_argument("--out", default=None, help="mặc định <run>/findings/findings.json")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    out = stage_file(ctx, a.out, "findings", "findings.json", "outputs/findings/findings.json")
    if not can_write(out, a.force, TAG):
        return
    cfg = load_config(a.config)
    metrics = read_json(stage_file(ctx, a.metrics, "metrics", "metrics.json", "outputs/metrics/metrics.json"))
    res = detect_findings(metrics, cfg, events=stage_dir(ctx, a.events_dir, "events", "outputs/events"),
                          pred_info=stage_dir(ctx, a.steps_dir, "steps", "outputs/steps"))
    doc = res.to_json_dict(metrics["meta"]["source"])
    findings, notes = doc["findings"], doc["notes"]
    write_json(out, doc)
    for n in notes:
        print(f"[{TAG}] {n}")
    for f in findings:
        print(f"[{TAG}] {f['id']} {f['rule']} {f['type']:<18} sev={f['severity']:<4} conf={f['confidence']:<4} "
              f"impact={f['impact_s_per_cycle']}s  {f['evidence']['metric']} = {f['evidence']['value']} {f['evidence']['unit']}")
    print(f"[{TAG}] đã ghi {out} ({len(findings)} finding)")
    if ctx:
        ctx.record_stage("06_findings", n_findings=len(findings), rules=sorted({f["rule"] for f in findings}), notes=notes)


if __name__ == "__main__":
    main()

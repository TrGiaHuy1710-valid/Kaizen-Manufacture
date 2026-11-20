"""Tầng 2 — Analyzer định lượng: bước, chỉ số 3M (Muda·Mura·Muri), findings rule-based. KHÔNG LLM/mạng/key.

    align_steps(events, source, station, labels=None) -> StepsResult      # 04
    evaluate(steps_pred, steps_oracle) -> EvalReport                      # F1@IoU0.5, MAE...
    compute_metrics(steps, events, features, station) -> Metrics          # 05: Muda + Muri 1 trial
    build_metrics_report(trials, station, source) -> MetricsReport        # 05: metrics.json (+ Mura)
    detect_findings(metrics_all, rules_cfg) -> FindingsResult             # 06: RULES R1–R7
Thêm rule = thêm 1 Rule vào RULES (kaizen/analyzer/findings.py). Dashboard đọc artifact do module này sinh.
"""
from kaizen.analyzer.findings import RULES, FindingsResult, Rule, RuleContext, detect_findings, run_rules
from kaizen.analyzer.metrics import build_metrics_report, compute_metrics, compute_mura
from kaizen.analyzer.steps import StepsResult, align_steps, evaluate

__all__ = ["RULES", "FindingsResult", "Rule", "RuleContext", "StepsResult", "align_steps", "build_metrics_report",
           "compute_metrics", "compute_mura", "detect_findings", "evaluate", "run_rules"]

"""Analyzer định lượng (kaizen.analyzer): từng rule R1–R7 bắn đúng ca dương/âm trên metrics tổng hợp,
registry RULES mở rộng được, steps/eval/metrics cơ bản, và chạy được trong môi trường KHÔNG có biến env key nào.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from kaizen.analyzer import (RULES, Rule, align_steps, build_metrics_report, compute_metrics, detect_findings,
                             evaluate)
from kaizen.core.config import with_defaults

REPO = Path(__file__).resolve().parents[1]
CFG = with_defaults({"fps": 30, "idle_threshold_s": 3.0, "muri_dev_deg": 30, "muri_pct_threshold": 20,
                     "zones": {}, "steps": [{"id": 1, "name": "S1", "zone": "bin_a", "reps": 1, "va": False},
                                            {"id": 2, "name": "S2", "zone": "bin_b", "reps": 2, "va": True},
                                            {"id": 3, "name": "S3", "zone": None, "reps": 1, "va": True}]})


def mv(v, unit="s"):
    return {"value": v, "unit": unit, "definition": "test"}


def trial(durs=(10.0, 20.0, 30.0), idle=(1.0, 1.0), picks=None, row_extra=None, muri=None, order=(1, 2, 3)):
    """Metrics 1 trial tổng hợp (đúng cấu trúc metrics.json → trials[stem])."""
    rows, t = [], 0.0
    for sid in order:
        d = dict(zip((1, 2, 3), durs))[sid]
        row = dict(step_id=sid, name=f"S{sid}", va=sid != 1, n_rep=1, dur_s=d, start_s=t, end_s=t + d,
                   pct_cycle=None, left_idle_pct=10.0, right_idle_pct=10.0, left_active_pct=80.0, right_active_pct=80.0)
        row.update((row_extra or {}).get(sid, {}))
        rows.append(row)
        t += d
    m = {"window_s": [0.0, t], "cycle_time_s": mv(t), "n_steps": mv(len(order), "count"), "unclassified_s": mv(0.0),
         "steps": {"unit": {}, "definition": {}, "rows": rows}, "va_pct": mv(50.0, "%"),
         "idle_left_s": mv(idle[0]), "lost_left_pct": mv(0.0, "%"), "idle_right_s": mv(idle[1]),
         "lost_right_pct": mv(0.0, "%"), "picks_by_zone": mv(picks or {"bin_a": 1, "bin_b": 2}, "count/cycle"),
         "pick_mean_dur_s": mv({"bin_a": 0.5, "bin_b": 0.6})}
    if muri:
        m["muri_wrist"] = muri
    return m


def run(trials: dict, source="oracle", pred_info=None, rules=None):
    rep = build_metrics_report(trials, CFG, source)
    return detect_findings(rep, CFG, pred_info=pred_info or {}, rules=rules)


def fired(res, rule):
    return [f for f in res.findings if f.rule == rule]


def only(rule_id):
    return [r for r in RULES if r.id == rule_id]


# ---------------------------------------------------------------- R1 bước chậm
def test_r1_slow_step_positive_negative():
    pos = run({"U1_V1": trial((10, 20, 30)), "U1_V2": trial((10, 21, 30)), "U1_V3": trial((10, 60, 30))},
              rules=only("R1"))
    f = fired(pos, "R1")
    assert len(f) == 1 and f[0].evidence["trials"] == ["U1_V3"] and f[0].evidence["step_id"] == 2
    assert f[0].impact_s_per_cycle == 39.0 and f[0].type == "slow_step"
    neg = run({"U1_V1": trial((10, 20, 30)), "U1_V2": trial((10, 21, 30)), "U1_V3": trial((10, 22, 30))},
              rules=only("R1"))
    assert not fired(neg, "R1")


def test_r1_needs_three_trials_note():
    res = run({"U1_V1": trial(), "U1_V2": trial((10, 60, 30))}, rules=only("R1"))
    assert not res.findings and any("R1 bỏ qua" in n for n in res.notes)


# ---------------------------------------------------------------- R2 pick dư
def test_r2_extra_pick():
    pos = run({"U1_V1": trial(picks={"bin_a": 1, "bin_b": 3})}, rules=only("R2"))
    f = fired(pos, "R2")
    assert len(f) == 1 and f[0].evidence["zone"] == "bin_b" and f[0].evidence["extra"] == 1
    assert f[0].evidence["sop_required"] == 2 and f[0].impact_s_per_cycle == 0.6
    assert not run({"U1_V1": trial(picks={"bin_a": 1, "bin_b": 2})}, rules=only("R2")).findings


# ---------------------------------------------------------------- R3 idle dài
def test_r3_long_idle():
    f = fired(run({"U1_V1": trial(idle=(5.0, 1.0))}, rules=only("R3")), "R3")
    assert len(f) == 1 and f[0].evidence["hand"] == "left" and f[0].impact_s_per_cycle == 2.0
    assert not run({"U1_V1": trial(idle=(3.0, 3.0))}, rules=only("R3")).findings      # = ngưỡng → không bắn


# ---------------------------------------------------------------- R4 thứ tự / thiếu bước
def test_r4_oracle_sanity():
    pos = run({"U1_V1": trial(order=(1, 3))}, rules=only("R4"))
    f = fired(pos, "R4")
    assert len(f) == 1 and f[0].evidence["missing"] == [2] and f[0].severity == "high"
    neg = run({"U1_V1": trial()}, rules=only("R4"))
    assert not neg.findings and any("khớp SOP" in n for n in neg.notes)


def test_r4_pred_not_found_and_out_of_order():
    t = {"U1_V1": trial()}
    assert fired(run(t, "pred", {"U1_V1": {"not_found": ["3.0"], "matched_order": [1, 2]}}, only("R4")), "R4")
    assert fired(run(t, "pred", {"U1_V1": {"not_found": [], "matched_order": [2, 1, 3]}}, only("R4")), "R4")
    assert not run(t, "pred", {"U1_V1": {"not_found": [], "matched_order": [1, 2, 2, 3]}}, only("R4")).findings


# ---------------------------------------------------------------- R5 mất cân bằng tay
def test_r5_hand_imbalance():
    pos = run({"U1_V1": trial(row_extra={2: {"left_idle_pct": 70.0, "right_active_pct": 75.0}})}, rules=only("R5"))
    f = fired(pos, "R5")
    assert len(f) == 1 and f[0].evidence["idle_hand"] == "left" and f[0].impact_s_per_cycle == 14.0
    neg = run({"U1_V1": trial(row_extra={2: {"left_idle_pct": 50.0, "right_active_pct": 75.0}})}, rules=only("R5"))
    assert not neg.findings


# ---------------------------------------------------------------- R6 biến động
def test_r6_high_variation():
    pos = run({"U1_V1": trial((10, 20, 30)), "U1_V2": trial((10, 20, 30)), "U1_V3": trial((10, 80, 30))},
              rules=only("R6"))
    f = fired(pos, "R6")
    assert len(f) == 1 and f[0].evidence["value"] > 0.25 and sorted(f[0].evidence["trials"]) == ["U1_V1", "U1_V2", "U1_V3"]
    neg = run({"U1_V1": trial((10, 20, 30)), "U1_V2": trial((10, 21, 30))}, rules=only("R6"))
    assert not neg.findings
    single = run({"U1_V1": trial()}, rules=only("R6"))
    assert any("R6 bỏ qua" in n for n in single.notes)


# ---------------------------------------------------------------- R7 Muri
def muri(pct, available=True):
    h = {"available": available, "coverage_pct": mv(80.0, "%"), "median_angle_deg": mv(20.0, "deg"),
         "dev_gt_thr_pct": mv(pct, "%"), "dev_p90_deg": mv(50.0, "deg")}
    return {"note": "ước lượng 2D", "left": h, "right": {"available": False, "coverage_pct": 1.0}}


def test_r7_wrist_posture():
    assert fired(run({"U1_V1": trial(muri=muri(25.0))}, rules=only("R7")), "R7")[0].severity == "low"
    assert fired(run({"U1_V1": trial(muri=muri(45.0))}, rules=only("R7")), "R7")[0].severity == "med"
    assert not run({"U1_V1": trial(muri=muri(15.0))}, rules=only("R7")).findings
    assert not run({"U1_V1": trial(muri=muri(90.0, available=False))}, rules=only("R7")).findings


# ---------------------------------------------------------------- registry + xếp hạng
def test_rules_registry_and_ranking():
    assert [r.id for r in RULES] == [f"R{i}" for i in range(1, 8)]
    assert all(r.description and r.type for r in RULES)
    res = run({"U1_V1": trial((10, 20, 30), idle=(9.0, 1.0)), "U1_V2": trial((10, 21, 30)),
               "U1_V3": trial((10, 60, 30), picks={"bin_a": 2, "bin_b": 2})})
    assert [f.id for f in res.findings] == [f"F{i:02d}" for i in range(1, len(res.findings) + 1)]
    scores = [f.rank_score for f in res.findings]
    assert scores == sorted(scores, reverse=True) and {"R1", "R2", "R3"} <= {f.rule for f in res.findings}
    doc = res.to_json_dict("oracle")
    assert list(doc) == ["source", "rank_formula", "notes", "findings"]
    assert list(doc["findings"][0]) == ["id", "type", "severity", "evidence", "impact_s_per_cycle", "confidence",
                                        "rule", "rank_score"]


def test_custom_rule_is_one_list_entry():
    def long_cycle(ctx):
        for st, m in ctx.trials.items():
            if m["cycle_time_s"]["value"] > 100:
                ctx.add(type="long_cycle", rule="R99", trials=[st], impact_s_per_cycle=1.0, severity="low",
                        confidence="high", evidence=dict(metric="cycle_time_s", value=m["cycle_time_s"]["value"],
                                                         unit="s", timestamps=[]))
    rules = RULES + [Rule("R99", "long_cycle", "cycle time > 100 s", long_cycle)]
    res = run({"U1_V1": trial((10, 20, 30)), "U1_V2": trial((50, 50, 50))}, rules=rules)
    assert [f.evidence["trials"] for f in fired(res, "R99")] == [["U1_V2"]]


# ---------------------------------------------------------------- steps / eval / metrics
def _steps(rows, source):
    df = pd.DataFrame(rows, columns=["step_id", "repetition", "name", "start_s", "end_s"])
    df["dur_s"], df["source"] = df["end_s"] - df["start_s"], source
    return df


def test_align_steps_oracle_and_evaluate_perfect():
    labels = pd.DataFrame({"trial": "V1", "action_id": [1, 2], "repetition": [0, 0], "action_name": ["a", "b"],
                           "start_frame": [0, 30], "end_frame": [29, 89]})
    orc = align_steps(None, "oracle", CFG, labels=labels).steps
    assert list(orc["start_s"]) == [0.0, 1.0] and list(orc["end_s"]) == [1.0, 3.0] and set(orc["source"]) == {"oracle"}
    e = evaluate(orc.assign(source="pred"), orc)
    assert (e.f1_iou05, e.mae_step_dur_s, e.cycle_time_err_s) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        align_steps(None, "oracle", CFG)


def test_compute_metrics_units_and_definitions():
    steps = _steps([(1, 0, "S1", 0.0, 10.0), (2, 0, "S2", 10.0, 25.0), (3, 0, "S3", 25.0, 30.0)], "oracle")
    ev = pd.DataFrame({"start_s": [0.0, 0.0, 12.0], "end_s": [30.0, 12.0, 30.0], "hand": ["left", "right", "right"],
                       "zone": ["none", "bin_a", "none"], "motion": ["still", "still", "still"],
                       "dur_s": [30.0, 12.0, 18.0], "event_type": ["idle", "pick", "idle"]})
    m = compute_metrics(steps, ev, None, CFG, trial="U1_V1")
    assert m.value("cycle_time_s") == 30.0 and m.value("idle_left_s") == 30.0 and m.value("picks_total") == 1
    assert m.value("va_pct") == pytest.approx(100 * 20 / 30, abs=0.1)
    for k, v in m.to_json_dict().items():
        if isinstance(v, dict) and "value" in v:
            assert v["unit"] and v["definition"], k                    # mọi metric giữ unit + definition
    assert list(m.steps_table["step_id"]) == [1, 2, 3]


# ---------------------------------------------------------------- không key / không LLM / không mạng
def test_runs_with_empty_env_and_without_llm_modules(tmp_path):
    fixture = {"U1_V1": trial((10, 20, 30), idle=(9.0, 1.0)), "U1_V2": trial((10, 21, 30)),
               "U1_V3": trial((10, 60, 30))}
    (tmp_path / "t.json").write_text(json.dumps(fixture))
    (tmp_path / "cfg.json").write_text(json.dumps(CFG))
    code = f"""
import json, sys
from kaizen.analyzer import build_metrics_report, detect_findings
cfg = json.load(open({str(tmp_path / 'cfg.json')!r}))
trials = json.load(open({str(tmp_path / 't.json')!r}))
res = detect_findings(build_metrics_report(trials, cfg, "oracle"), cfg)
bad = sorted(m for m in sys.modules if m.split('.')[0] in ('anthropic', 'openai', 'google', 'chromadb', 'httpx',
                                                           'sentence_transformers', 'torch') or m.startswith('kaizen.llm'))
print(json.dumps({{"rules": sorted({{f.rule for f in res.findings}}), "bad": bad}}))
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert {"R1", "R3"} <= set(out["rules"]) and out["bad"] == []

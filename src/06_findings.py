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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, load_config, parse_stem, r, read_json, setup, stage_dir, stage_file, write_json  # noqa: E402

TAG = "06"
CONF_W = {"high": 1.0, "med": 0.6, "low": 0.3}


def severity(impact: float, ct: float) -> str:
    frac = impact / ct if ct else 0
    return "high" if frac >= 0.05 else ("med" if frac >= 0.02 else "low")


def mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def load_events(d: Path, st: str) -> pd.DataFrame | None:
    p = d / f"{st}.csv"
    return pd.read_csv(p) if p.exists() else None


def step_rows(m: dict) -> dict[int, dict]:
    return {row["step_id"]: row for row in m["steps"]["rows"]}


def run_rules(metrics: dict, cfg: dict, steps_dir: Path, events_dir: Path) -> tuple[list[dict], list[str]]:
    trials = metrics["trials"]
    source = metrics["meta"]["source"]
    out, notes = [], []
    by_sub: dict[str, list[str]] = {}
    for st in trials:
        by_sub.setdefault(parse_stem(st)[0], []).append(st)

    def add(**kw):
        out.append(kw)

    # R1
    r1_ran = False
    for sub, sts in by_sub.items():
        pool = sts if len(sts) >= 3 else (list(trials) if len(trials) >= 3 else None)
        if pool is None:
            continue
        r1_ran = True
        scope = "subject" if pool is sts else "toàn bộ dữ liệu"
        sids = sorted({sid for st in pool for sid in step_rows(trials[st])})
        for sid in sids:
            vals = {st: step_rows(trials[st]).get(sid, {}).get("dur_s") for st in pool}
            v = np.array([x for x in vals.values() if x is not None], float)
            if len(v) < 3:
                continue
            med, md = float(np.median(v)), mad(v)
            thr = med + 2 * md
            for st in sts:
                row = step_rows(trials[st]).get(sid)
                if not row or md == 0 or row["dur_s"] <= thr:
                    continue
                imp = row["dur_s"] - med
                ct = trials[st]["cycle_time_s"]["value"]
                add(type="slow_step", rule="R1", trials=[st],
                    evidence=dict(metric=f"dur_s bước {sid}", value=row["dur_s"], unit="s", step_id=sid,
                                  step_name=row["name"], median_s=r(med), mad_s=r(md), threshold_s=r(thr),
                                  n_compare=len(v), compare_scope=scope,
                                  timestamps=[[row["start_s"], row["end_s"]]]),
                    impact_s_per_cycle=r(imp), severity=severity(imp, ct), confidence="high" if source == "oracle" else "med")
    if not r1_ran:
        notes.append("R1 bỏ qua: cần ≥ 3 trial (cùng subject hoặc toàn bộ) để tính median + 2·MAD")

    # R2
    need: dict[str, int] = {}
    for s in cfg.get("steps", []):
        if s.get("zone"):
            need[s["zone"]] = need.get(s["zone"], 0) + int(s.get("reps", 1))
    for st, m in trials.items():
        if "picks_by_zone" not in m:
            continue
        ct = m["cycle_time_s"]["value"]
        ev = load_events(events_dir, st)
        for zone, n in m["picks_by_zone"]["value"].items():
            n_need = need.get(zone, 0)
            if n <= n_need:
                continue
            extra = n - n_need
            pdur = m["pick_mean_dur_s"]["value"].get(zone) or 0.0
            imp = extra * pdur
            ts = []
            if ev is not None:
                w0, w1 = m["window_s"]
                pk = ev[(ev["event_type"] == "pick") & (ev["zone"] == zone) & (ev["start_s"] >= w0) & (ev["start_s"] <= w1)]
                ts = [[r(a), r(b)] for a, b in zip(pk["start_s"], pk["end_s"])]
            add(type="extra_pick", rule="R2", trials=[st],
                evidence=dict(metric=f"số pick zone {zone}", value=n, unit="count/cycle", zone=zone,
                              sop_required=n_need, extra=extra, pick_mean_dur_s=pdur, timestamps=ts),
                impact_s_per_cycle=r(imp), severity=severity(imp, ct), confidence="med")

    # R3
    thr3 = float(cfg.get("idle_threshold_s", 3.0))
    for st, m in trials.items():
        ct = m["cycle_time_s"]["value"]
        ev = load_events(events_dir, st)
        for hand in ("left", "right"):
            k = f"idle_{hand}_s"
            if k not in m or m[k]["value"] is None or m[k]["value"] <= thr3:
                continue
            idle = m[k]["value"]
            ts = []
            if ev is not None:
                w0, w1 = m["window_s"]
                e = ev[(ev["hand"] == hand) & (ev["event_type"] == "idle") & (ev["end_s"] > w0) & (ev["start_s"] < w1)]
                e = e.sort_values("dur_s", ascending=False).head(5)
                ts = [[r(a), r(b)] for a, b in zip(e["start_s"], e["end_s"])]
            imp = idle - thr3
            add(type="long_idle", rule="R3", trials=[st],
                evidence=dict(metric=f"idle_{hand}_s", value=idle, unit="s", hand=hand, threshold_s=thr3,
                              cycle_time_s=ct, idle_pct_cycle=r(100 * idle / ct, 1), timestamps=ts),
                impact_s_per_cycle=r(imp), severity=severity(imp, ct),
                confidence="med" if m.get(f"lost_{hand}_pct", {}).get("value", 0) < 10 else "low")

    # R4
    sop_order = [int(s["id"]) for s in cfg.get("steps", [])]
    for st, m in trials.items():
        if source == "pred":
            info_p = steps_dir / f"{st}_pred_info.json"
            info = read_json(info_p) if info_p.exists() else {}
            nf, order = info.get("not_found", []), info.get("matched_order", [])
            dedup = [s for i, s in enumerate(order) if i == 0 or s != order[i - 1]]
            out_of_order = dedup != sorted(dedup)
            if nf or out_of_order:
                add(type="sequence_deviation", rule="R4", trials=[st],
                    evidence=dict(metric="chuỗi bước dự đoán vs SOP", value=len(nf), unit="bước không tìm thấy",
                                  not_found=nf, matched_order=dedup, sop_order=sop_order, out_of_order=out_of_order,
                                  timestamps=[]),
                    impact_s_per_cycle=0.0, severity="med", confidence="low")
        else:
            seq = [row["step_id"] for row in sorted(m["steps"]["rows"], key=lambda x: x["start_s"])]
            if seq != sop_order:
                missing = [s for s in sop_order if s not in seq]
                add(type="sequence_deviation", rule="R4", trials=[st],
                    evidence=dict(metric="chuỗi nhãn vs SOP (sanity check)", value=len(missing), unit="bước thiếu",
                                  observed_order=seq, sop_order=sop_order, missing=missing, timestamps=[]),
                    impact_s_per_cycle=0.0, severity="high", confidence="high")
            else:
                notes.append(f"R4 (sanity, oracle) {st}: chuỗi nhãn khớp SOP 12 bước")

    # R5
    idle_lim, act_lim = 60.0, 60.0
    for st, m in trials.items():
        ct = m["cycle_time_s"]["value"]
        for idle_h, act_h in (("left", "right"), ("right", "left")):
            flagged, tot = [], 0.0
            for row in m["steps"]["rows"]:
                ip, ap_ = row.get(f"{idle_h}_idle_pct"), row.get(f"{act_h}_active_pct")
                if ip is None or ap_ is None:
                    continue
                if ip > idle_lim and ap_ > act_lim:
                    sec = row["dur_s"] * ip / 100
                    tot += sec
                    flagged.append(dict(step_id=row["step_id"], idle_pct=ip, other_active_pct=ap_, idle_s=r(sec),
                                        start_s=row["start_s"], end_s=row["end_s"]))
            if flagged:
                add(type="hand_imbalance", rule="R5", trials=[st],
                    evidence=dict(metric=f"% idle tay {idle_h} trong bước khi tay {act_h} hoạt động", value=r(tot),
                                  unit="s", idle_hand=idle_h, active_hand=act_h, idle_pct_threshold=idle_lim,
                                  steps=flagged, timestamps=[[f["start_s"], f["end_s"]] for f in flagged]),
                    impact_s_per_cycle=r(tot), severity=severity(tot, ct), confidence="low")

    # R6
    r6 = False
    for sub, v in metrics["mura"]["within_subject"].items():
        c = v["cycle_time"]
        if c.get("cv") is None:
            continue
        r6 = True
        if c["cv"] > 0.25:
            sts = by_sub[sub]
            add(type="high_variation", rule="R6", trials=sts,
                evidence=dict(metric=f"CV cycle time subject {sub}", value=c["cv"], unit="ratio", threshold=0.25,
                              mean_s=c["mean"], std_s=c["std"], n=c["n"],
                              cycle_times_s={st: trials[st]["cycle_time_s"]["value"] for st in sts}, timestamps=[]),
                impact_s_per_cycle=c["std"], severity=severity(c["std"], c["mean"]), confidence="high")
    if not r6:
        notes.append("R6 bỏ qua: cần ≥ 2 trial cùng subject để tính CV cycle time")

    # R7
    pct_thr = float(cfg.get("muri_pct_threshold", 20))
    for st, m in trials.items():
        mw = m.get("muri_wrist", {})
        for hand in ("left", "right"):
            h = mw.get(hand, {})
            if not h.get("available"):
                continue
            p = h["dev_gt_thr_pct"]["value"]
            if p is not None and p > pct_thr:
                add(type="wrist_posture", rule="R7", trials=[st],
                    evidence=dict(metric=f"% thời gian góc cổ tay {hand} lệch > {cfg.get('muri_dev_deg', 30)}°",
                                  value=p, unit="%", hand=hand, threshold_pct=pct_thr,
                                  median_angle_deg=h["median_angle_deg"]["value"],
                                  dev_p90_deg=h["dev_p90_deg"]["value"], coverage_pct=h["coverage_pct"]["value"],
                                  note=mw.get("note"), timestamps=[]),
                    impact_s_per_cycle=0.0, severity="med" if p > 2 * pct_thr else "low", confidence="low")

    # Xếp hạng + id
    for f in out:
        f["score"] = r((f["impact_s_per_cycle"] or 0) * CONF_W[f["confidence"]], 3)
    out.sort(key=lambda f: (-f["score"], f["rule"]))
    final = []
    for i, f in enumerate(out, 1):
        ev = f.pop("evidence")
        ev["trials"] = f.pop("trials")
        final.append({"id": f"F{i:02d}", "type": f["type"], "severity": f["severity"], "evidence": ev,
                      "impact_s_per_cycle": f["impact_s_per_cycle"], "confidence": f["confidence"],
                      "rule": f["rule"], "rank_score": f["score"]})
    return final, notes


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
    findings, notes = run_rules(metrics, cfg, stage_dir(ctx, a.steps_dir, "steps", "outputs/steps"),
                                stage_dir(ctx, a.events_dir, "events", "outputs/events"))
    write_json(out, {"source": metrics["meta"]["source"], "rank_formula": "impact_s_per_cycle × {high:1, med:0.6, low:0.3}",
                     "notes": notes, "findings": findings})
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

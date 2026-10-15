"""So sánh các run trong outputs/runs/ (đọc manifest.json, metrics/metrics.json, explained/explained.json).

    python src/tools/compare_runs.py --list
    python src/tools/compare_runs.py <run_id_1> <run_id_2> [...]     # 'latest' cũng được
    python src/tools/compare_runs.py --last 2                        # 2 run gần nhất

In bảng Markdown: tổng hợp mỗi run (trung bình các trial) + cycle time theo từng trial.
Dùng để so pred vs oracle, mediapipe vs rtmw, anthropic vs gemini.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_io  # noqa: E402


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(float(np.mean(xs)), 2) if xs else None


def summarize(ctx: run_io.RunContext) -> dict:
    man = ctx.manifest
    met = _load(ctx.root / "metrics" / "metrics.json")
    exp = _load(ctx.root / "explained" / "explained.json")
    fnd = _load(ctx.root / "findings" / "findings.json")
    tr = met.get("trials", {})
    val = lambda m, k: (m.get(k) or {}).get("value")  # noqa: E731
    ev = [m.get("eval_vs_oracle", {}) for m in tr.values()]
    return {
        "run": ctx.run_id, "created": man.get("created_at", ""), "source": met.get("meta", {}).get("source") or man.get("source"),
        "pose": man.get("pose_model"), "llm": llm_label(man),
        "n_trials": len(tr), "cycle_s": _mean([val(m, "cycle_time_s") for m in tr.values()]),
        "va_pct": _mean([val(m, "va_pct") for m in tr.values()]),
        "idle_l_s": _mean([val(m, "idle_left_s") for m in tr.values()]),
        "idle_r_s": _mean([val(m, "idle_right_s") for m in tr.values()]),
        "f1": _mean([e.get("f1_iou05") for e in ev]), "mae_dur_s": _mean([e.get("mae_step_dur_s") for e in ev]),
        "ct_err_s": _mean([e.get("cycle_time_err_s") for e in ev]),
        "n_findings": len(fnd.get("findings", [])) if fnd else None,
        "pass": exp.get("pass_rate"), "pass_1st": exp.get("pass_rate_first_attempt"),
        "per_trial": {st: val(m, "cycle_time_s") for st, m in tr.items()},
    }


def llm_label(m: dict) -> str:
    llm = m.get("llm") or {}
    return f"{llm.get('provider')}" + (f":{llm['model']}" if llm.get("model") else "")


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    f = lambda v: "—" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))  # noqa: E731
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(r.get(k)) for k, _ in cols) + " |" for r in rows]
    return "\n".join(out)


SUMMARY_COLS = [("run", "Run"), ("source", "Nguồn"), ("pose", "Pose"), ("llm", "LLM"), ("n_trials", "Trial"),
                ("cycle_s", "Cycle time (s)"), ("va_pct", "%VA"), ("idle_l_s", "Idle trái (s)"),
                ("idle_r_s", "Idle phải (s)"), ("f1", "F1@IoU0.5"), ("mae_dur_s", "MAE bước (s)"),
                ("ct_err_s", "Sai số CT (s)"), ("n_findings", "Findings"), ("pass", "Pass LLM"),
                ("pass_1st", "Pass lần 1")]
LIST_COLS = [("run", "Run"), ("created", "Thời điểm"), ("source", "Nguồn bước"), ("n_trials", "Số trial"),
             ("pose", "Pose"), ("llm", "LLM"), ("tag", "Tag"), ("latest", "")]


def list_table() -> list[dict]:
    """Dòng cho `--list`: mọi run (cũ → mới)."""
    latest = run_io.latest_id()
    return [{"run": m["run_id"], "created": m.get("created_at"), "source": m.get("source"),
             "n_trials": len(m.get("trials", {})), "pose": m.get("pose_model"), "llm": llm_label(m),
             "tag": m.get("tag") or "", "latest": "◀" if m["run_id"] == latest else ""} for m in run_io.list_runs()]


def compare(run_ids: list[str]) -> dict:
    """Bảng so sánh dùng chung cho CLI và UI: {summary: [dòng/run], per_trial: [dòng/trial], runs: [...]}.
    Raise FileNotFoundError nếu run không tồn tại ('latest' được chấp nhận)."""
    S = [summarize(run_io.open_run(r)) for r in run_ids]
    trials = sorted({t for s in S for t in s["per_trial"]})
    per_trial = [{"trial": t, **{s["run"]: s["per_trial"].get(t) for s in S}} for t in trials]
    return {"summary": [{k: v for k, v in s.items() if k != "per_trial"} for s in S], "per_trial": per_trial,
            "runs": [s["run"] for s in S]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--list", action="store_true", help="liệt kê mọi run")
    ap.add_argument("--last", type=int, default=0, help="so sánh N run gần nhất")
    a = ap.parse_args()

    if a.list or (not a.runs and not a.last):
        rows = list_table()
        if not rows:
            print("Chưa có run nào trong outputs/runs/")
            return
        print(md_table(rows, LIST_COLS))
        return

    ids = a.runs or [m["run_id"] for m in run_io.list_runs()][-a.last:]
    try:
        res = compare(ids)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print("### Tổng hợp theo run (trung bình các trial)\n")
    print(md_table(res["summary"], SUMMARY_COLS))
    if res["per_trial"]:
        print("\n### Cycle time theo trial (s)\n")
        print(md_table(res["per_trial"], [("trial", "Trial")] + [(r, r) for r in res["runs"]]))


if __name__ == "__main__":
    main()

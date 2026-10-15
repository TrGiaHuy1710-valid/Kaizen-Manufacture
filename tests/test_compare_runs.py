"""compare_runs: hàm compare() import được (dùng chung CLI + UI) và khớp output CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "tools"))


def _fake_run(root: Path, rid: str, source: str, ct: float, f1: float, pass_rate: float) -> None:
    d = root / "outputs" / "runs" / rid
    for s in ("metrics", "explained", "findings"):
        (d / s).mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"run_id": rid, "created_at": rid[:15], "source": source,
                                                 "pose_model": "mediapipe", "llm": {"provider": "dry-run"},
                                                 "trials": {"U001_V001": {}}}))
    (d / "metrics" / "metrics.json").write_text(json.dumps({"meta": {"source": source}, "trials": {"U001_V001": {
        "cycle_time_s": {"value": ct}, "va_pct": {"value": 80.0}, "idle_left_s": {"value": 1.0},
        "idle_right_s": {"value": 2.0}, "eval_vs_oracle": {"f1_iou05": f1, "mae_step_dur_s": 0.5, "cycle_time_err_s": 0.6}}}}))
    (d / "explained" / "explained.json").write_text(json.dumps({"pass_rate": pass_rate, "pass_rate_first_attempt": 0.5}))
    (d / "findings" / "findings.json").write_text(json.dumps({"findings": [{"id": "F01"}]}))
    (root / "outputs" / "runs" / "LATEST").write_text(rid)


def test_compare_function_matches_cli(tmp_path, monkeypatch):
    _fake_run(tmp_path, "20260101-000000_oracle_mediapipe_dry", "oracle", 77.9, 0.75, 1.0)
    _fake_run(tmp_path, "20260101-000100_pred_mediapipe_dry", "pred", 78.5, 0.75, 0.5)
    monkeypatch.chdir(tmp_path)
    import compare_runs
    ids = ["20260101-000000_oracle_mediapipe_dry", "latest"]
    res = compare_runs.compare(ids)
    assert res["runs"] == ["20260101-000000_oracle_mediapipe_dry", "20260101-000100_pred_mediapipe_dry"]
    s0, s1 = res["summary"]
    assert (s0["cycle_s"], s1["cycle_s"], s0["source"], s1["pass"], s1["pass_1st"]) == (77.9, 78.5, "oracle", 0.5, 0.5)
    assert res["per_trial"] == [{"trial": "U001_V001", ids[0]: 77.9, "20260101-000100_pred_mediapipe_dry": 78.5}]
    cli = subprocess.run([sys.executable, str(REPO / "src" / "tools" / "compare_runs.py"), *ids], cwd=tmp_path,
                         capture_output=True, text=True, check=True).stdout
    assert compare_runs.md_table(res["summary"], compare_runs.SUMMARY_COLS) in cli
    assert len(compare_runs.list_table()) == 2

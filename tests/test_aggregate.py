"""kaizen.analyzer.aggregate — 3M nhiều video: CV đúng, R1/R6 chạy/skip đúng điều kiện, lỗi mềm khi < 2 video,
lưu/mở lại/ghi chú đánh giá. Dự án giả (metrics.json tổng hợp) trong thư mục tạm, không cần dataset."""
from __future__ import annotations

import json

import numpy as np
import pytest

from kaizen.analyzer import aggregate as ag
from kaizen.core import projects


def mv(v, unit="s"):
    return {"value": v, "unit": unit, "definition": "test"}


def trial_metrics(durs, muri_cov=80.0):
    rows, t = [], 0.0
    for sid, d in zip((1, 2, 3), durs):
        rows.append(dict(step_id=sid, name=f"S{sid}", va=sid != 1, n_rep=1, dur_s=d, start_s=t, end_s=t + d,
                         pct_cycle=None, left_idle_pct=10.0, right_idle_pct=10.0, left_active_pct=80.0,
                         right_active_pct=80.0))
        t += d
    hand = {"available": True, "coverage_pct": mv(muri_cov, "%"), "median_angle_deg": mv(15.0, "°"),
            "dev_gt_thr_pct": mv(12.0, "%"), "dev_p90_deg": mv(35.0, "°")}
    return {"window_s": [0.0, t], "cycle_time_s": mv(t), "n_steps": mv(3, "count"), "unclassified_s": mv(0.0),
            "steps": {"unit": {}, "definition": {}, "rows": rows}, "va_pct": mv(50.0, "%"), "idle_left_s": mv(1.0),
            "idle_right_s": mv(1.0), "picks_total": mv(4, "count/cycle"), "distance_left": mv(100.0, "px"),
            "distance_right": mv(200.0, "px"), "muri_wrist": {"left": hand, "right": {"available": False},
                                                             "note": "ước lượng 2D"}}


def make_project(pid, durs, source="oracle", done=True):
    projects.save_project({"id": pid, "name": pid, "stem": pid, "kind": "ha4m", "source_uri": None,
                           "labels_available": source == "oracle", "created_at": "2026-01-01T00:00:00"})
    if not done:
        return
    run = projects.runs_root(pid) / "20260101-000000_oracle_mediapipe_dry_ui"
    (run / "metrics").mkdir(parents=True)
    (run / "findings").mkdir()
    (run / "metrics" / "metrics.json").write_text(json.dumps(
        {"meta": {"source": source}, "trials": {pid: trial_metrics(durs)}, "mura": {}}))
    (run / "findings" / "findings.json").write_text(json.dumps({"findings": []}))
    (projects.runs_root(pid) / "LATEST").write_text(run.name)


@pytest.fixture(autouse=True)
def dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZEN_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("KAIZEN_AGGREGATES_DIR", str(tmp_path / "aggregates"))


def test_three_videos_cv_r1_r6_and_reference():
    make_project("U001_V001", (10.0, 20.0, 30.0))
    make_project("U001_V002", (10.0, 21.0, 30.0))
    make_project("U001_V003", (10.0, 60.0, 30.0))
    agg = ag.aggregate_3m(["U001_V001", "U001_V002", "U001_V003"])
    assert agg.ok and len(agg.projects) == 3 and not agg.skipped
    ct = np.array([60.0, 61.0, 100.0])
    assert agg.mura["cycle_time"]["cv"] == round(ct.std(ddof=1) / ct.mean(), 3)
    assert agg.mura["cycle_time"]["unit"] == "s" and agg.mura["cycle_time"]["definition"]
    s2 = next(s for s in agg.mura["steps_cv"] if s["step_id"] == 2)
    assert s2["is_max"] and s2["name"] == "S2" and s2["cv"] == round(np.std([20, 21, 60], ddof=1) / np.mean([20, 21, 60]), 3)
    rules = {r["rule"]: r for r in agg.rules}
    assert rules["R1"]["ran"] and rules["R1"]["fired"] and rules["R6"]["ran"] and rules["R6"]["fired"]
    r1 = [f for f in agg.findings["findings"] if f["rule"] == "R1"]
    assert [f["evidence"]["trials"] for f in r1] == [["U001_V003"]] and r1[0]["evidence"]["step_id"] == 2
    assert agg.muda["reference"] == "U001_V001" and agg.muda["units"]["cycle_time_s"] == "s"
    ref = next(x for x in agg.muda["rows"] if x["is_reference"])
    assert ref["delta_vs_ref_s"] == 0.0 and max(x["delta_vs_ref_s"] for x in agg.muda["rows"]) == 40.0
    assert [s["step_id"] for s in agg.yamazumi["U001_V003"]] == [1, 2, 3]
    assert {r["hand"] for r in agg.muri["rows"]} == {"left", "right"} and any(r["low_coverage"] for r in agg.muri["rows"])


def test_two_videos_r1_skipped_with_reason_r6_runs():
    make_project("U001_V001", (10.0, 20.0, 30.0))
    make_project("U001_V002", (10.0, 21.0, 30.0))
    agg = ag.aggregate_3m(["U001_V001", "U001_V002"])
    rules = {r["rule"]: r for r in agg.rules}
    assert agg.ok and not rules["R1"]["ran"] and "≥ 3" in rules["R1"]["reason"]
    assert rules["R6"]["ran"] and not rules["R6"]["fired"]               # CV nhỏ → không bắn, nhưng đã chạy


def test_one_video_soft_error_and_skipped_projects():
    make_project("U001_V001", (10.0, 20.0, 30.0))
    make_project("U001_V009", (10.0, 20.0, 30.0), done=False)
    agg = ag.aggregate_3m(["U001_V001", "U001_V009", "khong-co"])
    assert not agg.ok and "cần ≥2" in agg.error and "(hiện có 1)" in agg.error
    reasons = {s["project_id"]: s["reason"] for s in agg.skipped}
    assert "chưa chạy xong" in reasons["U001_V009"] and "khong-co" in reasons
    assert ag.list_aggregates() == []                                     # lỗi mềm → không lưu


def test_save_load_review_roundtrip_and_mixed_sources():
    make_project("U001_V001", (10.0, 20.0, 30.0))
    make_project("VID_clip", (12.0, 25.0, 30.0), source="pred")
    agg = ag.aggregate_3m(["U001_V001", "VID_clip"])
    assert agg.ok and {p["source"] for p in agg.projects} == {"oracle", "pred"}
    assert any("trộn lẫn" in n for n in agg.findings["notes"])
    assert not {r["rule"]: r for r in agg.rules}["R6"]["ran"]           # 2 subject khác nhau → R6 skip có lý do
    lst = ag.list_aggregates()
    assert [x["agg_id"] for x in lst] == [agg.agg_id] and lst[0]["n_videos"] == 2
    back = ag.load_aggregate(agg.agg_id)
    assert back.mura == agg.mura and back.findings == agg.findings
    ag.set_review(agg.agg_id, "Bước 2 cần xem lại khay.")
    assert ag.load_aggregate(agg.agg_id).review["note"] == "Bước 2 cần xem lại khay."
    assert ag.list_aggregates()[0]["has_review"]
    with pytest.raises(FileNotFoundError):
        ag.load_aggregate("khong-co")

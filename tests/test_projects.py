"""kaizen.core.projects + pipeline — tạo (HA4M / thư mục ảnh / upload), trạng thái, khoá job toàn app, adopt run cũ,
benchmark, demo_ready, và 1 lần chạy THẬT full luồng 6 bước trên trial HA4M tổng hợp (offline, dry-run)."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from kaizen.core import pipeline, projects

REPO = Path(__file__).resolve().parents[1]


def _synthetic():
    spec = importlib.util.spec_from_file_location("synthetic_mod", REPO / "tests" / "synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIZEN_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("KAIZEN_AGGREGATES_DIR", str(tmp_path / "aggregates"))


def _frames_dir(d: Path, n=5) -> Path:
    d.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(d / f"frame_{i:04d}.jpg"), np.full((90, 160, 3), 30 * i, np.uint8))
    return d


def test_create_frames_project_status_and_listing(tmp_path):
    info = projects.create_project(f"frames://{_frames_dir(tmp_path / 'imgs')}?fps=10", "Quay thử #1")
    assert info["id"] == "VID_Quay-thu-1" and info["stem"] == info["id"] and info["kind"] == "frames"
    assert not info["labels_available"] and projects.media_path(info["id"], "thumb.jpg").exists()
    again = projects.create_project(f"frames://{tmp_path / 'imgs'}?fps=10", "Quay thử #1")
    assert again["id"] == "VID_Quay-thu-1-2"                                     # không ghi đè dự án cũ
    lst = projects.list_projects()
    assert {p["id"] for p in lst} == {info["id"], again["id"]} and {p["status"] for p in lst} == {"empty"}
    assert projects.thumb_data_uri(lst[0]["thumb"]).startswith("data:image/jpeg;base64,")
    with pytest.raises(projects.ProjectError):
        projects.create_project("video:///khong/co.mp4")
    with pytest.raises(projects.ProjectError):
        projects.load_project("khong-co")


def test_create_from_upload_moves_video_into_project(tmp_path):
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    for i in range(8):
        w.write(np.full((90, 160, 3), 20 * i, np.uint8))
    w.release()
    info = projects.create_project_from_upload(p.read_bytes(), "clip tự quay.mp4", max_seconds=60)
    src = projects.media_path(info["id"], "source.mp4")
    assert src.exists() and info["source_uri"] == f"video://{src.resolve()}" and info["max_seconds"] == 60
    assert projects.load_project(info["id"])["source_uri"] == info["source_uri"]
    assert not list((projects.projects_root() / ".uploads").glob("*"))


def test_status_stale_running_and_job_lock(tmp_path):
    info = projects.create_project(f"frames://{_frames_dir(tmp_path / 'imgs')}?fps=10", "a")
    pid = info["id"]
    projects.write_status(pid, state="running", pid=999999)              # tiến trình không tồn tại
    assert projects.project_status(pid) == "error" and "bất thường" in projects.read_status(pid)["message"]
    assert projects.current_job() is None
    projects.acquire_job("du-an-A", os.getppid())                        # tiến trình còn sống giữ khoá
    try:
        assert projects.current_job()["project_id"] == "du-an-A"
        n = len(pipeline.CALLS)
        with pytest.raises(projects.JobBusy, match="du-an-A"):
            pipeline.start_job(pid)                                       # dự án B bị khoá, không spawn gì
        assert projects.read_status(pid)["state"] != "running" and len(pipeline.CALLS) == n + 1
        with pytest.raises(projects.JobBusy):
            projects.acquire_job(pid, os.getpid())
    finally:
        projects.release_job(os.getppid())
    assert projects.current_job() is None
    projects.acquire_job(pid, 999999)                                    # khoá mồ côi → tự dọn
    assert projects.current_job() is None


def test_resolve_provider_missing_key_falls_back_to_dry_run():
    assert pipeline.resolve_provider("openai", {}) == ("dry-run", pipeline.resolve_provider("openai", {})[1])
    assert "OPENAI_API_KEY" in pipeline.resolve_provider("openai", {})[1]
    assert pipeline.resolve_provider(None, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "x"}) == ("gemini", "")
    assert pipeline.resolve_provider("dry-run", {})[0] == "dry-run"


def test_adopt_old_run_and_unadopted_listing(tmp_path):
    rid = "20260101-000000_oracle_mediapipe_dry_old"
    run = tmp_path / "runs" / rid
    (run / "metrics").mkdir(parents=True)
    (run / "findings").mkdir()
    (run / "report").mkdir()
    (run / "manifest.json").write_text(json.dumps({"run_id": rid, "source": "oracle", "data_root": str(tmp_path / "nope"),
                                                   "trials": {"U001_V001": {"labels_available": True}}}))
    (run / "metrics" / "metrics.json").write_text(json.dumps({"meta": {"source": "oracle"},
                                                              "trials": {"U001_V001": {"cycle_time_s": {"value": 77.9}}}}))
    (run / "findings" / "findings.json").write_text(json.dumps({"findings": []}))
    (run / "report" / "demo_U001_V001.mp4").write_bytes(b"fake")
    assert projects.unadopted_runs(tmp_path / "runs") == [rid]
    made = projects.adopt(rid, tmp_path / "runs")
    assert made == ["U001_V001"] and projects.latest_run("U001_V001") == rid
    assert projects.project_status("U001_V001") == "done" and projects.load_project("U001_V001")["kind"] == "adopted"
    assert projects.demo_video_meta("U001_V001")["source"] == "oracle"
    assert projects.run_summary("U001_V001")["cycle_time_s"] == 77.9
    assert projects.unadopted_runs(tmp_path / "runs") == [] and projects.adopt(rid, tmp_path / "runs") == []
    ok, miss = projects.demo_ready("U001_V001")
    assert not ok and "explained/explained.json" in miss


def test_run_project_full_pipeline_synthetic(tmp_path, monkeypatch):
    syn = _synthetic()
    root, kp = tmp_path / "ha4m", tmp_path / "kp"
    root.mkdir()
    kp.mkdir()
    syn.make_trial(root, kp, "U900", "V001", "normal", np.random.default_rng(0))
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump(syn.station_cfg(), sort_keys=False))
    monkeypatch.setenv("KAIZEN_KP_DIR", str(kp))
    info = projects.create_project(f"ha4m://{root}?subject=U900&trial=V001", station=str(station))
    pid = info["id"]
    assert pid == "U900_V001" and info["labels_available"] and info["kind"] == "ha4m"
    seen = []
    rid = pipeline.run_project(pid, provider="dry-run", log=lambda *_: None, on_step=lambda s: seen.append(s["current"]))
    st = projects.read_status(pid)
    assert st["state"] == "done" and [s["state"] for s in st["steps"]] == ["done"] * 6, st
    assert [k for k, _ in pipeline.STEPS] == list(dict.fromkeys(seen))
    assert "tái dùng" in st["steps"][0]["detail"]                         # npz có sẵn → không trích lại
    assert projects.project_status(pid) == "done" and projects.latest_run(pid) == rid
    run = projects.run_path(pid)
    for rel in ("metrics/metrics.json", "findings/findings.json", "explained/explained.json", "report/report.md",
                "steps/U900_V001_oracle.csv", "steps/U900_V001_eval.json"):
        assert (run / rel).exists(), rel
    assert projects.demo_video_meta(pid)["source"] == "oracle" and projects.media_path(pid, "demo.mp4").stat().st_size > 0
    b = projects.benchmark(pid)
    labels = [x["label"] for x in b["rows"]]
    assert "F1@IoU0.5" in labels and b["note"] == projects.BENCH_NOTE and b["model_line"] == projects.MODEL_LINE
    assert projects.demo_ready(pid) == (True, [])
    assert projects.run_summary(pid)["f1_iou05"] is not None

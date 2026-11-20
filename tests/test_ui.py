"""UI 4 trang (streamlit AppTest, headless): mọi trang render không lỗi; demo mode có badge ở mọi trang, chạy dự án
demo KHÔNG gọi pipeline thật; tắt demo → nút chạy gọi pipeline; khoá job toàn app; 3M + ghi chú đánh giá.
Dữ liệu: 1 dự án HA4M tổng hợp chạy THẬT full luồng (dry-run, offline) + 1 bản sao + 1 dự án rỗng, trong thư mục tạm."""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from kaizen.core import pipeline, projects  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
P_HOME, P_PROJ, P_DOCS, P_3M = "📁 Dự án", "🎬 Chi tiết dự án", "📚 Tài liệu (RAG)", "📊 So sánh 3M"
BADGE = "⚡ Demo — kết quả tính sẵn, overlay theo nhãn chuẩn"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Thư mục projects/aggregates tạm: U900_V001 (chạy thật, demo-ready), U900_V002 (bản sao), VID_rong (chưa chạy)."""
    tmp = tmp_path_factory.mktemp("ui")
    env = {"KAIZEN_PROJECTS_DIR": str(tmp / "projects"), "KAIZEN_AGGREGATES_DIR": str(tmp / "aggregates"),
           "KAIZEN_KP_DIR": str(tmp / "kp"), "KAIZEN_RAG_STORE": str(tmp / "rag"), "KAIZEN_DEMO_SPEED": "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("synthetic_ui", REPO / "tests" / "synthetic.py")
    syn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(syn)
    root = tmp / "ha4m"
    root.mkdir()
    (tmp / "kp").mkdir()
    syn.make_trial(root, tmp / "kp", "U900", "V001", "normal", np.random.default_rng(0))
    station = tmp / "station.yaml"
    station.write_text(yaml.safe_dump(syn.station_cfg(), sort_keys=False))
    pid = projects.create_project(f"ha4m://{root}?subject=U900&trial=V001", station=str(station))["id"]
    pipeline.run_project(pid, provider="dry-run", log=lambda *_: None)
    shutil.copytree(projects.project_dir(pid), projects.project_dir("U900_V002"))
    info = projects.load_project("U900_V002")
    info.update(id="U900_V002", name="U900_V002 (bản sao)", created_at="2026-01-01T00:00:00")
    projects.save_project(info)
    imgs = tmp / "imgs"
    imgs.mkdir()
    import cv2
    for i in range(3):
        cv2.imwrite(str(imgs / f"f{i}.jpg"), np.full((60, 80, 3), 50 * i, np.uint8))
    projects.create_project(f"frames://{imgs}?fps=10", "rong")
    yield {"pid": pid, "tmp": tmp}
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def keep_env(world, monkeypatch):
    """autouse của conftest đặt KAIZEN_RAG_STORE riêng — giữ nguyên env của world cho AppTest."""
    for k in ("KAIZEN_PROJECTS_DIR", "KAIZEN_AGGREGATES_DIR", "KAIZEN_KP_DIR", "KAIZEN_DEMO_SPEED"):
        monkeypatch.setenv(k, os.environ[k])


def app(page: str = P_HOME, demo: bool = False, **state) -> AppTest:
    at = AppTest.from_file(str(REPO / "app" / "ui.py"), default_timeout=120)
    at.session_state["nav"] = page
    at.session_state["demo"] = demo
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def clean(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    assert not [e.value for e in at.error], [e.value for e in at.error]
    return at


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def text(at: AppTest) -> str:
    return "\n".join(str(x.value) for kind in ("markdown", "caption", "info", "warning", "success", "title", "subheader")
                     for x in getattr(at, kind))


@pytest.mark.parametrize("page", [P_HOME, P_PROJ, P_DOCS, P_3M])
def test_every_page_renders(page, world):
    at = clean(app(page, project=world["pid"]))
    assert BADGE not in text(at)
    assert sum(b.proto.type == "primary" for b in at.button) <= 1          # mỗi màn hình tối đa 1 nút primary


@pytest.mark.parametrize("page", [P_HOME, P_PROJ, P_DOCS, P_3M])
def test_demo_badge_on_every_page(page, world):
    assert BADGE in text(clean(app(page, demo=True, project=world["pid"])))


def test_project_page_done_shows_video_brain_benchmark(world):
    at = clean(app(P_PROJ, project=world["pid"]))
    t = text(at)
    assert "🎞 Video có nhãn vẽ lại" in t and "chuẩn (ground truth)" in t and "🧠 Kaizen Brain" in t
    assert any(e.label == "📏 Benchmark" for e in at.expander) and any(e.label == "📄 Tài liệu đã dùng" for e in at.expander)
    assert "Baseline rule-based, chưa huấn luyện model" in t and "Model vision: base" in t
    assert len(at.get("video")) == 1


def test_demo_mode_simulates_without_pipeline(world, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("demo mode không được gọi pipeline thật")
    monkeypatch.setattr(pipeline, "start_job", boom)
    monkeypatch.setattr(pipeline, "run_project", boom)
    n = len(pipeline.CALLS)
    at = clean(app(P_PROJ, demo=True, project=world["pid"]))
    assert "Bấm **▶ Chạy full luồng** để xem kết quả." in text(at) and not at.get("video")
    at = clean(button(at, "▶ Chạy full luồng").click().run())
    assert len(pipeline.CALLS) == n and world["pid"] in at.session_state["revealed"]
    assert len(at.get("video")) == 1 and "chuẩn (ground truth)" in text(at)


def test_demo_mode_not_ready_project_offers_real_run(world):
    at = clean(app(P_PROJ, demo=True, project="VID_rong"))
    assert "Model hiện tại: base" in text(at)                             # video mới: chạy thật + ghi chú model
    proj = projects.load_project(world["pid"])
    shutil.copytree(projects.project_dir(world["pid"]), projects.project_dir("U900_V009"))
    projects.save_project({**proj, "id": "U900_V009"})
    projects.media_path("U900_V009", "demo.mp4").unlink()
    at = clean(app(P_PROJ, demo=True, project="U900_V009"))
    assert "Dự án này chưa tính sẵn — chạy thật?" in text(at) and button(at, "▶ Chạy thật")
    shutil.rmtree(projects.project_dir("U900_V009"))


def test_demo_off_run_button_calls_pipeline(world, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "start_job", lambda pid, **k: calls.append(pid) or 123)
    at = clean(app(P_PROJ, project=world["pid"]))
    button(at, "▶ Chạy full luồng").click().run()
    assert calls == [world["pid"]]


def test_job_lock_disables_other_project(world):
    projects.acquire_job("du-an-khac", os.getppid())
    try:
        at = clean(app(P_PROJ, project=world["pid"]))
        assert button(at, "▶ Chạy full luồng").disabled and "Đang chạy dự án **du-an-khac**" in text(at)
    finally:
        projects.release_job(os.getppid())


def test_3m_page_two_videos_r1_skipped_and_review(world):
    from kaizen.analyzer import aggregate
    agg = aggregate.aggregate_3m([world["pid"], "U900_V002"])
    assert agg.ok
    at = clean(app(P_3M, agg_id=agg.agg_id))
    t = text(at)
    assert "Mura — biến động giữa các video" in t and "⏸ <b>R1</b> — bỏ qua" in t and "R6" in t
    assert len(at.get("plotly_chart")) == 2
    at.text_area[0].set_value("Ghi chú thử")
    button(at, "💾 Lưu ghi chú đánh giá").click().run()
    assert aggregate.load_aggregate(agg.agg_id).review["note"] == "Ghi chú thử"


def test_home_soft_error_one_video(world):
    at = app(P_HOME, agg_error="cần ≥2 video đã chạy xong để tính 3M tổng hợp (hiện có 1)")
    assert "cần ≥2 video" in text(clean(at))

"""Smoke test UI bằng streamlit AppTest (headless, không cần trình duyệt). Bỏ qua nếu chưa có run nào."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs" / "runs"
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

runs = sorted(p.name for p in RUNS.glob("*/") if (p / "manifest.json").exists()) if RUNS.exists() else []
needs_runs = pytest.mark.skipif(not runs, reason="chưa có run trong outputs/runs/")


def _app(run_id: str | None = None) -> AppTest:
    at = AppTest.from_file(str(REPO / "app" / "ui.py"), default_timeout=120)
    if run_id:
        at.session_state["run_sel"] = run_id
    return at.run()


def _assert_clean(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]
    errs = [e.value for e in at.error]
    assert not errs, errs
    assert len(at.tabs) == 8


@needs_runs
@pytest.mark.parametrize("run_id", runs)
def test_every_run_renders_all_tabs(run_id):
    _assert_clean(_app(run_id))


@needs_runs
def test_missing_explained_shows_warning(tmp_path):
    run_id = runs[-1]
    p = RUNS / run_id / "explained" / "explained.json"
    if not p.exists():
        pytest.skip("run không có explained.json")
    bak = tmp_path / "explained.json"
    shutil.move(p, bak)
    try:
        at = _app(run_id)
        _assert_clean(at)
        assert any("chưa chạy 07" in w.value for w in at.warning)
    finally:
        shutil.move(bak, p)


def test_fmt_keeps_integer_zeros():
    sys.path.insert(0, str(REPO / "app"))
    from ui_utils import fmt, val
    assert fmt(18660.0, " px", 0) == "18660 px"
    assert fmt(77.90, " s") == "77.9 s"
    assert fmt(0.0, " s") == "0 s"
    assert fmt(None) == "—"
    assert val({"x": {"value": 3, "unit": "s"}}, "x") == 3 and val({}, "x") is None


@needs_runs
def test_deep_link_query_params():
    at = AppTest.from_file(str(REPO / "app" / "ui.py"), default_timeout=120)
    at.query_params["run"] = runs[0]
    at.run()
    _assert_clean(at)
    assert at.session_state["run_sel"] == runs[0]

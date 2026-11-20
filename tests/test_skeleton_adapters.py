"""Skeleton Engine: 3 adapter (ha4m / video / frames) + registry + extract→features→events trên video tổng hợp.

Không cần dataset/mạng. Backend pose giả (DotBackend: tìm chấm màu) giúp test chạy cả khi chưa có models/*.task;
nếu có model MediaPipe thì chạy thêm 1 đường thật.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from kaizen.core.schemas import EVENT_COLUMNS, FEATURE_COLUMNS, KeypointsMeta
from kaizen.skeleton import (SOURCES, compute_features, detect_events, extract, open_source,
                             register_source)
from kaizen.skeleton.pose import MP_NAMES
from kaizen.skeleton.sources import FramesSource, HA4MSource, VideoSource, parse_uri

REPO = Path(__file__).resolve().parents[1]
FPS = 15
W, H = 320, 240


def _img(idx: int) -> np.ndarray:
    """Ảnh mã hoá idx vào kênh B của pixel (0,0) để kiểm tra đúng frame."""
    img = np.full((H, W, 3), 30, np.uint8)
    img[0, 0, 0] = idx % 256
    return img


def _dot_xy(i: int, n: int) -> tuple[int, int]:
    """Quỹ đạo chấm đỏ (cổ tay phải): đi ngang qua khung rồi dừng nửa sau."""
    u = min(i / (n / 2), 1.0)
    return int(40 + u * 220), 120


def make_video(path: Path, seconds: float = 2.0) -> int:
    """mp4 tổng hợp `seconds` giây @FPS: chấm đỏ (tay phải) di chuyển, chấm xanh (tay trái) đứng yên."""
    n = int(seconds * FPS)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        img = np.full((H, W, 3), 30, np.uint8)
        cv2.circle(img, _dot_xy(i, n), 8, (0, 0, 255), -1)
        cv2.circle(img, (80, 60), 8, (0, 255, 0), -1)
        vw.write(img)
    vw.release()
    return n


class DotBackend:
    """PoseBackend giả: chấm đỏ → r_wrist, chấm xanh → l_wrist (score 0.9), còn lại NaN."""
    names = MP_NAMES

    def __call__(self, frame_bgr: np.ndarray, ts_ms: int) -> np.ndarray:
        out = np.full((len(self.names), 3), np.nan, np.float32)
        b, g, r = (frame_bgr[..., k].astype(int) for k in range(3))
        for name, mask in (("r_wrist", (r > 150) & (g < 100)), ("l_wrist", (g > 150) & (r < 100))):
            ys, xs = np.nonzero(mask)
            if len(xs):
                out[self.names.index(name)] = (xs.mean(), ys.mean(), 0.9)
        return out


def station() -> dict:
    return {"fps": FPS, "px_per_cm": None, "moving_speed_px_s": 60, "min_event_s": 0.2, "idle_threshold_s": 3.0,
            "zones": {"bin_a": [[200, 90], [300, 90], [300, 150], [200, 150]],
                      "assembly_area": [[20, 30], [140, 30], [140, 90], [20, 90]]},
            "steps": [{"id": 1, "name": "Pick A", "zone": "bin_a"}], "cycle_time_target_s": None}


# ---------------------------------------------------------------- adapters
def test_frames_adapter_numeric_order_and_fps(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for idx in (10, 9, 100, 2, 1):
        cv2.imwrite(str(d / f"frame_{idx}.png"), _img(idx))
    (d / "Thumbs.db").write_text("x")
    s = open_source(f"frames://{d}?fps=25")
    assert isinstance(s, FramesSource) and s.fps() == 25.0
    got = list(s.frames())
    assert [i for i, _ in got] == [1, 2, 9, 10, 100]                 # sort SỐ, không sort chuỗi
    assert [int(im[0, 0, 0]) for _, im in got] == [1, 2, 9, 10, 100]  # đúng ảnh tương ứng
    assert [i for i, _ in s.frames(stride=2)] == [1, 9, 100]
    assert s.meta()["n_frames"] == 5 and s.meta()["width"] == W


def test_frames_adapter_requires_fps(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    cv2.imwrite(str(d / "0.png"), _img(0))
    with pytest.raises(ValueError, match="fps"):
        open_source(f"frames://{d}")


def test_ha4m_adapter_order_fps_meta(tmp_path):
    vd = tmp_path / "ha4m" / "IDU900V003"
    cdir = vd / "Color" / "000000000000"
    cdir.mkdir(parents=True)
    for idx in (12, 3, 0, 101):
        cv2.imwrite(str(cdir / f"FrameID{idx:06d}_DeviceTimeStamp{1_000_000 + idx * 33333}us.png"), _img(idx))
    (vd / "Labels.txt").write_text("0 1 0\r\n1 1 0\r\n")
    s = open_source(f"ha4m://{tmp_path / 'ha4m'}?subject=U900&trial=V003")
    assert isinstance(s, HA4MSource) and s.fps() == 30.0
    assert [i for i, _ in s.frames()] == [0, 3, 12, 101]
    m = s.meta()
    assert (m["subject"], m["trial"], m["stem"], m["labels_available"]) == ("U900", "V003", "U900_V003", True)
    assert open_source(f"ha4m://{tmp_path / 'ha4m'}?subject=U900&trial=V003&fps=15").fps() == 15.0


def test_video_adapter_order_fps(tmp_path):
    p = tmp_path / "U007_V002.mp4"
    n = make_video(p)
    s = open_source(f"video://{p}")
    assert isinstance(s, VideoSource) and s.fps() == pytest.approx(FPS)
    idx = [i for i, _ in s.frames()]
    assert idx == list(range(n))
    assert [i for i, _ in s.frames(stride=3, max_frames=4)] == [0, 3, 6, 9]
    m = s.meta()
    assert (m["subject"], m["trial"], m["width"], m["height"]) == ("U007", "V002", W, H)
    assert isinstance(open_source(str(p)), VideoSource)               # không scheme: file → video


def test_registry_extensible_and_errors(tmp_path):
    assert set(SOURCES) >= {"ha4m", "video", "frames"}
    assert parse_uri("ha4m://data/ha4m?subject=U001&trial=V001") == ("ha4m", "data/ha4m",
                                                                      {"subject": "U001", "trial": "V001"})
    with pytest.raises(ValueError, match="scheme"):
        open_source("rtsp://cam1")

    class Synthetic:
        def __init__(self, n):
            self.uri, self.n = f"synthetic://{n}", n

        def frames(self, stride=1, max_frames=0):
            for i in range(0, self.n, stride):
                yield i, _img(i)

        def fps(self):
            return 10.0

        def meta(self):
            return {"stem": "syn", "width": W, "height": H, "frames_dir": None, "video": None}

    register_source("synthetic", lambda path, params, uri: Synthetic(int(path)))
    try:
        s = open_source("synthetic://4")
        assert [i for i, _ in s.frames()] == [0, 1, 2, 3]
    finally:
        SOURCES.pop("synthetic")


# ---------------------------------------------------------------- extract → features → events
def _pipeline(tmp_path, model, model_name=None):
    vid = tmp_path / "demo.mp4"
    n = make_video(vid)
    npz = extract(open_source(f"video://{vid}"), model, 1, tmp_path / "kp", model_name=model_name,
                  log=lambda *_: None)
    d = np.load(npz)
    assert d["kpts"].shape == (n, len(MP_NAMES), 3) and d["kpts"].dtype == np.float32
    assert np.allclose(d["t"], np.arange(n) / FPS) and list(d["frame"]) == list(range(n))
    KeypointsMeta.model_validate(json.loads(str(d["meta"])))
    feat = compute_features(npz, station())
    assert set(FEATURE_COLUMNS) <= set(feat.columns) and len(feat) == n
    ev = detect_events(feat, station())
    assert list(ev.columns) == EVENT_COLUMNS
    return npz, feat, ev


def test_synthetic_video_extract_features_events(tmp_path):
    npz, feat, ev = _pipeline(tmp_path, DotBackend(), "dot")
    assert npz.name == "demo_dot.npz"
    r = ev[ev["hand"] == "right"]
    assert (r["motion"] == "moving").any() and (r["motion"] == "still").any()   # đi rồi dừng
    assert "bin_a" in set(r["zone"])                                            # dừng trong khay
    assert set(ev.loc[ev["hand"] == "left", "zone"]) == {"assembly_area"}       # tay trái đứng yên ở vùng lắp
    assert ev["end_s"].max() == pytest.approx(len(feat) / FPS, abs=1e-3)


@pytest.mark.skipif(not (REPO / "models" / "pose_landmarker_full.task").exists(),
                    reason="chưa có models/*.task (chạy setup.sh)")
def test_synthetic_video_with_real_mediapipe(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe")
    monkeypatch.chdir(REPO)                                  # MediaPipeBackend đọc models/ tương đối
    _, feat, ev = _pipeline(tmp_path, "mediapipe")
    assert feat["rw_x"].isna().all() or len(ev)               # không có người → toàn NaN, vẫn không lỗi
    assert set(pd.unique(ev["motion"])) <= {"moving", "still", "lost"}

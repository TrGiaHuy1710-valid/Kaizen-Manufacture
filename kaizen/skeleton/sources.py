"""Nguồn video (adapter) sau 1 interface `Source` — tầng 2 chỉ thấy (frame_idx, ảnh BGR), fps, meta.

URI:
    ha4m://data/ha4m?subject=U001&trial=V001[&fps=30]   dataset HA4M (Color/<serial>/FrameID…png, Labels.txt)
    video://data/raw/test.mp4                             file video (OpenCV), fps đọc từ container (0 → 30)
    frames://path/to/dir?fps=30                           thư mục chuỗi ảnh bất kỳ, sort theo SỐ trong tên; fps bắt buộc
    (không có scheme: file → video, thư mục → frames)
Đường dẫn tuyệt đối: ha4m:///abs/path?... (3 dấu /).
Thêm nguồn mới = viết class có frames()/fps()/meta() + thêm 1 entry vào SOURCES (scheme → factory(path, params)).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterator, Protocol, runtime_checkable
from urllib.parse import parse_qsl

import cv2
import numpy as np

from kaizen.skeleton import ha4m_io

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
STEM_RE = re.compile(r"(U\d+)[_\-]?(V\d+)", re.I)


@runtime_checkable
class Source(Protocol):
    """Nguồn frame: yield (frame_idx, bgr) theo thứ tự SỐ frame; fps; meta (subject, trial, labels_available...)."""
    uri: str

    def frames(self, stride: int = 1, max_frames: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """yield (frame_idx, ảnh BGR) theo thứ tự số frame."""

    def fps(self) -> float:
        """fps của nguồn (t = frame_idx / fps)."""

    def meta(self) -> dict:
        """stem, subject, trial, labels_available, width, height, n_frames, video|frames_dir."""


def parse_uri(uri: str) -> tuple[str, str, dict[str, str]]:
    """'scheme://path?k=v' → (scheme, path, params). Không có scheme → suy từ đường dẫn (file video / thư mục)."""
    s = str(uri)
    if "://" in s:
        scheme, rest = s.split("://", 1)
    else:
        scheme, rest = "", s
    path, _, query = rest.partition("?")
    params = dict(parse_qsl(query, keep_blank_values=True))
    if not scheme:
        scheme = "frames" if Path(path).is_dir() else "video"
    return scheme.lower(), path, params


def guess_stem(path: str | Path) -> str:
    """.../IDU001V001/Color/<serial> → U001_V001; video U001_V001.mp4 → U001_V001; không khớp → tên file/thư mục."""
    p = Path(path)
    for part in reversed(p.resolve().parts + p.parts):
        m = re.match(r"^ID(U\d+)(V\d+)$", part)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
    return p.stem if p.suffix.lower() in VIDEO_EXT else p.name


def _ids_from_stem(st: str) -> tuple[str | None, str | None]:
    m = STEM_RE.search(st)
    return (m.group(1).upper(), m.group(2).upper()) if m else (None, None)


class VideoSource:
    """File video qua OpenCV. frame_idx = số thứ tự frame đọc được (0, 1, 2...)."""

    def __init__(self, path: str | Path, fps: float | None = None, uri: str | None = None):
        self.path = str(path)
        self.uri = uri or f"video://{self.path}"
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"không thấy video {self.path}")
        cap = cv2.VideoCapture(self.path)
        self._fps = float(fps) if fps else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._w, self._h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @classmethod
    def from_uri(cls, path: str, params: dict, uri: str) -> "VideoSource":
        """Factory cho registry."""
        return cls(path, float(params["fps"]) if params.get("fps") else None, uri)

    def frames(self, stride: int = 1, max_frames: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """yield (frame_idx, bgr) — đọc tuần tự, giữ 1 trên `stride` frame."""
        cap = cv2.VideoCapture(self.path)
        i, n = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, frame
                n += 1
                if max_frames and n >= max_frames:
                    break
            i += 1
        cap.release()

    def fps(self) -> float:
        """fps của container (0 → 30) hoặc tham số ?fps=."""
        return self._fps

    def meta(self) -> dict:
        """kind, stem, subject/trial (nếu tên file có U###_V###), kích thước, số frame."""
        st = guess_stem(self.path)
        sub, tri = _ids_from_stem(st)
        return {"kind": "video", "uri": self.uri, "stem": st, "subject": sub, "trial": tri, "labels_available": False,
                "video": self.path, "frames_dir": None, "width": self._w, "height": self._h, "n_frames": self._n}


class FramesSource:
    """Thư mục chuỗi ảnh; frame_idx = số trong tên file (FrameID###### hoặc số cuối), sort SỐ không sort chuỗi."""

    def __init__(self, frames_dir: str | Path, fps: float, uri: str | None = None):
        if not fps:
            raise ValueError("frames:// cần tham số fps (vd frames://dir?fps=30) — t = frame_idx / fps")
        self.frames_dir = str(frames_dir)
        self.uri = uri or f"frames://{self.frames_dir}?fps={fps:g}"
        self._fps = float(fps)
        self._files = ha4m_io.frames_dir_files(self.frames_dir)
        if not self._files:
            raise FileNotFoundError(f"Không có ảnh trong {self.frames_dir}")

    @classmethod
    def from_uri(cls, path: str, params: dict, uri: str) -> "FramesSource":
        """Factory cho registry (fps bắt buộc)."""
        return cls(path, float(params["fps"]) if params.get("fps") else 0.0, uri)

    def frames(self, stride: int = 1, max_frames: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """yield (frame_idx, bgr); frame bị bỏ qua theo stride KHÔNG được decode."""
        yield from ha4m_io.frames_dir_iter(self.frames_dir, stride, max_frames)

    def fps(self) -> float:
        """fps khai báo (?fps=)."""
        return self._fps

    def meta(self) -> dict:
        """kind, stem (suy từ đường dẫn HA4M nếu có), kích thước ảnh đầu, số frame."""
        st = guess_stem(self.frames_dir)
        sub, tri = _ids_from_stem(st)
        h, w = cv2.imread(str(self._files[0][1])).shape[:2]
        return {"kind": "frames", "uri": self.uri, "stem": st, "subject": sub, "trial": tri, "labels_available": False,
                "video": None, "frames_dir": self.frames_dir, "width": w, "height": h, "n_frames": len(self._files)}


class HA4MSource:
    """1 video của dataset HA4M (quét phòng thủ qua ha4m_io): Color/<serial>/ + Labels*.txt."""

    def __init__(self, root: str | Path, subject: str, trial: str, fps: float = 30.0, uri: str | None = None):
        self.root, self.subject, self.trial = str(root), subject.upper(), trial.upper()
        self.uri = uri or f"ha4m://{self.root}?subject={self.subject}&trial={self.trial}"
        self._fps = float(fps)
        self.frames_dir = str(ha4m_io.modality_dir(self.root, self.subject, self.trial, "Color"))
        self._files = ha4m_io.frame_files(self.root, self.subject, self.trial)

    @classmethod
    def from_uri(cls, path: str, params: dict, uri: str) -> "HA4MSource":
        """Factory cho registry: cần ?subject=&trial= (trial mặc định V001)."""
        if not params.get("subject"):
            raise ValueError("ha4m:// cần ?subject=U001&trial=V001")
        return cls(path or ha4m_io.DEFAULT_ROOT, params["subject"], params.get("trial", "V001"),
                   float(params.get("fps") or 30.0), uri)

    def frames(self, stride: int = 1, max_frames: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """yield (frame_idx theo FrameID, bgr) theo thứ tự số."""
        yield from ha4m_io.frames_iter(self.root, self.subject, self.trial, stride, max_frames)

    def fps(self) -> float:
        """fps HA4M (30, xác nhận bằng DeviceTimeStamp) hoặc ?fps=."""
        return self._fps

    def meta(self) -> dict:
        """subject, trial, labels_available, frames_dir, kích thước, số frame."""
        h, w = cv2.imread(str(self._files[0][2])).shape[:2] if self._files else (0, 0)
        return {"kind": "ha4m", "uri": self.uri, "stem": f"{self.subject}_{self.trial}", "subject": self.subject,
                "trial": self.trial,
                "labels_available": ha4m_io.labels_path(self.root, self.subject, self.trial) is not None,
                "video": None, "frames_dir": self.frames_dir, "width": w, "height": h, "n_frames": len(self._files),
                "data_root": self.root}


SourceFactory = Callable[[str, dict, str], Source]
SOURCES: dict[str, SourceFactory] = {
    "ha4m": HA4MSource.from_uri,
    "video": VideoSource.from_uri,
    "frames": FramesSource.from_uri,
}


def register_source(scheme: str, factory: SourceFactory) -> None:
    """Thêm nguồn mới: factory(path, params, uri) -> Source."""
    SOURCES[scheme.lower()] = factory


def open_source(uri: str) -> Source:
    """Mở nguồn theo URI (xem docstring module)."""
    scheme, path, params = parse_uri(uri)
    if scheme not in SOURCES:
        raise ValueError(f"scheme không hỗ trợ: {scheme!r} (có: {sorted(SOURCES)})")
    return SOURCES[scheme](path, params, uri)

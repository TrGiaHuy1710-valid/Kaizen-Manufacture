"""Dataset cho model phân đoạn bước: trial HA4M có ĐỦ keypoints (.npz) + nhãn (Labels*.txt).

Lớp = cặp (action_id, repetition) đọc từ nhãn ("0" = don't care/nền, "2.1" = bước 2 lần lặp thứ 2…), nên model
tách được cả các lần lặp liên tiếp của cùng một bước (2.0 / 2.1 / 2.2) — đúng khoá mà 04 eval dùng.
Chia train/val THEO SUBJECT (không để cùng một người ở cả hai tập).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kaizen.skeleton import ha4m_io
from kaizen.train.features import frame_features, load_npz

IGNORE = -100


def class_key(action_id: int, repetition: int) -> str:
    return "0" if int(action_id) == 0 else f"{int(action_id)}.{int(repetition)}"


def parse_key(key: str) -> tuple[int, int]:
    if key == "0":
        return 0, 0
    a, r = key.split(".")
    return int(a), int(r)


def sort_keys(keys) -> list[str]:
    """'0' trước, rồi theo (action, rep)."""
    return sorted(set(keys), key=lambda k: parse_key(k))


@dataclass
class Trial:
    stem: str                       # U001_V001
    subject: str
    npz: Path
    feats: np.ndarray               # (T, D)
    keys: list[str]                 # nhãn theo frame (cùng độ dài T); None-frame → ""
    t: np.ndarray                   # (T,) giây
    frame: np.ndarray               # (T,) frame index gốc
    fps: float
    pose_model: str = ""


@dataclass
class TrialRef:
    stem: str
    subject: str
    npz: Path | None
    has_labels: bool


def scan(data_root: str, kp_dir: str | Path, pose_model: str = "mediapipe") -> list[TrialRef]:
    """Mọi trial trong data_root + trạng thái keypoints/nhãn (để báo thiếu gì)."""
    out = []
    for s, t in ha4m_io.list_trials(data_root):
        p = Path(kp_dir) / f"{s}_{t}_{pose_model}.npz"
        out.append(TrialRef(f"{s}_{t}", s, p if p.exists() else None, ha4m_io.labels_path(data_root, s, t) is not None))
    return out


def load_trial(data_root: str | None, ref: TrialRef, fps: float = 30.0, with_labels: bool = True) -> Trial:
    """Đặc trưng + nhãn theo frame của 1 trial (nhãn căn theo frame index của .npz)."""
    kp = load_npz(ref.npz)
    feats = frame_features(kp, fps)
    keys = [""] * len(feats)
    if with_labels and ref.has_labels:
        s, t = ref.stem.split("_")
        fl = ha4m_io.load_frame_labels(data_root, s, t)
        lut = {int(f): class_key(a, r) for f, a, r in fl[["frame", "action_id", "repetition"]].itertuples(index=False)}
        keys = [lut.get(int(f), "") for f in kp["frame"]]
    return Trial(ref.stem, ref.subject, Path(ref.npz), feats, keys, kp["t"], kp["frame"], fps,
                 kp["meta"].get("model", ""))


def split_subjects(subjects: list[str], val_subjects: list[str] | None = None, val_frac: float = 0.15,
                   seed: int = 0) -> tuple[list[str], list[str]]:
    """(train, val) subject. val_subjects tường minh > val_frac (ít nhất 1 subject nếu có ≥ 2)."""
    subs = sorted(set(subjects))
    if val_subjects:
        val = [s for s in subs if s in set(val_subjects)]
        return [s for s in subs if s not in val], val
    if len(subs) < 2 or val_frac <= 0:
        return subs, []
    rng = random.Random(seed)
    shuffled = subs[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(subs) * val_frac))
    val = sorted(shuffled[:n_val])
    return [s for s in subs if s not in val], val


def group_folds(subjects: list[str], k: int, seed: int = 0) -> list[list[str]]:
    """Chia subject thành k fold (k = 0 hoặc ≥ số subject → leave-one-subject-out)."""
    subs = sorted(set(subjects))
    if k <= 0 or k >= len(subs):
        return [[s] for s in subs]
    rng = random.Random(seed)
    shuffled = subs[:]
    rng.shuffle(shuffled)
    return [sorted(shuffled[i::k]) for i in range(k)]


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "Normalizer":
        x = np.concatenate(arrays, axis=0).astype(np.float64)
        std = x.std(0)
        return cls(x.mean(0).astype(np.float32), np.where(std < 1e-6, 1.0, std).astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32))


@dataclass
class Encoded:
    """1 chuỗi đã chuẩn hoá + nhãn index (IGNORE cho frame không nhãn), đã lấy mẫu 1/sample_rate."""
    stem: str
    x: np.ndarray                   # (T', D)
    y: np.ndarray                   # (T',) int64
    extra: dict = field(default_factory=dict)


def encode(trial: Trial, classes: list[str], norm: Normalizer, sample_rate: int = 1) -> Encoded:
    idx = {k: i for i, k in enumerate(classes)}
    x = norm(trial.feats[::sample_rate])
    y = np.array([idx.get(k, IGNORE) if k else IGNORE for k in trial.keys[::sample_rate]], np.int64)
    return Encoded(trial.stem, x, y)

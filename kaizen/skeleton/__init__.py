"""Tầng 1 — Skeleton Engine: nguồn video (adapter) → pose → keypoints .npz → features → events.

Interface duy nhất tầng 2 được phép dùng:
    open_source(uri) -> Source                     # ha4m://… | video://… | frames://…?fps=
    extract(source, model, stride, out) -> Path    # .npz schema cũ
    compute_features(npz, station) -> DataFrame    # stage 02
    detect_events(features, station) -> DataFrame  # stage 03
"""
from kaizen.skeleton.events import detect_events
from kaizen.skeleton.extract import extract
from kaizen.skeleton.features import compute_features
from kaizen.skeleton.pose import POSE_BACKENDS, PoseBackend, get_pose_backend
from kaizen.skeleton.sources import SOURCES, Source, open_source, register_source

__all__ = ["POSE_BACKENDS", "SOURCES", "PoseBackend", "Source", "compute_features", "detect_events", "extract",
           "get_pose_backend", "open_source", "register_source"]

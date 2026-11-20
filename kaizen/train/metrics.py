"""Chỉ số chuẩn cho phân đoạn hành động theo thời gian (như MS-TCN, Farha & Gall 2019) — numpy thuần.

- frame accuracy (bỏ frame không nhãn)
- Edit score: 100·(1 − Levenshtein(chuỗi đoạn dự đoán, chuỗi đoạn thật) / max độ dài)
- F1@{10,25,50}: đoạn dự đoán đúng nếu cùng lớp và IoU ≥ ngưỡng với 1 đoạn thật chưa ghép
Lớp nền (bg) không tính là đoạn.
"""
from __future__ import annotations

import numpy as np


def segments(labels, bg=None) -> list[tuple[object, int, int]]:
    """[(lớp, start, end_exclusive)] các đoạn liên tiếp, bỏ lớp bg."""
    out = []
    labels = list(labels)
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if labels[i] != bg:
            out.append((labels[i], i, j))
        i = j
    return out


def frame_accuracy(pred, gt, ignore=None) -> float:
    p, g = np.asarray(pred), np.asarray(gt)
    m = np.ones(len(g), bool) if ignore is None else g != ignore
    return float((p[m] == g[m]).mean() * 100) if m.any() else 0.0


def _levenshtein(a: list, b: list) -> int:
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return int(d[len(b)])


def edit_score(pred, gt, bg=None) -> float:
    a = [s[0] for s in segments(pred, bg)]
    b = [s[0] for s in segments(gt, bg)]
    if not a and not b:
        return 100.0
    return float((1 - _levenshtein(a, b) / max(len(a), len(b))) * 100)


def f1_at(pred, gt, overlap: float, bg=None) -> float:
    ps, gs = segments(pred, bg), segments(gt, bg)
    used = [False] * len(gs)
    tp = 0
    for lab, s, e in ps:
        best, bj = 0.0, -1
        for j, (gl, gs_, ge) in enumerate(gs):
            if gl != lab or used[j]:
                continue
            inter = max(0, min(e, ge) - max(s, gs_))
            union = max(e, ge) - min(s, gs_)
            iou = inter / union if union else 0.0
            if iou > best:
                best, bj = iou, j
        if bj >= 0 and best >= overlap:
            tp += 1
            used[bj] = True
    fp, fn = len(ps) - tp, len(gs) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * prec * rec / (prec + rec) * 100) if prec + rec else 0.0


def segment_report(pred, gt, bg=None, ignore=None) -> dict:
    """{acc, edit, f1@10, f1@25, f1@50} (thang 0–100). Frame ignore bị bỏ khỏi cả pred và gt."""
    p, g = np.asarray(pred), np.asarray(gt)
    if ignore is not None:
        keep = g != ignore
        p, g = p[keep], g[keep]
    return {"acc": round(frame_accuracy(p, g), 2), "edit": round(edit_score(p, g, bg), 2),
            **{f"f1@{int(o * 100)}": round(f1_at(p, g, o, bg), 2) for o in (0.10, 0.25, 0.50)}}


def mean_reports(reports: list[dict]) -> dict:
    if not reports:
        return {}
    return {k: round(float(np.mean([r[k] for r in reports])), 2) for k in reports[0]}

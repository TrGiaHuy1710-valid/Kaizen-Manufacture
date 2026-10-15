"""Khảo sát dữ liệu HA4M thật (stage 00): cây thư mục, schema Labels.txt, pattern frame, skeleton, fps ước lượng."""
from __future__ import annotations

import collections
import re
from pathlib import Path

import cv2
import numpy as np

from kaizen.skeleton import ha4m_io


def tree(d: Path, depth: int = 2, prefix: str = "") -> list[str]:
    """Cây thư mục `depth` cấp (đếm mục con, 3 file đầu)."""
    lines = []
    if depth == 0:
        return lines
    kids = sorted(d.iterdir())
    dirs = [k for k in kids if k.is_dir()]
    files = [k for k in kids if not k.is_dir()]
    for k in dirs:
        n = sum(1 for _ in k.iterdir())
        lines.append(f"{prefix}{k.name}/  ({n} mục)")
        lines += tree(k, depth - 1, prefix + "    ")
    if files:
        shown = ", ".join(f.name for f in files[:3])
        lines.append(f"{prefix}[{len(files)} file] {shown}{' ...' if len(files) > 3 else ''}")
    return lines


def guess_label_schema(lines: list[str]) -> list[str]:
    """Đoán schema Labels.txt từ các dòng đầu → các dòng Markdown."""
    toks = [re.split(r"\s+|,|;", ln.strip()) for ln in lines if ln.strip()]
    ncol = collections.Counter(len(t) for t in toks).most_common(1)[0][0]
    sep = "tab" if "\t" in lines[0] else ("," if "," in lines[0] else "space")
    allnum = all(x.lstrip("-").isdigit() for t in toks for x in t)
    first = [int(t[0]) for t in toks if t[0].isdigit()]
    consecutive = all(b - a == 1 for a, b in zip(first, first[1:]))
    return [f"- Phân cách: **{sep}**, {ncol} cột, toàn số nguyên: {allnum}",
            f"- Cột 1 tăng liên tục +1 → **frame index** (nhãn theo frame, không phải timestamp): {consecutive}",
            "- Cột 2 = action_id (0 = don't care, 1..12), cột 3 = repetition (0,1,2 cho action lặp 2/3/11)",
            "- TrialID KHÔNG nằm trong file: mỗi video có Labels.txt riêng trong thư mục `ID<Subject><Trial>`"]


def inspect_report(data_root: str, subject: str = "", trial: str = "") -> list[str]:
    """Các dòng Markdown của outputs/inspect_report.md. FileNotFoundError nếu không thấy video nào."""
    trials = ha4m_io.list_trials(data_root)
    if not trials:
        raise FileNotFoundError(f"Không thấy thư mục video (có Color*) nào trong {data_root}")
    sub, tri = (subject, trial) if subject and trial else trials[0]
    vdir = ha4m_io.video_dir(data_root, sub, tri)
    md = [f"# Inspect report — HA4M", "", f"- data_root: `{data_root}`",
          f"- Số video tìm thấy: **{len(trials)}** → {', '.join(f'{s}_{t}' for s, t in trials[:20])}"
          + (" ..." if len(trials) > 20 else ""),
          f"- Video khảo sát: `{vdir.name}` (subject=`{sub}`, trial=`{tri}`)", "",
          "## Cây thư mục 2 cấp", "```", vdir.name + "/"] + ["    " + ln for ln in tree(vdir, 2)] + ["```", ""]

    # Labels
    lp = ha4m_io.labels_path(data_root, sub, tri)
    fl = None
    if lp is None:
        md += ["## Labels", "- **KHÔNG có Labels*.txt** — trích keypoints vẫn được; oracle/eval sẽ bỏ qua video này.", ""]
    else:
        raw = lp.read_text(errors="replace").splitlines()
        crlf = b"\r\n" in lp.read_bytes()
        md += [f"## {lp.name}", f"- Số dòng: {len(raw)}; CRLF: {crlf}",
               "```"] + raw[:10] + ["```", "Schema suy ra:"] + guess_label_schema(raw[:200]) + [""]
        fl = ha4m_io.load_frame_labels(data_root, sub, tri)
        seg = ha4m_io.labels_to_segments(fl, keep_dont_care=True)
        md += ["Các đoạn liên tiếp (action_id, repetition):", "",
               "| action_id | rep | start_frame | end_frame | n_frames | tên (paper) |", "|---|---|---|---|---|---|"]
        for r in seg.itertuples():
            md.append(f"| {r.action_id} | {r.repetition} | {r.start_frame} | {r.end_frame} | "
                      f"{r.end_frame - r.start_frame + 1} | {r.action_name} |")
        md.append("")
    no_lab = [f"{s}_{t}" for s, t in trials if ha4m_io.labels_path(data_root, s, t) is None]
    if no_lab:
        md += [f"- Video thiếu nhãn trong data_root: {', '.join(no_lab)}", ""]

    # Color frames
    files = ha4m_io.frame_files(data_root, sub, tri, "Color")
    cdir = ha4m_io.modality_dir(data_root, sub, tri, "Color")
    others = [p.name for p in cdir.iterdir() if not ha4m_io.FRAME_RE.search(p.name)]
    md += ["## Color frames", f"- Thư mục: `{cdir.relative_to(vdir)}` (thư mục con = serial camera)",
           "- 5 file đầu:", "```"] + [p.name for _, _, p in files[:5]] + ["```",
           f"- Pattern: `FrameID<6 số>_DeviceTimeStamp<micro giây>us.png`; file lạ bị bỏ qua: {others}",
           f"- Số frame: **{len(files)}** (frame {files[0][0]} → {files[-1][0]}); số dòng Labels: "
           f"{len(fl) if fl is not None else '— (không có nhãn)'}"]
    img = cv2.imread(str(files[len(files) // 2][2]), cv2.IMREAD_UNCHANGED)
    md.append(f"- Kích thước ảnh: **{img.shape[1]}x{img.shape[0]}**, {img.shape[2] if img.ndim == 3 else 1} kênh, {img.dtype}")
    ts = np.array([t for _, t, _ in files], dtype=np.int64)
    dt = np.diff(ts) / 1e6
    pos = dt[dt > 0]
    if (ts < 0).any() or len(pos) == 0:
        md += ["- Tên file không có DeviceTimeStamp → không ước lượng được fps; dùng `fps` trong config (GIẢ ĐỊNH 30).", ""]
    else:
        fps_est = 1.0 / np.median(pos)
        dup = int((dt == 0).sum())
        drops = int((pos > 1.5 * np.median(pos)).sum())
        md += [f"- fps ước lượng từ DeviceTimeStamp: **{fps_est:.2f}** (median Δt = {np.median(pos) * 1000:.2f} ms); "
               f"paper HA4M ghi 30 fps → dùng `fps: 30` trong config",
               f"- Δt = 0 (timestamp trùng): {dup} frame (thường ở đầu video); Δt > 1.5×median (rơi frame): {drops}",
               f"- Thời lượng theo frame_idx/30: {len(files) / 30:.1f} s; theo timestamp: {(ts[-1] - ts[0]) / 1e6:.1f} s", ""]

    # Skeleton
    try:
        sk = ha4m_io.frame_files(data_root, sub, tri, "Skeletons")
    except FileNotFoundError:
        sk = []
        md += ["## Skeletons", "- Không có thư mục Skeletons* (chỉ ảnh hưởng bước đối chứng Kinect).", ""]
    if sk:
        p = sk[min(100, len(sk) - 1)][2]
        lines = p.read_text().splitlines()
        md += ["## Skeletons (Azure Kinect)", f"- Số file: {len(sk)}; ví dụ `{p.name}` ({len(lines) - 1} dòng = joint):",
               "```"] + lines[:5] + ["```",
               "- TSV có header; 32 joint (0–31) theo Azure Kinect Body Tracking; X2DColor/Y2DColor = toạ độ pixel "
               "trên ảnh Color → dùng đối chứng cổ tay (joint 7 = WRIST_LEFT, 14 = WRIST_RIGHT).", ""]

    md += ["## Kết luận cho parser (`src/ha4m_io.py`)",
           "- `list_trials(root)` quét thư mục `ID(U\\d+)(V\\d+)` → (subject, trial).",
           "- `frames_iter` sort theo số `FrameID`, bỏ Thumbs.db.",
           "- `load_labels` gộp frame liên tiếp cùng (action_id, repetition) → đoạn; bỏ action 0.",
           "- Thời gian t = frame_idx / fps (fps = 30, khớp timestamp thiết bị)."]
    return md

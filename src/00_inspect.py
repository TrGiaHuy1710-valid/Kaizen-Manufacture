"""[00] Khảo sát dữ liệu HA4M thật → outputs/inspect_report.md.

In cây thư mục 2 cấp, 10 dòng đầu Labels.txt + đoán schema, pattern tên frame, 1 file skeleton,
số frame, kích thước ảnh, fps ước lượng từ DeviceTimeStamp trong tên file.

    python src/00_inspect.py --data-root data/ha4m [--subject U001 --trial V001]
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write  # noqa: E402
import ha4m_io  # noqa: E402

TAG = "00"


def tree(d: Path, depth: int = 2, prefix: str = "") -> list[str]:
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


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--subject", default="")
    ap.add_argument("--trial", default="")
    ap.add_argument("--out", default="outputs/inspect_report.md")
    a = ap.parse_args()
    out = Path(a.out)
    if not can_write(out, a.force, TAG):
        return

    trials = ha4m_io.list_trials(a.data_root)
    if not trials:
        sys.exit(f"[{TAG}] Không thấy thư mục IDU*V* nào trong {a.data_root}")
    sub, tri = (a.subject, a.trial) if a.subject and a.trial else trials[0]
    vdir = ha4m_io.video_dir(a.data_root, sub, tri)
    md = [f"# Inspect report — HA4M", "", f"- data_root: `{a.data_root}`",
          f"- Số video (IDU*V*) tìm thấy: **{len(trials)}** → {', '.join(s + t for s, t in trials[:20])}",
          f"- Video khảo sát: `{vdir.name}` (subject=`{sub}`, trial=`{tri}`)", "",
          "## Cây thư mục 2 cấp", "```", vdir.name + "/"] + ["    " + ln for ln in tree(vdir, 2)] + ["```", ""]

    # Labels
    raw = (vdir / "Labels.txt").read_text(errors="replace").splitlines()
    crlf = b"\r\n" in (vdir / "Labels.txt").read_bytes()
    md += ["## Labels.txt", f"- Số dòng: {len(raw)}; CRLF: {crlf}",
           "```"] + raw[:10] + ["```", "Schema suy ra:"] + guess_label_schema(raw[:200]) + [""]
    fl = ha4m_io.load_frame_labels(a.data_root, sub, tri)
    seg = ha4m_io.labels_to_segments(fl, keep_dont_care=True)
    md += ["Các đoạn liên tiếp (action_id, repetition):", "",
           "| action_id | rep | start_frame | end_frame | n_frames | tên (paper) |", "|---|---|---|---|---|---|"]
    for r in seg.itertuples():
        md.append(f"| {r.action_id} | {r.repetition} | {r.start_frame} | {r.end_frame} | "
                  f"{r.end_frame - r.start_frame + 1} | {r.action_name} |")
    md.append("")

    # Color frames
    files = ha4m_io.frame_files(a.data_root, sub, tri, "Color")
    cdir = ha4m_io.modality_dir(a.data_root, sub, tri, "Color")
    others = [p.name for p in cdir.iterdir() if not ha4m_io.FRAME_RE.search(p.name)]
    md += ["## Color frames", f"- Thư mục: `{cdir.relative_to(vdir)}` (thư mục con = serial camera)",
           "- 5 file đầu:", "```"] + [p.name for _, _, p in files[:5]] + ["```",
           f"- Pattern: `FrameID<6 số>_DeviceTimeStamp<micro giây>us.png`; file lạ bị bỏ qua: {others}",
           f"- Số frame: **{len(files)}** (frame {files[0][0]} → {files[-1][0]}); số dòng Labels: {len(fl)}"]
    img = cv2.imread(str(files[len(files) // 2][2]), cv2.IMREAD_UNCHANGED)
    md.append(f"- Kích thước ảnh: **{img.shape[1]}x{img.shape[0]}**, {img.shape[2] if img.ndim == 3 else 1} kênh, {img.dtype}")
    ts = np.array([t for _, t, _ in files], dtype=np.int64)
    dt = np.diff(ts) / 1e6
    pos = dt[dt > 0]
    fps_est = 1.0 / np.median(pos)
    dup = int((dt == 0).sum())
    drops = int((pos > 1.5 * np.median(pos)).sum())
    md += [f"- fps ước lượng từ DeviceTimeStamp: **{fps_est:.2f}** (median Δt = {np.median(pos) * 1000:.2f} ms); "
           f"paper HA4M ghi 30 fps → dùng `fps: 30` trong config",
           f"- Δt = 0 (timestamp trùng): {dup} frame (thường ở đầu video); Δt > 1.5×median (rơi frame): {drops}",
           f"- Thời lượng theo frame_idx/30: {len(files) / 30:.1f} s; theo timestamp: {(ts[-1] - ts[0]) / 1e6:.1f} s", ""]

    # Skeleton
    sk = ha4m_io.frame_files(a.data_root, sub, tri, "Skeletons")
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"[{TAG}] đã ghi {out}")


if __name__ == "__main__":
    main()

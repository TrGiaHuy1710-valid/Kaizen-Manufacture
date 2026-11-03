"""Render video overlay (skeleton cổ tay + zone + banner tên bước) cho MỌI nguồn (ha4m:// | video:// | frames://).

    render_overlay_video(run_root, trial, cfg, source_uri, out, prefer="oracle") -> dict meta
Ra H.264 + faststart (phát được trong trình duyệt, qua imageio-ffmpeg). Nhãn bước: `prefer` nếu run có
(<trial>_oracle.csv = ground truth), không thì nguồn còn lại — banner trên video ghi rõ nguồn nhãn.
Ghi kèm <out>.json: {source, run_id, trial, n_frames, fps, width, rendered_at} để UI biết bản render dùng nguồn nào.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Callable

import cv2

from kaizen.skeleton.overlay import SOURCE_LABEL, load_overlay_data, render_frame
from kaizen.skeleton.sources import open_source


def render_meta_path(out: str | Path) -> Path:
    """<out>.json cạnh video."""
    out = Path(out)
    return out.with_suffix(".json")


def render_overlay_video(run_root: str | Path, trial: str, cfg: dict, source_uri: str, out: str | Path, *,
                         prefer: str = "oracle", stride: int = 2, width: int = 960, max_seconds: float | None = None,
                         progress: Callable[[float], None] | None = None,
                         log: Callable[[str], None] = print) -> dict:
    """Vẽ overlay từng frame (1/stride) của nguồn → mp4; trả meta (cũng ghi <out>.json)."""
    import imageio_ffmpeg
    run_root, out = Path(run_root), Path(out)
    od = load_overlay_data(run_root, trial, cfg, prefer)
    if od.steps is None:
        log(f"[render] cảnh báo: run chưa có steps của {trial} — chỉ vẽ zone + cổ tay")
    src = open_source(source_uri)
    fps = float(src.fps() or cfg.get("fps", 30))
    n_total = int(src.meta().get("n_frames") or 0)
    max_frames = int(max_seconds * fps / stride) if max_seconds else 0
    n_exp = max(1, min(n_total // stride if n_total else 1, max_frames or 10 ** 9))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    writer, k, t0, size = None, 0, time.time(), None
    for idx, img in src.frames(stride=stride, max_frames=max_frames):
        frame = render_frame(img, idx / fps, od, width)
        if writer is None:
            h, w = frame.shape[:2]
            size = (w, h)
            writer = imageio_ffmpeg.write_frames(
                str(tmp), size, fps=fps / stride, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                quality=None, bitrate=None, macro_block_size=2, ffmpeg_log_level="error",
                output_params=["-crf", "26", "-preset", "veryfast", "-movflags", "+faststart"])
            writer.send(None)
        writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
        k += 1
        if progress and k % 25 == 0:
            progress(min(k / n_exp, 1.0))
        if k % 200 == 0:
            log(f"[render] {k}/{n_exp} frame ({k / max(time.time() - t0, 1e-6):.1f} fps)")
    if writer is None:
        raise RuntimeError(f"không đọc được frame nào từ {source_uri}")
    writer.close()
    tmp.replace(out)
    meta = {"source": od.source or None, "source_label": SOURCE_LABEL.get(od.source, od.source or "không có nhãn"),
            "run_id": run_root.name, "trial": trial, "n_frames": k, "fps": round(fps / stride, 3),
            "width": size[0], "height": size[1], "max_seconds": max_seconds,
            "rendered_at": dt.datetime.now().isoformat(timespec="seconds")}
    render_meta_path(out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(1.0)
    log(f"[render] đã ghi {out} ({k} frame, nguồn nhãn {meta['source_label']}, {time.time() - t0:.0f}s)")
    return meta

"""Render video demo có overlay cho 1 trial của 1 run → <run>/report/demo_<trial>.mp4 (H.264, phát được trong trình duyệt).

Overlay: zone (màu theo bước dùng khay), vệt quỹ đạo 2 cổ tay ~1 s, banner tên bước + đồng hồ bước/chu kỳ,
trạng thái tay (hoạt động/chờ/mất dấu). Chỉ đọc artifact của run (features, steps, events) + ảnh gốc.

    python src/tools/render_demo.py --run latest --trial U001_V001 [--stride 2] [--width 960] [--force]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ha4m_io  # noqa: E402
import overlay  # noqa: E402
import run_io  # noqa: E402
from common import load_config, parse_stem  # noqa: E402

TAG = "demo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest")
    ap.add_argument("--trial", required=True, help="vd U001_V001")
    ap.add_argument("--stride", type=int, default=2, help="lấy 1/N frame (video vẫn đúng thời gian thực)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--start", type=float, default=0.0, help="giây bắt đầu")
    ap.add_argument("--end", type=float, default=0.0, help="giây kết thúc (0 = hết video)")
    ap.add_argument("--out", default=None, help="mặc định <run>/report/demo_<trial>.mp4")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import imageio_ffmpeg
    try:
        ctx = run_io.open_run(a.run)
    except FileNotFoundError as e:
        print(f"[{TAG}] {e}", file=sys.stderr)
        sys.exit(1)
    man = ctx.manifest
    cfg = load_config(ctx.station_path if ctx.station_path.exists() else "config/station.yaml")
    out = Path(a.out) if a.out else ctx.path("report", f"demo_{a.trial}.mp4")
    if out.exists() and not a.force:
        print(f"[{TAG}] đã có {out} (--force để render lại)")
        return
    sub, tri = parse_stem(a.trial)
    root = man.get("data_root") or ha4m_io.DEFAULT_ROOT
    try:
        files = ha4m_io.frame_files(root, sub, tri)
    except FileNotFoundError as e:
        print(f"[{TAG}] không đọc được ảnh gốc: {e}", file=sys.stderr)
        sys.exit(1)
    od = overlay.load_overlay_data(ctx.root, a.trial, cfg, man.get("source") or "")
    if od.feat is None:
        print(f"[{TAG}] cảnh báo: run chưa có features của {a.trial} — chỉ vẽ zone + banner", file=sys.stderr)
    fps = float(cfg["fps"])
    sel = [(i, p) for i, _, p in files if i / fps >= a.start and (not a.end or i / fps <= a.end)][::a.stride]
    if not sel:
        print(f"[{TAG}] không có frame trong khoảng thời gian yêu cầu", file=sys.stderr)
        sys.exit(1)

    first = overlay.render_frame(cv2.imread(str(sel[0][1])), sel[0][0] / fps, od, a.width)
    h, w = first.shape[:2]
    tmp = out.with_suffix(".part.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(tmp), (w, h), fps=fps / a.stride, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        quality=None, bitrate=None, macro_block_size=2, ffmpeg_log_level="error",
        output_params=["-crf", "26", "-preset", "veryfast", "-movflags", "+faststart"])
    writer.send(None)
    t0 = time.time()
    for k, (i, p) in enumerate(sel):
        img = first if k == 0 else overlay.render_frame(cv2.imread(str(p)), i / fps, od, a.width)
        writer.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
        if k % 100 == 0:
            print(f"[{TAG}] {k}/{len(sel)} frame ({k / max(time.time() - t0, 1e-6):.1f} fps)", flush=True)
    writer.close()
    tmp.replace(out)
    print(f"[{TAG}] đã ghi {out} ({len(sel)} frame, {w}x{h}, {fps / a.stride:g} fps, "
          f"{out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

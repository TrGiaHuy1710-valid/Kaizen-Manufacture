"""
Trích xuất keypoints từ video HOẶC thư mục chuỗi ảnh -> file .npz theo một schema chung,
để code ngày 2 không phụ thuộc model nào được dùng. (Wrapper mỏng — logic ở kaizen.skeleton.)

Schema output (.npz):
  kpts   : float32 (T, K, 3)  -> x_px, y_px, score   (NaN nếu không thấy), toạ độ theo ảnh GỐC
  t      : float32 (T,)       -> thời gian (giây) của từng frame đã xử lý
  frame  : int32 (T,)         -> frame index gốc (thêm mới, không phá schema cũ)
  names  : (K,)               -> tên keypoint
  meta   : json string        -> model, fps gốc, stride, kích thước frame, fps xử lý, roi

Ví dụ:
  python extract_keypoints.py --video data/raw/test.mp4 --model mediapipe
  python extract_keypoints.py --video data/raw/test.mp4 --model rtmw --stride 2 --preview
  # HA4M: thư mục chuỗi ảnh, t = frame_idx / fps; tên output <subject>_<trial>_<model>.npz
  python extract_keypoints.py --frames-dir data/ha4m/IDU001V001/Color/000124702712 --fps 30 \
      --roi 350,250,1500,1150 --preview
  # Tương đương qua package: python -m kaizen.skeleton extract "video://data/raw/test.mp4" --model mediapipe
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaizen.skeleton.extract import extract, parse_roi  # noqa: E402,F401
from kaizen.skeleton.pose import (COCO17, HAND, MP_NAMES, MP_POSE, RTMW_NAMES, MediaPipeBackend,  # noqa: E402,F401
                                  RTMWBackend, draw)
from kaizen.skeleton.sources import FramesSource, VideoSource, guess_stem  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--frames-dir", help="thư mục chuỗi ảnh (HA4M Color/<serial>)")
    ap.add_argument("--fps", type=float, default=None, help="bắt buộc với --frames-dir (t = frame_idx/fps)")
    ap.add_argument("--name", default="", help="stem output, mặc định <subject>_<trial> suy từ đường dẫn")
    ap.add_argument("--model", choices=["mediapipe", "mediapipe-heavy", "rtmw"], default="mediapipe")
    ap.add_argument("--rtmw-mode", choices=["lightweight", "balanced", "performance"], default="balanced")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (cho rtmw)")
    ap.add_argument("--roi", default="", help="x,y,w,h: chỉ chạy model trong vùng này (toạ độ output vẫn theo ảnh gốc)")
    ap.add_argument("--no-hand-crop", action="store_true",
                    help="MediaPipe: dò tay trên cả ảnh thay vì quanh cổ tay pose (mặc định crop khi dùng --frames-dir)")
    ap.add_argument("--stride", type=int, default=1, help="xử lý 1 trên N frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="data/keypoints")
    ap.add_argument("--preview", action="store_true", help="xuất video overlay để kiểm tra")
    ap.add_argument("--preview-scale", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="ghi đè .npz đã có")
    a = ap.parse_args()

    if a.frames_dir and not a.fps:
        ap.error("--fps là bắt buộc khi dùng --frames-dir")
    try:
        source = VideoSource(a.video) if a.video else FramesSource(a.frames_dir, a.fps)
    except FileNotFoundError as e:
        sys.exit(str(e))
    stem = a.name or (Path(a.video).stem if a.video else guess_stem(a.frames_dir))
    npz_path = Path(a.out) / f"{stem}_{a.model}.npz"
    if npz_path.exists() and not a.force:
        print(f"[extract] bỏ qua {npz_path} (đã có, dùng --force)")
        return
    extract(source, a.model, a.stride, npz_path, max_frames=a.max_frames, roi=parse_roi(a.roi),
            hand_crop=bool(a.frames_dir) and not a.no_hand_crop, preview=a.preview, preview_scale=a.preview_scale,
            rtmw_mode=a.rtmw_mode, device=a.device, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()

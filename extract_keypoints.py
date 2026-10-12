"""
Trích xuất keypoints từ video HOẶC thư mục chuỗi ảnh -> file .npz theo một schema chung,
để code ngày 2 không phụ thuộc model nào được dùng.

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
"""
import argparse, json, re, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ---------------- Tên keypoint ----------------
MP_POSE = ["nose", "l_eye_in", "l_eye", "l_eye_out", "r_eye_in", "r_eye", "r_eye_out",
           "l_ear", "r_ear", "mouth_l", "mouth_r", "l_shoulder", "r_shoulder",
           "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_pinky", "r_pinky",
           "l_index", "r_index", "l_thumb", "r_thumb", "l_hip", "r_hip",
           "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel",
           "l_foot", "r_foot"]
HAND = ["wrist"] + [f"{f}_{j}" for f in ["thumb", "index", "middle", "ring", "pinky"]
                    for j in range(1, 5)]
MP_NAMES = MP_POSE + [f"lh_{n}" for n in HAND] + [f"rh_{n}" for n in HAND]  # 33 + 42 = 75

COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
          "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
          "l_knee", "r_knee", "l_ankle", "r_ankle"]
RTMW_NAMES = (COCO17 + [f"foot_{i}" for i in range(6)] + [f"face_{i}" for i in range(68)]
              + [f"lh_{n}" for n in HAND] + [f"rh_{n}" for n in HAND])  # 133


# ---------------- Backend MediaPipe ----------------
class MediaPipeBackend:
    """Pose (VIDEO mode) + Hand.

    hand_crop=True: dò bàn tay trên vùng vuông quanh từng cổ tay pose (IMAGE mode) thay vì cả ảnh —
    cần cho ảnh lớn (HA4M 2048x1536) nơi bàn tay chỉ ~60px, và tránh bắt nhầm tay người khác.
    """

    def __init__(self, model_dir="models", heavy=False, hand_crop=False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.hand_crop = hand_crop
        pose_file = "pose_landmarker_heavy.task" if heavy else "pose_landmarker_full.task"
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(model_dir) / pose_file)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        mode = vision.RunningMode.IMAGE if hand_crop else vision.RunningMode.VIDEO
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(model_dir) / "hand_landmarker.task")),
            running_mode=mode, num_hands=2))
        self.names = MP_NAMES

    def _img(self, bgr):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    def __call__(self, frame_bgr, ts_ms):
        h, w = frame_bgr.shape[:2]
        out = np.full((len(self.names), 3), np.nan, np.float32)

        pr = self.pose.detect_for_video(self._img(frame_bgr), ts_ms)
        if pr.pose_landmarks:
            for i, lm in enumerate(pr.pose_landmarks[0]):
                out[i] = (lm.x * w, lm.y * h, lm.visibility)

        if self.hand_crop:
            self._hands_cropped(frame_bgr, out)
            return out

        hr = self.hand.detect_for_video(self._img(frame_bgr), ts_ms)
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            # Ghép tay vào cổ tay pose gần nhất -> tránh lỗi nhãn Left/Right
            # của MediaPipe (nó giả định ảnh bị lật kiểu selfie).
            wx, wy = lms[0].x * w, lms[0].y * h
            side = self._closest_side(out, wx, wy, hd[0].category_name)
            base = 33 if side == "l" else 54
            score = hd[0].score
            for j, lm in enumerate(lms):
                out[base + j] = (lm.x * w, lm.y * h, score)
        return out

    def _hands_cropped(self, frame, out):
        h, w = frame.shape[:2]
        for side, wi, ei, base in (("l", 15, 13, 33), ("r", 16, 14, 54)):
            wx, wy, ws = out[wi]
            if np.isnan(wx) or ws < 0.2:
                continue
            ex, ey = out[ei, :2]
            fore = np.hypot(wx - ex, wy - ey) if not np.isnan(ex) else 100.0
            half = int(np.clip(1.3 * fore, 96, 300))
            # dời tâm crop về phía bàn tay (hướng elbow->wrist)
            cx, cy = wx, wy
            if not np.isnan(ex) and fore > 1:
                cx, cy = wx + 0.5 * half * (wx - ex) / fore, wy + 0.5 * half * (wy - ey) / fore
            x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
            x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            crop = frame[y0:y1, x0:x1]
            scale = 256.0 / max(crop.shape[:2])
            if scale > 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
            hr = self.hand.detect(self._img(crop))
            best, bd = None, 1e9
            for lms, hd in zip(hr.hand_landmarks, hr.handedness):
                hx = x0 + lms[0].x * crop.shape[1] / scale
                hy = y0 + lms[0].y * crop.shape[0] / scale
                d = np.hypot(hx - wx, hy - wy)
                if d < bd:
                    best, bd = (lms, hd[0].score), d
            if best is None or bd > 0.6 * half:   # tay khác (vd tay còn lại) -> bỏ
                continue
            lms, score = best
            for j, lm in enumerate(lms):
                out[base + j] = (x0 + lm.x * crop.shape[1] / scale, y0 + lm.y * crop.shape[0] / scale, score)

    @staticmethod
    def _closest_side(out, x, y, fallback):
        lw, rw = out[15, :2], out[16, :2]
        if np.isnan(lw).any() or np.isnan(rw).any():
            return "r" if fallback == "Left" else "l"  # đảo vì camera không lật
        return "l" if np.hypot(*(lw - (x, y))) < np.hypot(*(rw - (x, y))) else "r"


# ---------------- Backend RTMW (rtmlib) ----------------
class RTMWBackend:
    def __init__(self, mode="balanced", device="cpu"):
        from rtmlib import Wholebody
        self.model = Wholebody(mode=mode, backend="onnxruntime", device=device)
        self.names = RTMW_NAMES

    def __call__(self, frame_bgr, ts_ms):
        kps, scores = self.model(frame_bgr)  # (N,133,2), (N,133)
        out = np.full((len(self.names), 3), np.nan, np.float32)
        if kps is None or len(kps) == 0:
            return out
        # Một công nhân / trạm: chọn người có tổng score body lớn nhất
        idx = int(np.argmax(scores[:, :17].sum(1)))
        out[:, :2], out[:, 2] = kps[idx], scores[idx]
        return out


def draw(frame, kp, thr=0.3):
    for x, y, s in kp:
        if not np.isnan(x) and s >= thr:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    return frame


def parse_roi(s):
    if not s:
        return None
    x, y, w, h = (int(v) for v in s.split(","))
    return x, y, w, h


def video_source(path, stride, max_frames):
    """yield (frame_idx, bgr) từ .mp4; trả fps, W, H qua thuộc tính hàm."""
    cap = cv2.VideoCapture(path)
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


def guess_stem(frames_dir):
    """.../IDU001V001/Color/<serial> -> U001_V001; nếu không khớp thì dùng tên thư mục."""
    for part in reversed(Path(frames_dir).resolve().parts + Path(frames_dir).parts):
        m = re.match(r"^ID(U\d+)(V\d+)$", part)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
    return Path(frames_dir).name


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

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    if a.video:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        stem = f"{a.name or Path(a.video).stem}_{a.model}"
        frames = video_source(a.video, a.stride, a.max_frames)
    else:
        import ha4m_io
        fps = float(a.fps)
        files = ha4m_io.frames_dir_files(a.frames_dir)
        if not files:
            sys.exit(f"Không có ảnh trong {a.frames_dir}")
        H, W = cv2.imread(str(files[0][1])).shape[:2]
        stem = f"{a.name or guess_stem(a.frames_dir)}_{a.model}"
        frames = ha4m_io.frames_dir_iter(a.frames_dir, a.stride, a.max_frames)

    npz_path = out_dir / f"{stem}.npz"
    if npz_path.exists() and not a.force:
        print(f"[extract] bỏ qua {npz_path} (đã có, dùng --force)")
        return

    if a.model.startswith("mediapipe"):
        be = MediaPipeBackend(heavy=a.model.endswith("heavy"),
                              hand_crop=bool(a.frames_dir) and not a.no_hand_crop)
    else:
        be = RTMWBackend(a.rtmw_mode, a.device)

    roi = parse_roi(a.roi)
    writer = None
    if a.preview:
        pw, ph = int(W * a.preview_scale), int(H * a.preview_scale)
        writer = cv2.VideoWriter(str(out_dir / f"{stem}_preview.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (pw, ph))

    kpts, ts, fidx, t0 = [], [], [], time.time()
    for i, frame in frames:
        t_ms = int(round(i * 1000 / fps))
        if roi:
            x, y, w, h = roi
            kp = be(np.ascontiguousarray(frame[y:y + h, x:x + w]), t_ms)
            kp[:, 0] += x; kp[:, 1] += y
        else:
            kp = be(frame, t_ms)
        kpts.append(kp); ts.append(i / fps); fidx.append(i)
        if writer:
            vis = draw(frame.copy(), kp)
            if roi:
                cv2.rectangle(vis, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 128, 0), 2)
            cv2.putText(vis, f"f{i} t={i / fps:.2f}s", (20, 50), 0, 1.5, (0, 255, 255), 3)
            writer.write(cv2.resize(vis, (pw, ph)))
        if len(kpts) % 100 == 0:
            print(f"{len(kpts)} frames, {len(kpts) / (time.time() - t0):.1f} fps", flush=True)
    if writer:
        writer.release()

    proc_fps = len(kpts) / max(time.time() - t0, 1e-6)
    kpts = np.stack(kpts) if kpts else np.empty((0, len(be.names), 3), np.float32)
    detect_rate = float(np.mean(~np.isnan(kpts[:, :, 0]).all(1))) if len(kpts) else 0.0
    names = list(be.names)
    wr = {}
    for k in ("l_wrist", "r_wrist", "lh_wrist", "rh_wrist", "lh_middle_1", "rh_middle_1"):
        if k in names:
            s = kpts[:, names.index(k), 2]
            wr[k] = round(float(np.mean(np.nan_to_num(s, nan=0) >= 0.3)), 3) if len(s) else 0.0
    meta = dict(model=a.model, video=a.video, frames_dir=a.frames_dir, src_fps=fps, stride=a.stride,
                width=W, height=H, roi=roi, hand_crop=getattr(be, "hand_crop", None),
                proc_fps=round(proc_fps, 2), detect_rate=round(detect_rate, 3), valid_rate_score03=wr)
    np.savez_compressed(npz_path, kpts=kpts, t=np.array(ts, np.float32), frame=np.array(fidx, np.int32),
                        names=np.array(names), meta=json.dumps(meta))
    print(json.dumps(meta, indent=2))
    print(f"[extract] đã ghi {npz_path}")


if __name__ == "__main__":
    main()

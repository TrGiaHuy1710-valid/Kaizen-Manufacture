"""kaizen.train — chỉ số, đặc trưng, smoke-train MS-TCN, chuẩn bị dữ liệu pose. Dữ liệu tổng hợp, không cần dataset/GPU.

Smoke-train chỉ chạy vài epoch trên chuỗi giả để kiểm vòng train/lưu/nạp/suy luận — KHÔNG phải train model thật.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from kaizen.skeleton import ha4m_io
from kaizen.skeleton.pose import COCO17, MP_NAMES
from kaizen.train import dataset as ds
from kaizen.train.features import JOINTS, feature_dim, frame_features
from kaizen.train.metrics import edit_score, f1_at, segment_report, segments

FPS = 30
W, H = 1000, 800
# (khoá lớp, số frame, x cổ tay phải /W) — mỗi bước tay ở một vị trí khác
PLAN = [("0", 40, 0.5), ("1.0", 120, 0.2), ("2.0", 90, 0.8), ("2.1", 90, 0.65), ("3.0", 120, 0.35), ("0", 40, 0.5)]


# ---------------------------------------------------------------- chỉ số
def test_segments_and_scores():
    gt = [0, 0, 1, 1, 1, 2, 2, 2, 0]
    assert segments(gt, bg=0) == [(1, 2, 5), (2, 5, 8)]
    assert edit_score(gt, gt, bg=0) == 100.0
    assert edit_score([0, 1, 1, 2, 2, 1, 1, 0, 0], gt, bg=0) == pytest.approx(100 * (1 - 1 / 3))
    assert f1_at(gt, gt, 0.5, bg=0) == 100.0
    shifted = [0, 0, 0, 1, 1, 1, 2, 2, 2]          # cả 2 đoạn lệch 1 frame → IoU = 2/4 = 0.5
    assert f1_at(shifted, gt, 0.5, bg=0) == 100.0 and f1_at(shifted, gt, 0.6, bg=0) == 0.0
    r = segment_report(shifted, gt, bg=0)
    assert set(r) == {"acc", "edit", "f1@10", "f1@25", "f1@50"}


# ---------------------------------------------------------------- dữ liệu giả
def _npz(path: Path, names: list[str], rng, plan=PLAN) -> list[str]:
    T = sum(n for _, n, _ in plan)
    k = np.full((T, len(names), 3), np.nan, np.float32)
    keys, xs = [], []
    for key, n, x in plan:
        keys += [key] * n
        xs += [x] * n
    xs = np.convolve(np.array(xs), np.ones(9) / 9, mode="same")
    t = np.arange(T) / FPS

    def put(name, x, y):
        if name in names:
            i = names.index(name)
            k[:, i, 0], k[:, i, 1], k[:, i, 2] = x + rng.normal(0, 3, T), y + rng.normal(0, 3, T), 0.9
    put("l_shoulder", 420, 300); put("r_shoulder", 580, 300); put("nose", 500, 220)
    put("l_hip", 440, 520); put("r_hip", 560, 520); put("l_elbow", 400, 400); put("l_wrist", 420, 470)
    put("r_wrist", xs * W, 520 + 40 * np.sin(2 * np.pi * t)); put("r_elbow", (xs * W + 580) / 2, 420)
    np.savez_compressed(path, kpts=k, t=t.astype(np.float32), frame=np.arange(T, dtype=np.int32),
                        names=np.array(names), meta=json.dumps({"model": "mediapipe", "src_fps": FPS,
                                                                "width": W, "height": H}))
    return keys


def _ha4m(tmp: Path, n_subjects: int = 5, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    root, kp = tmp / "ha4m", tmp / "keypoints"
    kp.mkdir(parents=True)
    for i in range(n_subjects):
        sub = f"U9{i:02d}"
        plan = [(key, max(20, int(n * rng.uniform(0.8, 1.2))), x) for key, n, x in PLAN]
        keys = _npz(kp / f"{sub}_V001_mediapipe.npz", MP_NAMES, rng, plan)
        vd = root / f"ID{sub}V001"
        (vd / "Color" / "000").mkdir(parents=True)
        with open(vd / "Labels.txt", "w") as f:
            for fi, key in enumerate(keys):
                a, r = ds.parse_key(key)
                f.write(f"{fi} {a} {r}\r\n")
    return root, kp


def _station(tmp: Path) -> Path:
    cfg = {"fps": FPS, "steps": [{"id": 1, "name": "Step 1", "reps": 1}, {"id": 2, "name": "Step 2", "reps": 2},
                                 {"id": 3, "name": "Step 3", "reps": 1}]}
    p = tmp / "station.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# ---------------------------------------------------------------- đặc trưng + dataset
def test_features_fixed_dim_for_any_pose_model(tmp_path):
    rng = np.random.default_rng(0)
    _npz(tmp_path / "mp.npz", MP_NAMES, rng)
    _npz(tmp_path / "coco.npz", COCO17, rng)
    from kaizen.train.features import load_npz
    f_mp, f_coco = frame_features(load_npz(tmp_path / "mp.npz")), frame_features(load_npz(tmp_path / "coco.npz"))
    assert f_mp.shape == f_coco.shape == (sum(n for _, n, _ in PLAN), feature_dim())
    assert np.isfinite(f_mp).all()
    mask = f_coco[:, 6 * len(JOINTS):]
    assert mask[:, JOINTS.index("lh_middle_1")].max() == 0 and mask[:, JOINTS.index("r_wrist")].min() == 1


def test_scan_split_and_labels(tmp_path):
    root, kp = _ha4m(tmp_path, 4)
    refs = ds.scan(str(root), kp)
    assert len(refs) == 4 and all(r.npz and r.has_labels for r in refs)
    tr = ds.load_trial(str(root), refs[0])
    assert len(tr.keys) == len(tr.feats) and tr.keys[0] == "0" and "2.1" in tr.keys
    a, b = ds.split_subjects([r.subject for r in refs], val_frac=0.25)
    assert len(b) == 1 and not set(a) & set(b)
    assert [len(f) for f in ds.group_folds([r.subject for r in refs], 0)] == [1, 1, 1, 1]


# ---------------------------------------------------------------- MS-TCN (smoke)
torch = pytest.importorskip("torch")


def test_smoke_train_save_load_predict(tmp_path, monkeypatch):
    from kaizen.train import segmenter as sg
    root, kp = _ha4m(tmp_path, 5)
    station = _station(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = sg.main(["train", "--data-root", str(root), "--keypoints-dir", str(kp), "--config", str(station),
                  "--name", "smoke", "--val-subjects", "U904", "--epochs", "25", "--stages", "2", "--layers", "4",
                  "--channels", "16", "--dropout", "0.1", "--sample-rate", "3", "--eval-every", "5",
                  "--patience", "100", "--lr", "0.005", "--device", "cpu", "--threads", "1"])
    assert rc == 0
    mdir = Path("models/segmenter/smoke")
    cfg = json.loads((mdir / "config.json").read_text())
    assert (mdir / "model.pt").exists() and (mdir / "train_log.csv").exists()
    assert Path("models/segmenter/LATEST").read_text().strip() == "smoke"
    assert cfg["classes"][0] == "0" and "2.1" in cfg["classes"] and cfg["val_subjects"] == ["U904"]
    assert cfg["val_step_eval"]["summary"]["acc"]["mean"] > 50     # chuỗi giả dễ: phải học được rõ ràng
    st = yaml.safe_load(station.read_text())
    df, info, _ = sg.predict_steps("latest", kp / "U904_V001_mediapipe.npz", st)
    assert list(df.columns) == ["step_id", "repetition", "name", "start_s", "end_s", "dur_s", "source", "missing"]
    assert (df["step_id"] > 0).all() and set(info) == {"not_found", "no_zone", "matched_order"}


def test_frames_to_steps_smoothing():
    from kaizen.train.segmenter import frames_to_steps
    keys = ["0"] * 10 + ["1.0"] * 60 + ["2.0"] * 3 + ["1.0"] * 30 + ["2.0"] * 60 + ["0"] * 10
    t = np.arange(len(keys)) / FPS
    st = {"fps": FPS, "steps": [{"id": 1, "name": "A"}, {"id": 2, "name": "B", "reps": 2}]}
    df, info = frames_to_steps(keys, t, st, min_seg_s=0.5)
    assert df[["step_id", "repetition"]].values.tolist() == [[1, 0], [2, 0]]   # mảnh 3 frame bị nhập
    assert info["not_found"] == ["2.1"]
    assert df["start_s"].iloc[0] == pytest.approx(10 / FPS, abs=1e-3)


# ---------------------------------------------------------------- pose: dữ liệu COCO từ Kinect + eval
def _skeleton_file(path: Path, pts: dict[str, tuple[float, float]], conf: float = 0.666667):
    rows = ["BodyID\tJointID\tConfidance\tX\tY\tZ\tQw\tQx\tQy\tQz\tX2DColor\tY2DColor\tX2DDepth\tY2DDepth"]
    for j, name in enumerate(ha4m_io.KINECT_JOINTS):
        x, y = pts.get(name, (0.0, 0.0))
        c = conf if name in pts else 0.0
        rows.append(f"1\t{j}\t{c}\t0\t0\t1000\t1\t0\t0\t0\t{x}\t{y}\t0\t0")
    path.write_text("\n".join(rows) + "\n")


def test_pose_prepare_and_eval(tmp_path):
    from kaizen.train.pose import eval_npz
    from kaizen.train.pose_data import prepare
    root = tmp_path / "ha4m"
    base = {n: (150 + 10 * (i % 5), 60 + 12 * i) for i, n in enumerate(COCO17)}
    for sub in ("U801", "U802"):
        vd = root / f"ID{sub}V001"
        (vd / "Color" / "s").mkdir(parents=True)
        (vd / "Skeletons" / "s").mkdir(parents=True)
        kpts = np.zeros((6, len(COCO17), 3), np.float32)
        for fi in range(6):
            cv2.imwrite(str(vd / "Color" / "s" / f"FrameID{fi:06d}_DeviceTimeStamp{1000 + fi}us.png"),
                        np.full((320, 320, 3), 80, np.uint8))
            _skeleton_file(vd / "Skeletons" / "s" / f"FrameID{fi:06d}_DeviceTimeStamp{1000 + fi}us.txt", base)
            kpts[fi, :, :2] = [base[n] for n in COCO17]
            kpts[fi, :, 2] = 0.9
        kpts[:, COCO17.index("r_wrist"), 0] += 40                      # 1 khớp lệch 40 px
        np.savez_compressed(tmp_path / f"{sub}_V001_rtmpose-custom.npz", kpts=kpts,
                            t=(np.arange(6) / FPS).astype(np.float32), frame=np.arange(6, dtype=np.int32),
                            names=np.array(COCO17), meta=json.dumps({"model": "rtmpose-custom", "width": 320,
                                                                     "height": 320, "src_fps": FPS}))
    out = tmp_path / "pose_ft"
    stats = prepare(str(root), out, stride=2, val_subjects=["U802"], log=lambda *_: None)
    assert stats["frames"] == {"train": 3, "val": 3}
    coco = json.loads((out / "annotations" / "person_keypoints_train.json").read_text())
    ann, img = coco["annotations"][0], coco["images"][0]
    assert len(ann["keypoints"]) == 51 and ann["num_keypoints"] == 17
    assert (out / "images" / img["file_name"]).exists()
    xs = ann["keypoints"][0::3]
    assert min(xs) >= 0 and max(xs) <= img["width"]                   # toạ độ theo ảnh crop
    r = eval_npz(str(root), tmp_path / "U801_V001_rtmpose-custom.npz", stride=1)
    assert r["per_joint"]["l_shoulder"]["median_px"] == 0.0 and r["per_joint"]["r_wrist"]["median_px"] == 40.0
    assert r["pck@0.1"] < 100 and r["per_joint"]["nose"]["pck@0.05"] == 100.0


def test_04_steps_cli_with_mstcn(tmp_path, monkeypatch):
    """04 --segmenter mstcn ghi đúng _pred.csv / _pred_info.json / _eval.json như đường rule."""
    import subprocess
    import sys as _sys
    from kaizen.train import segmenter as sg
    root, kp = _ha4m(tmp_path, 4)
    station = _station(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert sg.main(["train", "--data-root", str(root), "--keypoints-dir", str(kp), "--config", str(station),
                    "--name", "m", "--val-subjects", "U903", "--epochs", "10", "--stages", "2", "--layers", "3",
                    "--channels", "16", "--sample-rate", "3", "--device", "cpu", "--threads", "1"]) == 0
    ev = tmp_path / "outputs" / "events"
    ev.mkdir(parents=True)
    (ev / "U903_V001.csv").write_text("start_s,end_s,hand,zone,motion,dur_s,event_type\n")
    repo = Path(__file__).resolve().parents[1]
    base = [_sys.executable, str(repo / "src" / "04_steps.py"), "--data-root", str(root), "--config", str(station),
            "--force"]
    subprocess.run(base + ["--oracle"], check=True, capture_output=True, text=True)
    r = subprocess.run(base + ["--segmenter", "mstcn", "--segmenter-model", "models/segmenter/m",
                               "--keypoints-dir", str(kp)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    steps = tmp_path / "outputs" / "steps"
    assert (steps / "U903_V001_pred.csv").exists() and (steps / "U903_V001_pred_info.json").exists()
    e = json.loads((steps / "U903_V001_eval.json").read_text())
    assert e["n_oracle"] == 4 and "f1_iou05" in e          # 1.0, 2.0, 2.1, 3.0

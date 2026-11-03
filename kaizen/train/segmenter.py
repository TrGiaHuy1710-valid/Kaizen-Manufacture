"""Model phân đoạn bước SOP (MS-TCN) — train / cross-validation / eval / predict + suy luận cho stage 04.

Học từ chuỗi keypoint (.npz, data/keypoints/) + nhãn theo frame của HA4M (Labels.txt). KHÔNG tự chạy ngầm:
mọi lần train đều do người dùng gõ lệnh. Model lưu ở models/segmenter/<name>/ (model.pt + config.json + log),
models/segmenter/LATEST trỏ model train gần nhất.

    python -m kaizen.train.segmenter status                       # bao nhiêu trial đủ keypoints + nhãn
    python -m kaizen.train.segmenter train --name mstcn_v1 --val-subjects U040,U041
    python -m kaizen.train.segmenter cv --folds 5                 # cross-validation theo subject (0 = LOSO)
    python -m kaizen.train.segmenter eval --model models/segmenter/mstcn_v1 --subjects U040,U041
    python -m kaizen.train.segmenter predict --model models/segmenter/mstcn_v1 --npz data/keypoints/U001_V001_mediapipe.npz
Dùng trong pipeline:  python src/04_steps.py --segmenter mstcn [--segmenter-model models/segmenter/mstcn_v1]
                      bash run_all.sh --segmenter mstcn ...
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kaizen.core.config import DEFAULT_CONFIG, DEFAULT_DATA_ROOT, load_config
from kaizen.train import dataset as ds
from kaizen.train.features import FEATURE_VERSION, feature_dim, frame_features, load_npz
from kaizen.train.metrics import mean_reports, segment_report

TAG = "seg"
MODELS_DIR = Path("models/segmenter")
LATEST = MODELS_DIR / "LATEST"
DEFAULTS = dict(stages=4, layers=10, channels=64, dropout=0.5, lambda_smooth=0.15, lr=5e-4, weight_decay=1e-4,
                epochs=100, batch_size=4, sample_rate=2, patience=20, eval_every=2, seed=0, min_seg_s=0.5,
                threads=0)


def log(msg: str) -> None:
    print(f"[{TAG}] {msg}", flush=True)


# ---------------------------------------------------------------- torch helpers (import trễ)
def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise SystemExit("[seg] cần torch: pip install torch (CPU) hoặc bản CUDA — xem docs/training.md") from e
    return torch


def pick_device(name: str = "auto") -> str:
    torch = _torch()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def set_threads(n: int = 0) -> int:
    """Số luồng CPU cho torch. 0 = tự chọn theo số lõi đang rảnh (máy chia sẻ, load cao → dùng ít luồng:
    tranh chấp luồng có thể làm chậm hàng chục lần)."""
    torch = _torch()
    if n <= 0:
        cpus = os.cpu_count() or 1
        try:
            load = os.getloadavg()[0]
        except (AttributeError, OSError):
            load = 0.0
        n = max(1, min(cpus, int(cpus - load + 0.5)))
    torch.set_num_threads(n)
    return n


def set_seed(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch(items: list[ds.Encoded], device: str):
    torch = _torch()
    B, T, D = len(items), max(len(e.x) for e in items), items[0].x.shape[1]
    x = torch.zeros(B, D, T)
    y = torch.full((B, T), ds.IGNORE, dtype=torch.long)
    m = torch.zeros(B, 1, T)
    for i, e in enumerate(items):
        n = len(e.x)
        x[i, :, :n] = torch.from_numpy(e.x.T)
        y[i, :n] = torch.from_numpy(e.y)
        m[i, :, :n] = 1
    return x.to(device), y.to(device), m.to(device)


def build_model(hp: dict, dim_in: int, n_classes: int):
    from kaizen.train.mstcn import MSTCN
    return MSTCN(dim_in, n_classes, hp["stages"], hp["layers"], hp["channels"], hp["dropout"])


def _predict_encoded(model, enc: ds.Encoded, device: str) -> np.ndarray:
    """Lớp dự đoán theo frame (ở độ phân giải đã lấy mẫu)."""
    torch = _torch()
    model.eval()
    with torch.no_grad():
        x, _, m = _batch([enc], device)
        return model(x, m)[-1, 0].argmax(0).cpu().numpy()[: len(enc.x)]


def _eval_frames(model, encs: list[ds.Encoded], device: str) -> dict:
    reps = [segment_report(_predict_encoded(model, e, device), e.y, bg=0, ignore=ds.IGNORE) for e in encs]
    return mean_reports(reps)


# ---------------------------------------------------------------- dữ liệu
def load_trials(data_root: str, kp_dir: str, pose_model: str, subjects: set[str] | None = None,
                fps: float = 30.0, max_trials: int = 0) -> tuple[list[ds.Trial], list[ds.TrialRef]]:
    refs = [r for r in ds.scan(data_root, kp_dir, pose_model) if r.npz and r.has_labels
            and (not subjects or r.subject in subjects)]
    if max_trials:
        refs = refs[:max_trials]
    t0 = time.time()
    trials = [ds.load_trial(data_root, r, fps) for r in refs]
    log(f"nạp {len(trials)} trial ({sum(len(t.feats) for t in trials)} frame) trong {time.time() - t0:.1f}s")
    return trials, refs


def train_model(train: list[ds.Trial], val: list[ds.Trial], hp: dict, device: str = "auto",
                out_dir: Path | None = None, extra_config: dict | None = None) -> tuple[object, dict, list[dict]]:
    """Train MS-TCN; early stopping theo F1@50 trên val (không có val → theo loss train).
    Trả (model ở epoch tốt nhất, config, lịch sử từng epoch)."""
    torch = _torch()
    device = pick_device(device)
    set_seed(hp["seed"])
    n_threads = set_threads(hp.get("threads", 0)) if device == "cpu" else None
    classes = ds.sort_keys(k for t in train for k in t.keys if k)
    if "0" not in classes:
        classes = ["0"] + classes
    norm = ds.Normalizer.fit([t.feats for t in train])
    sr = hp["sample_rate"]
    tr = [ds.encode(t, classes, norm, sr) for t in train]
    va = [ds.encode(t, classes, norm, sr) for t in val]
    model = build_model(hp, feature_dim(), len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    from kaizen.train.mstcn import mstcn_loss
    n_params = sum(p.numel() for p in model.parameters())
    log(f"device={device}{f' ({n_threads} luồng)' if n_threads else ''} · {len(tr)} train / {len(va)} val chuỗi · "
        f"{len(classes)} lớp · {n_params / 1e3:.0f}k tham số")
    best, best_state, best_ep, wait, hist = -math.inf, None, 0, 0, []
    rng = random.Random(hp["seed"])
    t_start = time.time()
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        order = list(range(len(tr)))
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for i in range(0, len(order), hp["batch_size"]):
            x, y, m = _batch([tr[j] for j in order[i:i + hp["batch_size"]]], device)
            loss = mstcn_loss(model(x, m), y, m, hp["lambda_smooth"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        row = {"epoch": ep, "loss": round(tot / max(nb, 1), 4)}
        if ep % hp["eval_every"] == 0 or ep == hp["epochs"]:
            if va:
                row.update({f"val_{k}": v for k, v in _eval_frames(model, va, device).items()})
                score = row["val_f1@50"]
            else:
                score = -row["loss"]
            if score > best:
                best, best_ep, wait = score, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += hp["eval_every"]
            log(" · ".join(f"{k}={v}" for k, v in row.items()) + f" · {time.time() - t_start:.0f}s")
            if wait >= hp["patience"]:
                log(f"early stop ở epoch {ep} (tốt nhất epoch {best_ep})")
                hist.append(row)
                break
        hist.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    cfg = {"version": 1, "arch": "mstcn", "feature_version": FEATURE_VERSION, "feature_dim": feature_dim(),
           "classes": classes, "hparams": hp, "normalizer": norm.to_json(), "best_epoch": best_ep,
           "epochs_run": hist[-1]["epoch"] if hist else 0, "train_seconds": round(time.time() - t_start, 1),
           "device": device, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
           "train_trials": [t.stem for t in train], "val_trials": [t.stem for t in val],
           "train_subjects": sorted({t.subject for t in train}), "val_subjects": sorted({t.subject for t in val}),
           "pose_model": sorted({t.pose_model for t in train}), **(extra_config or {})}
    if out_dir is not None:
        save_model(model, cfg, out_dir, hist)
    return model, cfg, hist


def save_model(model, cfg: dict, out_dir: Path, hist: list[dict] | None = None) -> None:
    torch = _torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    if hist:
        keys = list(dict.fromkeys(k for r in hist for k in r))
        with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(hist)
    log(f"đã lưu model → {out_dir}")


def load_model(model_dir: str | Path, device: str = "cpu"):
    """(model, config) từ models/segmenter/<name>/ (hoặc 'latest')."""
    torch = _torch()
    model_dir = resolve_model_dir(model_dir)
    ck = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    if cfg.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"model dùng đặc trưng {cfg.get('feature_version')} ≠ {FEATURE_VERSION} hiện tại")
    model = build_model(cfg["hparams"], cfg["feature_dim"], len(cfg["classes"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


def resolve_model_dir(model_dir: str | Path | None) -> Path:
    if model_dir in (None, "", "latest"):
        if not LATEST.exists():
            raise FileNotFoundError("chưa có model nào (models/segmenter/LATEST) — train trước: "
                                    "python -m kaizen.train.segmenter train")
        return MODELS_DIR / LATEST.read_text(encoding="utf-8").strip()
    return Path(model_dir)


# ---------------------------------------------------------------- suy luận → bảng bước (schema stage 04)
def _smooth(keys: list[str], t: np.ndarray, fps: float, min_seg_s: float, bg: str = "0") -> list[tuple[str, int, int]]:
    """Đoạn [lớp, start, end) liên tiếp; MẢNH BƯỚC < min_seg_s nhập vào hàng xóm dài hơn (lặp), rồi gộp đoạn kề
    cùng lớp. Đoạn nền (bg) không bị nhập vào bước — tránh kéo dài bước sang phần đầu/cuối không thao tác."""
    segs: list[list] = []
    for i, k in enumerate(keys):
        if segs and segs[-1][0] == k:
            segs[-1][2] = i + 1
        else:
            segs.append([k, i, i + 1])
    min_n = max(1, int(round(min_seg_s * fps)))
    while len(segs) > 1:
        lens = [e - b if k != bg else np.inf for k, b, e in segs]
        i = int(np.argmin(lens))
        if lens[i] >= min_n:
            break
        lens = [e - b for _, b, e in segs]
        left = lens[i - 1] if i > 0 else -1
        right = lens[i + 1] if i + 1 < len(segs) else -1
        j = i - 1 if left >= right else i + 1
        if j < i:
            segs[j][2] = segs[i][2]
        else:
            segs[j][1] = segs[i][1]
        del segs[i]
        merged = [segs[0]]
        for sg in segs[1:]:
            if sg[0] == merged[-1][0]:
                merged[-1][2] = sg[2]
            else:
                merged.append(sg)
        segs = merged
    return [tuple(sg) for sg in segs]


def predict_frames(model, cfg: dict, kp: dict, fps: float, device: str = "cpu") -> list[str]:
    """Lớp ('0', '2.1', …) cho TỪNG frame của .npz (dự đoán ở 1/sample_rate rồi lặp lại)."""
    norm = ds.Normalizer.from_json(cfg["normalizer"])
    sr = int(cfg["hparams"]["sample_rate"])
    feats = frame_features(kp, fps)
    enc = ds.Encoded("x", norm(feats[::sr]), np.zeros(len(feats[::sr]), np.int64))
    pred = _predict_encoded(model, enc, device)
    full = np.repeat(pred, sr)[: len(feats)]
    if len(full) < len(feats):
        full = np.concatenate([full, np.full(len(feats) - len(full), full[-1] if len(full) else 0)])
    return [cfg["classes"][i] for i in full]


def frames_to_steps(keys: list[str], t: np.ndarray, station: dict, min_seg_s: float = 0.5):
    """Nhãn theo frame → (DataFrame cột như _pred.csv của 04, info dict not_found/no_zone/matched_order)."""
    from kaizen.analyzer.steps import COLS, step_names
    fps = float(station["fps"])
    names = step_names(station)
    rows = []
    for k, s, e in _smooth(keys, t, fps, min_seg_s):
        if k == "0":
            continue
        a, rep = ds.parse_key(k)
        rows.append(dict(step_id=a, repetition=rep, name=names.get(a, str(a)), start_s=round(float(t[s]), 3),
                         end_s=round(float(t[e - 1]) + 1.0 / fps, 3)))
    df = pd.DataFrame(rows, columns=COLS[:-2])
    df["dur_s"] = (df["end_s"] - df["start_s"]).round(3)
    df["source"] = "pred"
    plan = [f"{int(st['id'])}.{r}" for st in station.get("steps", []) for r in range(int(st.get("reps", 1)))]
    got = {f"{a}.{r}" for a, r in zip(df["step_id"], df["repetition"])}
    df["missing"] = ""
    info = {"not_found": [k for k in plan if k not in got], "no_zone": [],
            "matched_order": [int(a) for a in df["step_id"]]}
    return df[COLS + ["missing"]], info


def predict_steps(model_dir: str | Path | None, npz: str | Path, station: dict, device: str = "cpu",
                  min_seg_s: float | None = None):
    """Dùng trong 04: .npz → (bảng bước pred, info, config model)."""
    model, cfg = load_model(model_dir, device)
    kp = load_npz(npz)
    pose = kp["meta"].get("model", "")
    if cfg.get("pose_model") and pose and pose not in cfg["pose_model"]:
        log(f"cảnh báo: model train trên pose {cfg['pose_model']} nhưng .npz là {pose!r} — kết quả có thể kém")
    keys = predict_frames(model, cfg, kp, float(station["fps"]), device)
    msm = cfg["hparams"].get("min_seg_s", 0.5) if min_seg_s is None else min_seg_s
    df, info = frames_to_steps(keys, kp["t"], station, msm)
    return df, info, cfg


# ---------------------------------------------------------------- đánh giá mức bước (giống 04 eval)
def step_eval(model, cfg: dict, trial: ds.Trial, data_root: str, station: dict, device: str) -> dict:
    from kaizen.analyzer.steps import evaluate, oracle_from_labels
    from kaizen.skeleton import ha4m_io
    s, t = trial.stem.split("_")
    kp = load_npz(trial.npz)
    keys = predict_frames(model, cfg, kp, float(station["fps"]), device)
    pred, _ = frames_to_steps(keys, kp["t"], station, cfg["hparams"].get("min_seg_s", 0.5))
    orc = oracle_from_labels(ha4m_io.load_labels(data_root, s, t), station)
    e = evaluate(pred, orc).to_json_dict()
    gt = [k for k in trial.keys]
    frame_rep = segment_report(keys, gt, bg="0", ignore="")
    return {"trial": trial.stem, **{k: e[k] for k in ("f1_iou05", "mae_step_dur_s", "cycle_time_err_s",
                                                       "n_pred", "n_oracle")}, **frame_rep}


def summarize(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in ("trial", "fold") and isinstance(rows[0][k], (int, float))] if rows else []
    out = {}
    for k in keys:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)} if v else None
    return out


# ---------------------------------------------------------------- CLI
def _hp_from_args(a) -> dict:
    return {k: getattr(a, k) for k in DEFAULTS}


def _add_hp(p: argparse.ArgumentParser) -> None:
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v), default=v)


def _add_data(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="thư mục HA4M (nhiều thư mục: a,b)")
    p.add_argument("--keypoints-dir", default="data/keypoints")
    p.add_argument("--pose-model", default="mediapipe", help="hậu tố .npz: <stem>_<pose>.npz")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="station.yaml (fps, tên + số lần lặp bước)")
    p.add_argument("--subjects", default="", help="chỉ dùng các subject này (vd U001,U002)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--max-trials", type=int, default=0, help="giới hạn số trial (thử nhanh)")


def _subset(s: str) -> set[str] | None:
    v = {x.strip() for x in s.split(",") if x.strip()}
    return v or None


def cmd_status(a) -> int:
    refs = ds.scan(a.data_root, a.keypoints_dir, a.pose_model)
    ready = [r for r in refs if r.npz and r.has_labels]
    no_kp = [r for r in refs if not r.npz]
    no_lab = [r for r in refs if r.npz and not r.has_labels]
    subs = sorted({r.subject for r in ready})
    print(f"Trial trong {a.data_root}: {len(refs)} · sẵn sàng train (có keypoints {a.pose_model} + nhãn): "
          f"{len(ready)} trial / {len(subs)} subject")
    if subs:
        print("  subject sẵn sàng:", ", ".join(subs))
    if no_lab:
        print(f"  có keypoints nhưng THIẾU nhãn ({len(no_lab)}): {', '.join(r.stem for r in no_lab[:10])}")
    if no_kp:
        print(f"  THIẾU keypoints ({len(no_kp)}): {', '.join(r.stem for r in no_kp[:10])}"
              f"{' ...' if len(no_kp) > 10 else ''}")
        print(f"  → trích trước:  python src/01_extract.py --data-root {a.data_root} --subjects all --model {a.pose_model}")
    if len(subs) < 3:
        print("  ⚠ cần ≥ 3 subject để train + giữ subject riêng cho val (càng nhiều càng tốt; HA4M có 41 người).")
    if LATEST.exists():
        print(f"Model mới nhất: {MODELS_DIR / LATEST.read_text().strip()}")
    return 0


def cmd_train(a) -> int:
    hp = _hp_from_args(a)
    station = load_config(a.config)
    test = _subset(a.test_subjects) or set()
    trials, _ = load_trials(a.data_root, a.keypoints_dir, a.pose_model, _subset(a.subjects), float(station["fps"]),
                            a.max_trials)
    trials = [t for t in trials if t.subject not in test]
    if len(trials) < 2:
        log("cần ≥ 2 trial có keypoints + nhãn (xem: python -m kaizen.train.segmenter status)")
        return 1
    tr_s, va_s = ds.split_subjects([t.subject for t in trials], list(_subset(a.val_subjects) or []), a.val_frac, a.seed)
    train = [t for t in trials if t.subject in tr_s]
    val = [t for t in trials if t.subject in va_s]
    name = a.name or f"mstcn_{a.pose_model}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = MODELS_DIR / name
    if (out / "model.pt").exists() and not a.force:
        log(f"{out} đã có model (--force để ghi đè hoặc đổi --name)")
        return 1
    log(f"train subject: {', '.join(tr_s)} · val subject: {', '.join(va_s) or '—'} · test (loại khỏi train): "
        f"{', '.join(sorted(test)) or '—'}")
    model, cfg, hist = train_model(train, val, hp, a.device, None,
                                   {"name": name, "data_root": a.data_root, "test_subjects": sorted(test),
                                    "cmd": " ".join(sys.argv)})
    device = pick_device(a.device)
    if val:
        rows = [step_eval(model, cfg, t, a.data_root, station, device) for t in val]
        cfg["val_step_eval"] = {"per_trial": rows, "summary": summarize(rows)}
        s = cfg["val_step_eval"]["summary"]
        log(f"VAL (mức bước, như 04 eval): F1@IoU0.5 {s['f1_iou05']} · MAE bước {s['mae_step_dur_s']} s · "
            f"sai số CT {s['cycle_time_err_s']} s")
    save_model(model, cfg, out, hist)
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(name + "\n", encoding="utf-8")
    log(f"LATEST → {name}. Dùng: python src/04_steps.py --segmenter mstcn --segmenter-model {out}")
    return 0


def cmd_cv(a) -> int:
    hp = _hp_from_args(a)
    station = load_config(a.config)
    trials, _ = load_trials(a.data_root, a.keypoints_dir, a.pose_model, _subset(a.subjects), float(station["fps"]),
                            a.max_trials)
    folds = ds.group_folds([t.subject for t in trials], a.folds, a.seed)
    if len(folds) < 2:
        log("cần ≥ 2 subject để cross-validation")
        return 1
    device = pick_device(a.device)
    out = MODELS_DIR / (a.name or f"cv_{a.pose_model}_{dt.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, test_subs in enumerate(folds, 1):
        rest = [t for t in trials if t.subject not in test_subs]
        tr_s, va_s = ds.split_subjects([t.subject for t in rest], None, a.val_frac, a.seed + i)
        log(f"=== fold {i}/{len(folds)} · test {', '.join(test_subs)} · val {', '.join(va_s) or '—'}")
        model, cfg, _ = train_model([t for t in rest if t.subject in tr_s], [t for t in rest if t.subject in va_s],
                                    hp, device, (out / f"fold{i}") if a.save_folds else None)
        for t in trials:
            if t.subject in test_subs:
                rows.append({"fold": i, **step_eval(model, cfg, t, a.data_root, station, device)})
    rep = {"folds": [list(f) for f in folds], "hparams": hp, "per_trial": rows, "summary": summarize(rows),
           "cmd": " ".join(sys.argv)}
    (out / "cv_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    s = rep["summary"]
    log(f"CV ({len(folds)} fold, {len(rows)} trial test): F1@IoU0.5 {s['f1_iou05']} · frame acc {s['acc']} · "
        f"edit {s['edit']} · F1@50 {s['f1@50']} · MAE bước {s['mae_step_dur_s']} s → {out / 'cv_report.json'}")
    return 0


def cmd_eval(a) -> int:
    station = load_config(a.config)
    device = pick_device(a.device)
    model, cfg = load_model(a.model, device)
    trials, _ = load_trials(a.data_root, a.keypoints_dir, a.pose_model, _subset(a.subjects), float(station["fps"]),
                            a.max_trials)
    seen = set(cfg.get("train_trials", [])) | set(cfg.get("val_trials", []))
    leak = [t.stem for t in trials if t.stem in seen]
    if leak:
        log(f"⚠ {len(leak)} trial đã dùng khi train/val — số liệu sẽ lạc quan: {', '.join(leak[:8])}")
    rows = [step_eval(model, cfg, t, a.data_root, station, device) for t in trials]
    if not rows:
        log("không có trial nào để đánh giá")
        return 1
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    s = summarize(rows)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({"per_trial": rows, "summary": s}, ensure_ascii=False, indent=2), "utf-8")
    return 0


def cmd_predict(a) -> int:
    station = load_config(a.config)
    df, info, cfg = predict_steps(a.model, a.npz, station, pick_device(a.device))
    print(df.to_string(index=False))
    print("không tìm thấy:", info["not_found"] or "không")
    if a.out:
        df.to_csv(a.out, index=False)
        log(f"đã ghi {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status", help="kiểm tra dữ liệu sẵn sàng train")
    _add_data(p)
    p = sub.add_parser("train", help="train 1 model và lưu vào models/segmenter/<name>/")
    _add_data(p)
    _add_hp(p)
    p.add_argument("--name", default="")
    p.add_argument("--val-subjects", default="", help="subject dùng làm val (mặc định chọn ngẫu nhiên theo --val-frac)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-subjects", default="", help="subject giữ riêng để test (loại hẳn khỏi train/val)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("cv", help="cross-validation theo subject")
    _add_data(p)
    _add_hp(p)
    p.add_argument("--folds", type=int, default=5, help="số fold theo subject (0 = leave-one-subject-out)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--name", default="")
    p.add_argument("--save-folds", action="store_true", help="lưu model từng fold")
    p = sub.add_parser("eval", help="đánh giá model trên các trial có nhãn")
    _add_data(p)
    p.add_argument("--model", default="latest")
    p.add_argument("--out", default="")
    p = sub.add_parser("predict", help="dự đoán bước cho 1 file .npz")
    p.add_argument("--model", default="latest")
    p.add_argument("--npz", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="")
    a = ap.parse_args(argv)
    return {"status": cmd_status, "train": cmd_train, "cv": cmd_cv, "eval": cmd_eval, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

"""Model phân đoạn bước SOP (MS-TCN) — train / val / benchmark checkpoint / test / cv / predict + suy luận cho 04.

Học từ chuỗi keypoint (.npz, data/keypoints/) + nhãn theo frame của HA4M (Labels.txt). KHÔNG tự chạy ngầm:
mọi lần train đều do người dùng gõ lệnh. Hướng dẫn đầy đủ: docs/training.md (mục A).

    python -m kaizen.train.segmenter status
    python -m kaizen.train.segmenter train --name mstcn_v1 --val-subjects U008 --test-subjects U009,U010
    tensorboard --logdir models/segmenter                         # theo dõi loss / val metric theo epoch
    python -m kaizen.train.segmenter benchmark --model models/segmenter/mstcn_v1 --split val
    python -m kaizen.train.segmenter test --model models/segmenter/mstcn_v1          # subject test đã giữ riêng
    python -m kaizen.train.segmenter cv --folds 0                 # leave-one-subject-out
    python -m kaizen.train.segmenter predict --model models/segmenter/mstcn_v1 --npz data/keypoints/U009_V001_mediapipe.npz

Thư mục model models/segmenter/<name>/:
    model.pt            model ở epoch tốt nhất (theo F1@50 trên val) — dùng cho 04 / test
    config.json         lớp, chuẩn hoá, split train/val/test, siêu tham số, epoch tốt nhất, val_step_eval
    train_log.csv       lịch sử từng epoch (loss train/val, CE, smooth, acc, F1…)   train_curve.png
    checkpoints/        epoch_XXXX.pt mỗi --ckpt-every epoch (để benchmark theo epoch) + last.pt (để --resume)
    tb/                 log TensorBoard
    benchmark_<split>.{csv,json,png}   test/ (dự đoán từng frame, bảng bước, confusion matrix, Gantt, summary.json)
Dùng trong pipeline:  python src/04_steps.py --segmenter mstcn [--segmenter-model models/segmenter/mstcn_v1]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
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
from kaizen.train.features import FEATURE_VERSION, feature_dim, feature_names, frame_features, load_npz
from kaizen.train.metrics import mean_reports, segment_report

TAG = "seg"
MODELS_DIR = Path("models/segmenter")
LATEST = MODELS_DIR / "LATEST"
DEFAULTS = dict(stages=4, layers=10, channels=64, dropout=0.5, lambda_smooth=0.15, lr=5e-4, weight_decay=1e-4,
                epochs=100, batch_size=4, sample_rate=2, patience=20, eval_every=2, ckpt_every=10, seed=0,
                min_seg_s=0.5, threads=0, speed_min=1.0, speed_max=1.0, speed_mode="global",
                decode="argmax", feat_drop="")
FRAME_METRICS = ("acc", "edit", "f1@10", "f1@25", "f1@50")
STEP_METRICS = ("f1_iou05", "mae_step_dur_s", "cycle_time_err_s")


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TB:
    """SummaryWriter của TensorBoard; thiếu gói tensorboard → no-op (chỉ cảnh báo 1 lần)."""

    def __init__(self, logdir: Path | None, purge_step: int | None = None):
        self.w = None
        if logdir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:  # pragma: no cover
            log(f"không ghi TensorBoard ({e}) — cài: pip install tensorboard")
            return
        self.w = SummaryWriter(str(logdir), purge_step=purge_step)

    def scalars(self, prefix: str, d: dict, step: int) -> None:
        if self.w:
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    self.w.add_scalar(f"{prefix}/{k}", v, step)

    def text(self, tag: str, s: str) -> None:
        if self.w:
            self.w.add_text(tag, s)

    def close(self) -> None:
        if self.w:
            self.w.flush()
            self.w.close()


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


def speed_augment(e: ds.Encoded, s: float, norm: ds.Normalizer) -> ds.Encoded:
    """Co/giãn chuỗi theo thời gian hệ số s (s > 1 = làm chậm hơn, chuỗi dài hơn) — mô phỏng người thao tác
    nhanh/chậm khác nhau. Đặc trưng liên tục: nội suy tuyến tính; mask + nhãn: frame gần nhất; vận tốc ÷ s
    (tính trong không gian thô rồi chuẩn hoá lại, vì x đã z-score)."""
    L = len(e.x)
    n = max(2, int(round(L * s)))
    return _time_warp(e, np.clip(np.arange(n) / s, 0, L - 1), np.full(n, s, np.float32), norm)


def segment_speed_augment(e: ds.Encoded, factors, norm: ds.Normalizer) -> ds.Encoded:
    """Co/giãn TỪNG ĐOẠN nhãn (run liên tiếp cùng lớp) với hệ số riêng — người thật chậm/nhanh không đều giữa các
    bước (U002: bước 11 chậm 2.8×, bước 7 như U001). factors: callable() → hệ số cho mỗi đoạn."""
    y = e.y
    cuts = np.flatnonzero(np.diff(y)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(y)]])
    pos, fac = [], []
    for a, b in zip(starts, ends):
        f = float(factors())
        n = max(1, int(round((b - a) * f)))
        pos.append(a + np.arange(n) / f)
        fac.append(np.full(n, f, np.float32))
    p = np.clip(np.concatenate(pos), 0, len(y) - 1)
    return _time_warp(e, p, np.concatenate(fac), norm)


def _time_warp(e: ds.Encoded, p: np.ndarray, fac: np.ndarray, norm: ds.Normalizer) -> ds.Encoded:
    """Lấy mẫu lại chuỗi tại vị trí thực p (theo frame gốc); fac = hệ số giãn tại từng frame mới (chia vận tốc)."""
    L = len(e.x)
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, L - 1)
    w = (p - i0)[:, None].astype(np.float32)
    x = e.x[i0] * (1 - w) + e.x[i1] * w
    near = np.rint(p).astype(int)
    names = e.extra.get("names") or feature_names()
    msk = np.array([k.startswith("mask_") for k in names])
    vel = np.array([k.startswith("vel_") for k in names])
    x[:, msk] = e.x[near][:, msk]
    mu, sd = norm.mean[vel], norm.std[vel]
    x[:, vel] = ((x[:, vel] * sd + mu) / fac[:, None] - mu) / sd
    return ds.Encoded(e.stem, x.astype(np.float32), e.y[near], e.extra)


def _draw_speed(hp: dict) -> float:
    lo, hi = float(hp.get("speed_min", 1.0)), float(hp.get("speed_max", 1.0))
    if hi <= lo:
        return lo
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))   # log-uniform: 0.67 và 1.5 cân xứng


FEAT_GROUPS = ("hands", "abs", "rel", "vel", "mask")


def feature_columns(feat_drop: str = "") -> list[str]:
    """Tên cột đặc trưng giữ lại sau khi bỏ các nhóm trong feat_drop (phẩy): hands = điểm ngón tay lh_/rh_ (MediaPipe
    Hand hỏng khi đeo găng), abs = toạ độ tuyệt đối (lệch khi camera/chỗ đứng đổi), rel, vel, mask."""
    drop = {x.strip() for x in str(feat_drop).split(",") if x.strip()}
    bad = drop - set(FEAT_GROUPS)
    if bad:
        raise ValueError(f"feat_drop không hợp lệ: {sorted(bad)} (chọn trong {FEAT_GROUPS})")
    keep = []
    for n in feature_names():
        kind, joint = n.split("_", 1)
        is_hand = joint.startswith(("lh_", "rh_"))
        if ("hands" in drop and is_hand) or kind in drop:
            continue
        keep.append(n)
    if not keep:
        raise ValueError("feat_drop bỏ hết đặc trưng")
    return keep


def _col_index(names: list[str]) -> np.ndarray:
    allc = {n: i for i, n in enumerate(feature_names())}
    return np.array([allc[n] for n in names])


def build_model(hp: dict, dim_in: int, n_classes: int):
    from kaizen.train.mstcn import MSTCN
    return MSTCN(dim_in, n_classes, hp["stages"], hp["layers"], hp["channels"], hp["dropout"])


def _predict_encoded(model, enc: ds.Encoded, device: str) -> np.ndarray:
    """Lớp dự đoán theo frame (ở độ phân giải đã lấy mẫu)."""
    return _logprobs_encoded(model, enc, device).argmax(1)


def _logprobs_encoded(model, enc: ds.Encoded, device: str) -> np.ndarray:
    """log-softmax stage cuối, (T', C)."""
    torch = _torch()
    model.eval()
    with torch.no_grad():
        x, _, m = _batch([enc], device)
        return torch.log_softmax(model(x, m)[-1, 0], 0).T.cpu().numpy()[: len(enc.x)]


def sop_plan(classes: list[str], station: dict | None = None) -> list[str]:
    """Thứ tự bước chuẩn (SOP) = khoá 'bước.lần' theo station.yaml, chỉ giữ lớp model có; không có station → theo số."""
    plan = [f"{int(st['id'])}.{r}" for st in (station or {}).get("steps", []) for r in range(int(st.get("reps", 1)))]
    plan = [k for k in plan if k in classes] or [k for k in ds.sort_keys(classes) if k != "0"]
    return plan


def sop_decode(logp: np.ndarray, classes: list[str], plan: list[str], min_len: int = 1) -> np.ndarray:
    """Giải mã có ràng buộc thứ tự SOP (Viterbi / forced alignment): tìm cách chia chuỗi thành đúng các bước của plan
    THEO THỨ TỰ, mỗi bước ≥ min_len frame, nền '0' tuỳ chọn ở đầu/cuối/giữa hai bước, cực đại tổng log-prob.
    Lần lặp (2.0/2.1/2.2) vì vậy được phân biệt bằng thứ tự thay vì hình dáng. Trả chỉ số lớp theo frame.
    Giả định người làm ĐÚNG thứ tự SOP — sai thứ tự/bỏ bước sẽ bị che (dùng argmax để phát hiện lệch SOP)."""
    T = len(logp)
    idx = {k: i for i, k in enumerate(classes)}
    bg = idx.get("0")
    segs = []                                       # (lớp, độ dài tối thiểu, tuỳ chọn?)
    for k in plan:
        if bg is not None:
            segs.append((bg, 1, True))
        segs.append((idx[k], max(1, int(min_len)), False))
    if bg is not None:
        segs.append((bg, 1, True))
    need = sum(m for _, m, opt in segs if not opt)
    if T < need:                                    # chuỗi quá ngắn để chứa đủ bước → co min_len
        scale = max(1, T // max(1, len(plan)))
        segs = [(c, min(m, scale), o) for c, m, o in segs]
    st_cls, st_last, st_first_of, seg_first = [], [], [], []
    for si, (c, m, _) in enumerate(segs):
        seg_first.append(len(st_cls))
        for j in range(m):
            st_cls.append(c)
            st_last.append(j == m - 1)
    S = len(st_cls)
    st_cls, st_last = np.array(st_cls), np.array(st_last)
    seg_last = [seg_first[i] + segs[i][1] - 1 for i in range(len(segs))]
    skip_src = np.full(S, -1)                       # bỏ qua đoạn nền tuỳ chọn: cuối đoạn i → đầu đoạn i+2
    for i in range(len(segs) - 2):
        if segs[i + 1][2]:
            skip_src[seg_first[i + 2]] = seg_last[i]
    NEG = -1e18
    score = np.full(S, NEG)
    score[seg_first[0]] = 0.0
    if segs[0][2] and len(segs) > 1:
        score[seg_first[1]] = 0.0
    score = score + logp[0, st_cls]
    back = np.zeros((T, S), np.int8)                # 0 ở lại, 1 từ trạng thái trước, 2 nhảy qua nền
    has_skip = skip_src >= 0
    for t in range(1, T):
        stay = np.where(st_last, score, NEG)
        prev = np.concatenate([[NEG], score[:-1]])
        skip = np.where(has_skip, score[np.maximum(skip_src, 0)], NEG)
        cand = np.stack([stay, prev, skip])
        b = cand.argmax(0)
        score = cand[b, np.arange(S)] + logp[t, st_cls]
        back[t] = b
    ends = [seg_last[-1]] + ([seg_last[-2]] if segs[-1][2] and len(segs) > 1 else [])
    s = max(ends, key=lambda j: score[j])
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = st_cls[s]
        b = back[t, s]
        s = s if b == 0 else s - 1 if b == 1 else skip_src[s]
    return path


def _eval_split(model, encs: list[ds.Encoded], device: str, lam: float) -> dict:
    """Loss + chỉ số frame (acc, edit, F1@k) trung bình trên các chuỗi (mỗi chuỗi 1 forward, eval mode)."""
    torch = _torch()
    from kaizen.train.mstcn import mstcn_loss
    model.eval()
    losses, reps = [], []
    with torch.no_grad():
        for e in encs:
            x, y, m = _batch([e], device)
            out = model(x, m)
            losses.append(float(mstcn_loss(out, y, m, lam)))
            pred = out[-1, 0].argmax(0).cpu().numpy()[: len(e.x)]
            reps.append(segment_report(pred, e.y, bg=0, ignore=ds.IGNORE))
    return {"loss": round(float(np.mean(losses)), 4), **mean_reports(reps)} if encs else {}


def _eval_frames(model, encs: list[ds.Encoded], device: str) -> dict:
    reps = [segment_report(_predict_encoded(model, e, device), e.y, bg=0, ignore=ds.IGNORE) for e in encs]
    return mean_reports(reps)


# ---------------------------------------------------------------- dữ liệu
def load_trials(data_root: str, kp_dir: str, pose_model: str, subjects: set[str] | None = None,
                fps: float = 30.0, max_trials: int = 0, stems: set[str] | None = None
                ) -> tuple[list[ds.Trial], list[ds.TrialRef]]:
    refs = [r for r in ds.scan(data_root, kp_dir, pose_model) if r.npz and r.has_labels
            and (not subjects or r.subject in subjects) and (not stems or r.stem in stems)]
    if max_trials:
        refs = refs[:max_trials]
    t0 = time.time()
    trials = [ds.load_trial(data_root, r, fps) for r in refs]
    log(f"nạp {len(trials)} trial ({sum(len(t.feats) for t in trials)} frame) trong {time.time() - t0:.1f}s")
    return trials, refs


# ---------------------------------------------------------------- train
def _rng_state() -> dict:
    torch = _torch()
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _ckpt_payload(model, cfg_base: dict, ep: int, row: dict) -> dict:
    return {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": {**cfg_base, "epoch": ep, "epoch_metrics": row}}


def train_model(train: list[ds.Trial], val: list[ds.Trial], hp: dict, device: str = "auto",
                out_dir: Path | None = None, extra_config: dict | None = None, *,
                ckpt_dir: Path | None = None, tb_dir: Path | None = None,
                resume: bool = False) -> tuple[object, dict, list[dict]]:
    """Train MS-TCN; early stopping theo F1@50 trên val (không có val → theo loss train).

    ckpt_dir: lưu checkpoints/epoch_XXXX.pt mỗi hp['ckpt_every'] epoch (0 = tắt) + last.pt mỗi epoch (resume).
    tb_dir: thư mục log TensorBoard (train/*, val/*, time/*).  resume: tiếp tục từ ckpt_dir/last.pt.
    Trả (model ở epoch tốt nhất, config, lịch sử từng epoch)."""
    torch = _torch()
    from kaizen.train.mstcn import mstcn_loss
    device = pick_device(device)
    set_seed(hp["seed"])
    n_threads = set_threads(hp.get("threads", 0)) if device == "cpu" else None
    last_path = ckpt_dir / "last.pt" if ckpt_dir else None
    state = None
    feat_names = feature_columns(hp.get("feat_drop", ""))
    if len(feat_names) != feature_dim():
        cols = _col_index(feat_names)
        train = [dataclasses.replace(t, feats=t.feats[:, cols]) for t in train]
        val = [dataclasses.replace(t, feats=t.feats[:, cols]) for t in val]
    if resume:
        if not last_path or not last_path.exists():
            raise FileNotFoundError(f"không có checkpoint để resume: {last_path}")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        classes = state["config"]["classes"]
        norm = ds.Normalizer.from_json(state["config"]["normalizer"])
        log(f"resume từ {last_path} (epoch {state['epoch']})")
    else:
        classes = ds.sort_keys(k for t in train for k in t.keys if k)
        if "0" not in classes:
            classes = ["0"] + classes
        norm = ds.Normalizer.fit([t.feats for t in train])
    sr = hp["sample_rate"]
    tr = [ds.encode(t, classes, norm, sr) for t in train]
    va = [ds.encode(t, classes, norm, sr) for t in val]
    for e in tr + va:
        e.extra["names"] = feat_names
    model = build_model(hp, len(feat_names), len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    cfg_base = {"version": 1, "arch": "mstcn", "feature_version": FEATURE_VERSION, "feature_dim": len(feat_names),
                "feature_cols": feat_names,
                "classes": classes, "hparams": hp, "normalizer": norm.to_json(), "n_params": n_params,
                "train_trials": [t.stem for t in train], "val_trials": [t.stem for t in val],
                "train_subjects": sorted({t.subject for t in train}), "val_subjects": sorted({t.subject for t in val}),
                "pose_model": sorted({t.pose_model for t in train}), **(extra_config or {})}
    best, best_state, best_ep, wait, hist, start_ep = -math.inf, None, 0, 0, [], 1
    rng = random.Random(hp["seed"])
    if state is not None:
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        best, best_ep, wait, hist = state["best"], state["best_epoch"], state["wait"], state["hist"]
        best_state = state["best_state"]
        start_ep = state["epoch"] + 1
        rng.setstate(state["shuffle_rng"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb = _TB(tb_dir, purge_step=start_ep if state is not None else None)
    if state is None:
        tb.text("config/hparams", "\n".join(f"- {k}: {v}" for k, v in hp.items()))
        tb.text("config/split", f"train: {cfg_base['train_trials']}\n\nval: {cfg_base['val_trials']}")
    log(f"device={device}{f' ({n_threads} luồng)' if n_threads else ''} · {len(tr)} train / {len(va)} val chuỗi · "
        f"{len(classes)} lớp · {n_params / 1e3:.0f}k tham số · ckpt mỗi {hp.get('ckpt_every', 0) or '—'} epoch")
    t_start = time.time() - (hist[-1].get("elapsed_s", 0) if hist else 0)
    for ep in range(start_ep, hp["epochs"] + 1):
        t_ep = time.time()
        model.train()
        order = list(range(len(tr)))
        rng.shuffle(order)
        tot = ce = sm = 0.0
        nb = correct = counted = 0
        for i in range(0, len(order), hp["batch_size"]):
            items = [tr[j] for j in order[i:i + hp["batch_size"]]]
            if hp.get("speed_max", 1.0) > hp.get("speed_min", 1.0) or hp.get("speed_min", 1.0) != 1.0:
                if hp.get("speed_mode", "global") == "segment":
                    items = [segment_speed_augment(e, lambda: _draw_speed(hp), norm) for e in items]
                else:
                    items = [speed_augment(e, _draw_speed(hp), norm) for e in items]
            x, y, m = _batch(items, device)
            out = model(x, m)
            loss, parts = mstcn_loss(out, y, m, hp["lambda_smooth"], return_parts=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, ce, sm, nb = tot + loss.item(), ce + parts["ce"], sm + parts["smooth"], nb + 1
            with torch.no_grad():
                keep = y != ds.IGNORE
                correct += int((out[-1].argmax(1)[keep] == y[keep]).sum())
                counted += int(keep.sum())
        row = {"epoch": ep, "loss": round(tot / max(nb, 1), 4), "loss_ce": round(ce / max(nb, 1), 4),
               "loss_smooth": round(sm / max(nb, 1), 4), "train_acc": round(100 * correct / max(counted, 1), 2),
               "lr": opt.param_groups[0]["lr"]}
        evaluated = ep % hp["eval_every"] == 0 or ep == hp["epochs"]
        if evaluated:
            if va:
                row.update({f"val_{k}": v for k, v in _eval_split(model, va, device, hp["lambda_smooth"]).items()})
                score = row["val_f1@50"]
            else:
                score = -row["loss"]
            if score > best:
                best, best_ep, wait = score, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += hp["eval_every"]
        row["epoch_s"] = round(time.time() - t_ep, 2)
        row["elapsed_s"] = round(time.time() - t_start, 1)
        hist.append(row)
        tb.scalars("train", {k: row[k] for k in ("loss", "loss_ce", "loss_smooth", "train_acc", "lr")}, ep)
        tb.scalars("val", {k[4:]: v for k, v in row.items() if k.startswith("val_")}, ep)
        tb.scalars("time", {"epoch_s": row["epoch_s"]}, ep)
        if evaluated:
            tb.scalars("val", {"best_f1@50" if va else "best_score": best}, ep)
            log(" · ".join(f"{k}={v}" for k, v in row.items() if k not in ("lr", "elapsed_s"))
                + f" · {row['elapsed_s']:.0f}s")
        ce_every = hp.get("ckpt_every", 0)
        if ckpt_dir and ce_every and (ep % ce_every == 0 or ep == hp["epochs"]):
            torch.save(_ckpt_payload(model, cfg_base, ep, row), ckpt_dir / f"epoch_{ep:04d}.pt")
        stop = evaluated and wait >= hp["patience"]
        if last_path:
            torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep, "best": best,
                        "best_epoch": best_ep, "best_state": best_state, "wait": wait, "hist": hist,
                        "shuffle_rng": rng.getstate(), "rng": _rng_state(), "config": cfg_base,
                        "stopped": stop}, last_path)
        if stop:
            log(f"early stop ở epoch {ep} (tốt nhất epoch {best_ep})")
            if ckpt_dir and ce_every and ep % ce_every:
                torch.save(_ckpt_payload(model, cfg_base, ep, row), ckpt_dir / f"epoch_{ep:04d}.pt")
            break
    tb.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    cfg = {**cfg_base, "best_epoch": best_ep, "epochs_run": hist[-1]["epoch"] if hist else 0,
           "train_seconds": round(time.time() - t_start, 1), "device": device,
           "trained_at": dt.datetime.now().isoformat(timespec="seconds")}
    if out_dir is not None:
        save_model(model, cfg, out_dir, hist)
    return model, cfg, hist


def plot_history(hist: list[dict], path: Path, best_epoch: int | None = None) -> None:
    """train_curve.png: loss train/val (trái) + acc / F1@50 val (phải) theo epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(hist)
    if df.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(df["epoch"], df["loss"], label="train loss")
    if "val_loss" in df:
        v = df.dropna(subset=["val_loss"])
        ax[0].plot(v["epoch"], v["val_loss"], "o-", ms=3, label="val loss")
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("loss"), ax[0].legend(), ax[0].grid(alpha=.3)
    ax[1].plot(df["epoch"], df["train_acc"], label="train acc (frame)")
    for k in ("val_acc", "val_edit", "val_f1@50"):
        if k in df:
            v = df.dropna(subset=[k])
            ax[1].plot(v["epoch"], v[k], "o-", ms=3, label=k)
    if best_epoch:
        for a in ax:
            a.axvline(best_epoch, color="gray", ls="--", lw=1)
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("%"), ax[1].legend(fontsize=8), ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        try:
            plot_history(hist, out_dir / "train_curve.png", cfg.get("best_epoch"))
        except Exception as e:  # pragma: no cover — vẽ hình không được làm hỏng việc lưu model
            log(f"không vẽ được train_curve.png: {e}")
    log(f"đã lưu model → {out_dir}")


def load_model(model_dir: str | Path, device: str = "cpu"):
    """(model, config) từ models/segmenter/<name>/ (hoặc 'latest'), hoặc trực tiếp 1 file .pt (checkpoint)."""
    torch = _torch()
    p = Path(model_dir) if model_dir not in (None, "", "latest") else None
    path = p if p is not None and p.suffix == ".pt" else resolve_model_dir(model_dir) / "model.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
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
    p = Path(model_dir)
    return p.parent.parent if p.suffix == ".pt" and p.parent.name == "checkpoints" else p


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


def predict_feats(model, cfg: dict, feats: np.ndarray, device: str = "cpu", decode: str | None = None,
                  station: dict | None = None) -> list[str]:
    """Lớp ('0', '2.1', …) cho TỪNG frame từ đặc trưng (T, D) (dự đoán ở 1/sample_rate rồi lặp lại).
    decode: argmax (từng frame) | sop (ràng buộc thứ tự SOP, xem sop_decode); None → hparams['decode'] của model."""
    norm = ds.Normalizer.from_json(cfg["normalizer"])
    sr = int(cfg["hparams"]["sample_rate"])
    if cfg.get("feature_cols") and len(cfg["feature_cols"]) != feats.shape[1]:
        feats = feats[:, _col_index(cfg["feature_cols"])]
    enc = ds.Encoded("x", norm(feats[::sr]), np.zeros(len(feats[::sr]), np.int64))
    decode = decode or cfg["hparams"].get("decode", "argmax")
    if decode == "sop":
        fps_eff = float((station or {}).get("fps", 30.0)) / sr
        min_len = max(1, int(round(cfg["hparams"].get("min_seg_s", 0.5) * fps_eff)))
        pred = sop_decode(_logprobs_encoded(model, enc, device), cfg["classes"],
                          sop_plan(cfg["classes"], station), min_len)
    elif decode == "argmax":
        pred = _predict_encoded(model, enc, device)
    else:
        raise ValueError(f"decode không hợp lệ: {decode} (argmax | sop)")
    full = np.repeat(pred, sr)[: len(feats)]
    if len(full) < len(feats):
        full = np.concatenate([full, np.full(len(feats) - len(full), full[-1] if len(full) else 0)])
    return [cfg["classes"][i] for i in full]


def predict_frames(model, cfg: dict, kp: dict, fps: float, device: str = "cpu", decode: str | None = None,
                   station: dict | None = None) -> list[str]:
    """Lớp cho TỪNG frame của .npz."""
    return predict_feats(model, cfg, frame_features(kp, fps), device, decode, station)


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
    df[["start_s", "end_s"]] = df[["start_s", "end_s"]].astype(float)    # rỗng (model chỉ đoán nền) vẫn là số
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
    keys = predict_frames(model, cfg, kp, float(station["fps"]), device, station=station)
    msm = cfg["hparams"].get("min_seg_s", 0.5) if min_seg_s is None else min_seg_s
    df, info = frames_to_steps(keys, kp["t"], station, msm)
    return df, info, cfg


# ---------------------------------------------------------------- đánh giá mức bước (giống 04 eval)
def infer_trial(model, cfg: dict, trial: ds.Trial, data_root: str, station: dict, device: str,
                decode: str | None = None) -> dict:
    """Suy luận 1 trial có nhãn → {keys, pred (bảng bước), oracle, metrics (frame + mức bước), infer_s}."""
    from kaizen.analyzer.steps import evaluate, oracle_from_labels
    from kaizen.skeleton import ha4m_io
    s, t = trial.stem.split("_")
    t0 = time.time()
    keys = predict_feats(model, cfg, trial.feats, device, decode, station)
    infer_s = time.time() - t0
    pred, _ = frames_to_steps(keys, trial.t, station, cfg["hparams"].get("min_seg_s", 0.5))
    orc = oracle_from_labels(ha4m_io.load_labels(data_root, s, t), station)
    pred, orc, gt, cut_s, dropped = crop_to_observed(pred, orc, list(trial.keys), trial.t, float(station["fps"]))
    e = evaluate(pred, orc).to_json_dict()
    frame_rep = segment_report(keys, gt, bg="0", ignore="")
    metrics = {"trial": trial.stem, **{k: e[k] for k in STEP_METRICS + ("n_pred", "n_oracle")}, **frame_rep}
    if cut_s is not None:
        metrics.update(eval_cut_s=round(cut_s, 3), steps_unobserved=",".join(dropped))
    return {"keys": keys, "gt": gt, "pred": pred, "oracle": orc, "metrics": metrics, "infer_s": infer_s}


def crop_to_observed(pred: pd.DataFrame, orc: pd.DataFrame, gt: list[str], t: np.ndarray, fps: float):
    """Video thiếu ảnh ở cuối (keypoints ngắn hơn nhãn, vd zip giải nén dở): chỉ chấm phần ĐÃ QUAN SÁT.
    Cắt tại đầu bước oracle đầu tiên bị cụt → bỏ các bước từ đó khỏi oracle, cắt pred, frame sau mốc → "" (bỏ qua).
    Video đủ → trả nguyên. Trả (pred, orc, gt, cut_s | None, [khoá bước bị bỏ])."""
    t_obs = float(t[-1]) + 1.0 / fps if len(t) else 0.0
    cut_rows = orc[orc["end_s"] > t_obs + 1.5 / fps]
    if cut_rows.empty:
        return pred, orc, gt, None, []
    cut = float(cut_rows["start_s"].min())
    dropped = [f"{int(a)}.{int(r)}" for a, r in orc.loc[orc["end_s"] > cut + 1e-6, ["step_id", "repetition"]]
               .itertuples(index=False)]
    orc = orc[orc["end_s"] <= cut + 1e-6].reset_index(drop=True)
    pred = pred[pred["start_s"] < cut].copy()
    pred["end_s"] = pred["end_s"].clip(upper=cut)
    pred["dur_s"] = (pred["end_s"] - pred["start_s"]).round(3)
    gt = [k if tt < cut else "" for k, tt in zip(gt, t)]
    return pred.reset_index(drop=True), orc, gt, cut, dropped


def step_eval(model, cfg: dict, trial: ds.Trial, data_root: str, station: dict, device: str,
              decode: str | None = None) -> dict:
    return infer_trial(model, cfg, trial, data_root, station, device, decode)["metrics"]


def summarize(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in ("trial", "fold") and isinstance(rows[0][k], (int, float))] if rows else []
    out = {}
    for k in keys:
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)} if v else None
    return out


def confusion(gt: list[str], pred: list[str], classes: list[str]) -> np.ndarray:
    """Ma trận nhầm lẫn theo frame (hàng = nhãn thật, cột = dự đoán); frame không nhãn ("") bị bỏ."""
    idx = {k: i for i, k in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), np.int64)
    for g, p in zip(gt, pred):
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1
    return cm


def per_class_report(cm: np.ndarray, classes: list[str]) -> pd.DataFrame:
    tp = np.diag(cm).astype(float)
    sup, npred = cm.sum(1).astype(float), cm.sum(0).astype(float)
    prec = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return pd.DataFrame({"class": classes, "support_frames": sup.astype(int), "precision": (100 * prec).round(2),
                         "recall": (100 * rec).round(2), "f1": (100 * f1).round(2)})


def _plot_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, rs, out=np.zeros(cm.shape), where=rs > 0) * 100
    fig, ax = plt.subplots(figsize=(0.45 * len(classes) + 2.5, 0.45 * len(classes) + 2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] >= 1:
                ax.text(j, i, f"{cmn[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 50 else "black")
    ax.set_xlabel("dự đoán"), ax.set_ylabel("nhãn thật"), ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="% frame theo hàng")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gantt(gt: list[str], pred: list[str], t: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kaizen.train.metrics import segments
    cmap = plt.get_cmap("tab20")
    col = {k: cmap(i % 20) for i, k in enumerate(k for k in classes if k != "0")}
    fig, ax = plt.subplots(figsize=(12, 2.2))
    dt_ = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    for row, lab in ((1, gt), (0, pred)):
        for k, s, e in segments(lab, bg="0"):
            if k in col:
                ax.broken_barh([(t[s], t[e - 1] - t[s] + dt_)], (row - 0.4, 0.8), color=col[k])
                if t[e - 1] - t[s] > 2:
                    ax.text(t[s] + (t[e - 1] - t[s]) / 2, row, k, ha="center", va="center", fontsize=7)
    ax.set_yticks([0, 1], ["MS-TCN", "nhãn HA4M"])
    ax.set_xlabel("thời gian (s)"), ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    for s in subs:
        print(f"  {s}: {', '.join(r.stem.split('_')[1] for r in ready if r.subject == s)}")
    if no_lab:
        print(f"  có keypoints nhưng THIẾU nhãn ({len(no_lab)}): {', '.join(r.stem for r in no_lab[:10])}")
    if no_kp:
        print(f"  THIẾU keypoints ({len(no_kp)}): {', '.join(r.stem for r in no_kp[:10])}"
              f"{' ...' if len(no_kp) > 10 else ''}")
        print(f"  → trích trước:  python src/01_extract.py --data-root {a.data_root} --subjects all --model {a.pose_model}")
    if len(subs) < 3:
        print("  ⚠ cần ≥ 3 subject để có train / val / test tách theo người (HA4M có 41 người).")
    if LATEST.exists():
        print(f"Model mới nhất: {MODELS_DIR / LATEST.read_text().strip()}")
    return 0


def cmd_train(a) -> int:
    hp = _hp_from_args(a)
    station = load_config(a.config)
    test_s = _subset(a.test_subjects) or set()
    test_t = _subset(a.test_trials) or set()
    val_t = _subset(a.val_trials) or set()
    name = a.name or f"mstcn_{a.pose_model}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = MODELS_DIR / name
    if a.resume and not (out / "checkpoints" / "last.pt").exists():
        log(f"--resume nhưng không thấy {out / 'checkpoints' / 'last.pt'}")
        return 1
    if (out / "model.pt").exists() and not (a.force or a.resume):
        log(f"{out} đã có model (--force để ghi đè, --resume để train tiếp, hoặc đổi --name)")
        return 1
    split_info: dict = {}
    if a.split_file:
        from kaizen.train.split import load_split
        sp = load_split(a.split_file)
        if a.val_subjects or a.val_trials or a.test_subjects or a.test_trials:
            log("⚠ có --split-file → bỏ qua --val-*/--test-*")
        want_tr, want_va = set(sp["train"]), set(sp["val"])
        trials, _ = load_trials(a.data_root, a.keypoints_dir, a.pose_model, None, float(station["fps"]), 0,
                                want_tr | want_va)
        missing = sorted((want_tr | want_va) - {t.stem for t in trials})
        if missing:
            log(f"⚠ {len(missing)} trial trong split thiếu .npz/nhãn, bỏ qua: {', '.join(missing)} → trích: "
                f"python src/01_extract.py --data-root {a.data_root} --subjects all --trials all --model {a.pose_model}")
        train = [t for t in trials if t.stem in want_tr]
        val = [t for t in trials if t.stem in want_va]
        test_s, test_t = set(), set(sp["test"])
        split_info = {"split_file": str(a.split_file), "split_name": sp.get("name", ""),
                      "split_strategy": sp.get("strategy", "")}
        for w in sp.get("warnings", []):
            log(f"⚠ split: {w}")
    else:
        trials, _ = load_trials(a.data_root, a.keypoints_dir, a.pose_model, _subset(a.subjects),
                                float(station["fps"]), a.max_trials)
        trials = [t for t in trials if t.subject not in test_s and t.stem not in test_t]
        if len(trials) < 2:
            log("cần ≥ 2 trial có keypoints + nhãn (xem: python -m kaizen.train.segmenter status)")
            return 1
    if a.split_file:
        pass
    elif val_t:
        val = [t for t in trials if t.stem in val_t]
        train = [t for t in trials if t.stem not in val_t]
        overlap = sorted({t.subject for t in val} & {t.subject for t in train})
        if overlap:
            log(f"⚠ val chia THEO TRIAL, subject {overlap} có ở cả train và val → số val lạc quan "
                "(chỉ dùng để chạy thử; báo cáo phải chia theo subject)")
    else:
        tr_s, va_s = ds.split_subjects([t.subject for t in trials], list(_subset(a.val_subjects) or []),
                                       a.val_frac, a.seed)
        train = [t for t in trials if t.subject in tr_s]
        val = [t for t in trials if t.subject in va_s]
    if not train:
        log("tập train rỗng")
        return 1
    log(f"train: {', '.join(t.stem for t in train)}")
    log(f"val:   {', '.join(t.stem for t in val) or '—'}")
    log(f"test (loại khỏi train/val): {', '.join(sorted(test_s | test_t)) or '—'}")
    if a.force and not a.resume and out.exists():
        for p in list((out / "checkpoints").glob("*.pt")) + list((out / "tb").glob("events.*")):
            p.unlink()
    model, cfg, hist = train_model(train, val, hp, a.device, None,
                                   {"name": name, "data_root": a.data_root, "test_subjects": sorted(test_s),
                                    "test_trials": sorted(test_t), "cmd": " ".join(sys.argv), **split_info,
                                    "torch": _torch().__version__},
                                   ckpt_dir=out / "checkpoints", tb_dir=out / "tb", resume=a.resume)
    device = pick_device(a.device)
    if val:
        rows = [step_eval(model, cfg, t, a.data_root, station, device) for t in val]
        cfg["val_step_eval"] = {"per_trial": rows, "summary": summarize(rows)}
        s = cfg["val_step_eval"]["summary"]
        log(f"VAL @ epoch {cfg['best_epoch']} (mức bước, như 04 eval): F1@IoU0.5 {s.get('f1_iou05', '—')} · "
            f"MAE bước {s.get('mae_step_dur_s', '—')} s · sai số CT {s.get('cycle_time_err_s', '—')} s · "
            f"F1@50 {s.get('f1@50', '—')}")
    save_model(model, cfg, out, hist)
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(name + "\n", encoding="utf-8")
    log(f"LATEST → {name} · train {cfg['train_seconds']:.0f}s · TensorBoard: tensorboard --logdir {out / 'tb'}")
    log(f"Tiếp: python -m kaizen.train.segmenter benchmark --model {out} --split val   |   "
        f"python -m kaizen.train.segmenter test --model {out}")
    return 0


def _split_trials(a, cfg: dict, station: dict) -> tuple[str, list[ds.Trial]]:
    """Trial cho benchmark/test: --subjects/--trials tường minh > split ghi trong config.json của model."""
    fps = float(station["fps"])
    if a.subjects or getattr(a, "trials", ""):
        trials, _ = load_trials(a.data_root, a.keypoints_dir, a.pose_model, _subset(a.subjects), fps,
                                a.max_trials, _subset(getattr(a, "trials", "")))
        return "custom", trials
    split = getattr(a, "split", "test")
    if split == "val":
        stems = set(cfg.get("val_trials", []))
        subs = None
    elif split == "train":
        stems = set(cfg.get("train_trials", []))
        subs = None
    else:
        stems = set(cfg.get("test_trials", []))
        subs = set(cfg.get("test_subjects", []))
    if not stems and not subs:
        log(f"config.json không ghi tập {split} — truyền --subjects hoặc --trials")
        return split, []
    refs = [r for r in ds.scan(a.data_root, a.keypoints_dir, a.pose_model) if r.npz and r.has_labels
            and (r.stem in stems or (subs and r.subject in subs))]
    missing = sorted(stems - {r.stem for r in refs})
    if missing:
        log(f"⚠ thiếu keypoints/nhãn cho {len(missing)} trial của tập {split}: {', '.join(missing[:8])}")
    t0 = time.time()
    trials = [ds.load_trial(a.data_root, r, fps) for r in refs]
    log(f"tập {split}: {len(trials)} trial ({time.time() - t0:.1f}s)")
    return split, trials


def _leak_check(cfg: dict, trials: list[ds.Trial], split: str) -> list[str]:
    seen = set(cfg.get("train_trials", [])) | (set(cfg.get("val_trials", [])) if split != "val" else set())
    seen_s = set(cfg.get("train_subjects", [])) | (set(cfg.get("val_subjects", [])) if split != "val" else set())
    leak = [t.stem for t in trials if t.stem in seen or t.subject in seen_s]
    if leak and split not in ("train",):
        log(f"⚠ {len(leak)} trial thuộc trial/subject đã dùng khi train"
            f"{'' if split == 'val' else '/val'} — số liệu sẽ lạc quan: {', '.join(leak[:8])}")
    return leak


def cmd_benchmark(a) -> int:
    """Đánh giá MỌI checkpoint (epoch_XXXX.pt) + model.pt trên 1 tập → bảng/biểu đồ chỉ số theo epoch."""
    station = load_config(a.config)
    device = pick_device(a.device)
    mdir = resolve_model_dir(a.model)
    cfg_best = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    split, trials = _split_trials(a, cfg_best, station)
    if not trials:
        return 1
    if split == "test":
        log("⚠ benchmark trên TEST chỉ để vẽ đường cong cho báo cáo — KHÔNG chọn epoch/siêu tham số theo test "
            "(chọn theo val; model.pt đã là epoch tốt nhất theo val)")
    _leak_check(cfg_best, trials, split)
    ckpts = sorted((mdir / "checkpoints").glob("epoch_*.pt"))
    if not ckpts:
        log(f"không có checkpoint trong {mdir / 'checkpoints'} (train với --ckpt-every > 0)")
        return 1
    rows = []
    tb = _TB(mdir / "tb" / f"benchmark_{split}") if not a.no_tb else _TB(None)
    for ck in ckpts:
        model, cfg = load_model(ck, device)
        ep = int(cfg.get("epoch", int(ck.stem.split("_")[1])))
        per = [step_eval(model, cfg, t, a.data_root, station, device) for t in trials]
        s = summarize(per)
        row = {"epoch": ep, "checkpoint": ck.name, **{k: s[k]["mean"] for k in FRAME_METRICS + STEP_METRICS},
               **{f"{k}_std": s[k]["std"] for k in ("f1@50", "f1_iou05")}}
        rows.append(row)
        tb.scalars(f"benchmark_{split}", {k: row[k] for k in FRAME_METRICS + STEP_METRICS}, ep)
        log(f"epoch {ep:4d}: " + " · ".join(f"{k}={row[k]}" for k in FRAME_METRICS + STEP_METRICS))
    tb.close()
    df = pd.DataFrame(rows).sort_values("epoch")
    best_ep = cfg_best.get("best_epoch")
    suffix = split if split != "custom" else (a.tag or "custom")
    df.to_csv(mdir / f"benchmark_{suffix}.csv", index=False)
    (mdir / f"benchmark_{suffix}.json").write_text(json.dumps(
        {"split": split, "trials": [t.stem for t in trials], "best_epoch_by_val": best_ep,
         "rows": df.to_dict("records"), "cmd": " ".join(sys.argv)}, ensure_ascii=False, indent=2), "utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for k in FRAME_METRICS:
            ax[0].plot(df["epoch"], df[k], "o-", ms=3, label=k)
        ax[0].set_ylabel("%"), ax[0].set_title(f"Chỉ số frame/đoạn — tập {suffix}", fontsize=10)
        ax[1].plot(df["epoch"], df["mae_step_dur_s"], "o-", ms=3, label="MAE thời lượng bước (s)")
        ax[1].plot(df["epoch"], df["cycle_time_err_s"], "o-", ms=3, label="|sai số cycle time| (s)")
        ax2 = ax[1].twinx()
        ax2.plot(df["epoch"], df["f1_iou05"], "k--", lw=1, label="F1@IoU0.5 (mức bước)")
        ax2.set_ylabel("F1@IoU0.5"), ax2.legend(loc="upper center", fontsize=8)
        ax[1].set_ylabel("giây"), ax[1].set_title("Chỉ số mức bước (pipeline 04)", fontsize=10)
        for x in ax:
            x.set_xlabel("epoch"), x.grid(alpha=.3), x.legend(fontsize=8, loc="lower right")
            if best_ep:
                x.axvline(best_ep, color="gray", ls=":", lw=1)
        fig.tight_layout()
        fig.savefig(mdir / f"benchmark_{suffix}.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        log(f"không vẽ được biểu đồ: {e}")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    log(f"→ {mdir / f'benchmark_{suffix}.csv'} (+ .json, .png) · epoch tốt nhất theo val: {best_ep}")
    return 0


def cmd_test(a) -> int:
    """Suy luận + đánh giá trên tập test giữ riêng: dự đoán từng frame, bảng bước, confusion matrix, Gantt."""
    station = load_config(a.config)
    device = pick_device(a.device)
    mdir = resolve_model_dir(a.model)
    src = mdir / "checkpoints" / f"{a.checkpoint}.pt" if a.checkpoint not in ("", "best") else mdir
    model, cfg = load_model(src, device)
    a.split = "test"
    split, trials = _split_trials(a, cfg, station)
    if not trials:
        return 1
    leak = _leak_check(cfg, trials, split)
    out = Path(a.out_dir) if a.out_dir else mdir / (("test" if a.checkpoint in ("", "best") else f"test_{a.checkpoint}")
                                                    + (f"_{a.decode}" if a.decode else ""))
    (out / "trials").mkdir(parents=True, exist_ok=True)
    classes = cfg["classes"]
    rows, all_gt, all_pred, times = [], [], [], []
    for tr in trials:
        r = infer_trial(model, cfg, tr, a.data_root, station, device, a.decode or None)
        rows.append(r["metrics"])
        times.append({"trial": tr.stem, "frames": len(tr.feats), "infer_s": round(r["infer_s"], 4)})
        if "eval_cut_s" in r["metrics"]:
            log(f"⚠ {tr.stem}: video thiếu ảnh cuối → chỉ chấm đến {r['metrics']['eval_cut_s']} s, "
                f"bỏ bước {r['metrics']['steps_unobserved']}")
        all_gt += r["gt"]
        all_pred += r["keys"]
        pd.DataFrame({"frame": tr.frame, "t": np.round(tr.t, 4), "gt": r["gt"], "pred": r["keys"]}).to_csv(
            out / "trials" / f"{tr.stem}_frames.csv", index=False)
        r["pred"].to_csv(out / "trials" / f"{tr.stem}_steps_pred.csv", index=False)
        r["oracle"].to_csv(out / "trials" / f"{tr.stem}_steps_oracle.csv", index=False)
        _plot_gantt(r["gt"], r["keys"], tr.t, classes, out / "trials" / f"{tr.stem}_gantt.png",
                    f"{tr.stem} · F1@50 {r['metrics']['f1@50']} · F1@IoU0.5 {r['metrics']['f1_iou05']}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_trial.csv", index=False)
    cm = confusion(all_gt, all_pred, classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_frames.csv")
    pcr = per_class_report(cm, classes)
    pcr.to_csv(out / "per_class.csv", index=False)
    _plot_confusion(cm, classes, out / "confusion_frames.png",
                    f"{mdir.name} · {len(trials)} trial test · frame-level (%, chuẩn hoá theo hàng)")
    s = summarize(rows)
    tot_f = sum(x["frames"] for x in times)
    summary = {"model": str(mdir), "checkpoint": a.checkpoint or "best",
               "decode": a.decode or cfg["hparams"].get("decode", "argmax"), "epoch": cfg.get("epoch", cfg.get("best_epoch")),
               "n_params": cfg.get("n_params"), "device": device, "trials": [t.stem for t in trials],
               "subjects": sorted({t.subject for t in trials}), "leak_trials": leak, "summary": s,
               "macro_f1_frames": round(float(pcr.loc[pcr.support_frames > 0, "f1"].mean()), 2),
               "inference": {"per_trial": times, "ms_per_video_mean": round(1000 * np.mean([x["infer_s"] for x in times]), 1),
                             "frames_per_s": round(tot_f / max(sum(x["infer_s"] for x in times), 1e-9), 1)},
               "cmd": " ".join(sys.argv)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(pcr.to_string(index=False))
    log("TEST: " + " · ".join(f"{k} {s[k]['mean']}±{s[k]['std']}" for k in FRAME_METRICS + STEP_METRICS))
    log(f"macro-F1 (frame) {summary['macro_f1_frames']} · suy luận {summary['inference']['ms_per_video_mean']} ms/video "
        f"({device}) → {out}")
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
        fdir = out / f"fold{i}"
        model, cfg, _ = train_model([t for t in rest if t.subject in tr_s], [t for t in rest if t.subject in va_s],
                                    hp, device, fdir if a.save_folds else None,
                                    {"test_subjects": list(test_subs)},
                                    ckpt_dir=(fdir / "checkpoints") if a.save_folds else None, tb_dir=fdir / "tb")
        for t in trials:
            if t.subject in test_subs:
                rows.append({"fold": i, **step_eval(model, cfg, t, a.data_root, station, device)})
    rep = {"folds": [list(f) for f in folds], "hparams": hp, "per_trial": rows, "summary": summarize(rows),
           "cmd": " ".join(sys.argv)}
    (out / "cv_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "cv_per_trial.csv", index=False)
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
    p = sub.add_parser("train", help="train 1 model (+ checkpoint theo epoch, TensorBoard) → models/segmenter/<name>/")
    _add_data(p)
    _add_hp(p)
    p.add_argument("--name", default="")
    p.add_argument("--val-subjects", default="", help="subject dùng làm val (mặc định chọn ngẫu nhiên theo --val-frac)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--val-trials", default="", help="val theo TRIAL (vd U001_V002) — chỉ để chạy thử, có rò rỉ theo người")
    p.add_argument("--test-subjects", default="", help="subject giữ riêng để test (loại hẳn khỏi train/val)")
    p.add_argument("--test-trials", default="", help="trial giữ riêng để test (vd U001_V003)")
    p.add_argument("--split-file", default="", help="data/splits/<name>.json từ `python -m kaizen.train.split make` "
                                                     "(thay cho --val-*/--test-*)")
    p.add_argument("--resume", action="store_true", help="train tiếp từ checkpoints/last.pt của --name")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("benchmark", help="đánh giá mọi checkpoint theo epoch trên tập val/test → bảng + biểu đồ")
    _add_data(p)
    p.add_argument("--model", default="latest")
    p.add_argument("--split", default="val", choices=["val", "test", "train"], help="tập lấy từ config.json của model")
    p.add_argument("--trials", default="", help="trial tường minh (thay cho --split)")
    p.add_argument("--tag", default="", help="hậu tố file khi dùng --subjects/--trials")
    p.add_argument("--no-tb", action="store_true", help="không ghi kết quả benchmark vào TensorBoard")
    p = sub.add_parser("test", help="suy luận + đánh giá trên tập test giữ riêng (dự đoán, confusion, Gantt)")
    _add_data(p)
    p.add_argument("--model", default="latest")
    p.add_argument("--checkpoint", default="best", help="best (model.pt) | epoch_0040 …")
    p.add_argument("--trials", default="", help="trial tường minh (mặc định: test_subjects/test_trials trong config)")
    p.add_argument("--out-dir", default="")
    p.add_argument("--decode", default="", choices=["", "argmax", "sop"],
                   help="giải mã: argmax | sop (ràng buộc thứ tự SOP); mặc định theo hparams của model")
    p = sub.add_parser("cv", help="cross-validation theo subject")
    _add_data(p)
    _add_hp(p)
    p.add_argument("--folds", type=int, default=5, help="số fold theo subject (0 = leave-one-subject-out)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--name", default="")
    p.add_argument("--save-folds", action="store_true", help="lưu model + checkpoint từng fold")
    p = sub.add_parser("eval", help="đánh giá nhanh model trên các trial có nhãn (chỉ in số)")
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
    return {"status": cmd_status, "train": cmd_train, "benchmark": cmd_benchmark, "test": cmd_test, "cv": cmd_cv,
            "eval": cmd_eval, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

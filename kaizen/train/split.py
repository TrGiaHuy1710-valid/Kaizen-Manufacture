"""Thống kê dataset HA4M + chia train/val/test → file split JSON dùng thẳng cho `segmenter train --split-file`.

    python -m kaizen.train.split stats                         # tình trạng từng video + zip → outputs/dataset/
    python -m kaizen.train.split make --name v1                # → data/splits/v1.json (strategy auto)
    python -m kaizen.train.split show data/splits/v1.json
    python -m kaizen.train.segmenter train --split-file data/splits/v1.json --name mstcn_v1

Trạng thái 1 video (trial):
    READY         có keypoints .npz + Labels.txt → train được ngay
    NEED_EXTRACT  có ảnh Color + nhãn, chưa có .npz → chạy src/01_extract.py
    NO_LABELS     không có Labels*.txt → chỉ suy luận được, không train
    NO_DATA       không có ảnh Color lẫn .npz
Zip trong data-root: INCOMPLETE (đang tải / hỏng — chưa đọc được central directory, hoặc dung lượng còn tăng),
EXTRACTED (đã có thư mục video), TO_PREPARE (zip đủ, chưa giải nén → src/tools/prepare_ha4m.py).

Chiến lược chia (`--strategy auto` chọn theo số subject / số video có nhãn):
    subject   ≥ 3 subject      — chia THEO NGƯỜI (đúng chuẩn báo cáo, không rò rỉ)
    trial     ≥ 2 video        — chia theo VIDEO; nếu cùng 1 người ở nhiều tập → rò rỉ theo người, ghi warning
    overfit   1 video          — train = val = video đó, không có test: CHỈ kiểm tra code train chạy thông
U001 (hoặc --pin-train) luôn ở train vì zone/ngưỡng của baseline rule (station.yaml) đã chỉnh trên U001.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from kaizen.core.config import DEFAULT_DATA_ROOT
from kaizen.skeleton import ha4m_io
from kaizen.train import dataset as ds

TAG = "split"
SPLITS_DIR = Path("data/splits")
STATS_DIR = Path("outputs/dataset")
FPS = 30.0
# 17 khoá bước.lần của HA4M (action 2, 3 lặp 3 lần; 11 lặp 2 lần) — MS-TCN học 17 lớp + nền "0"
EXPECTED_KEYS = (["1.0"] + [f"2.{r}" for r in range(3)] + [f"3.{r}" for r in range(3)]
                 + [f"{a}.0" for a in range(4, 11)] + ["11.0", "11.1", "12.0"])
WRIST_NAMES = ("l_wrist", "r_wrist")


def log(msg: str) -> None:
    print(f"[{TAG}] {msg}", flush=True)


# ---------------------------------------------------------------- thống kê
def _count_files(d: Path | None, exts: tuple[str, ...]) -> int:
    if d is None or not d.is_dir():
        return 0
    with os.scandir(d) as it:
        return sum(1 for e in it if e.is_file() and e.name.lower().endswith(exts))


def _modality_count(root: str, s: str, t: str, modality: str, exts: tuple[str, ...]) -> int:
    try:
        return _count_files(ha4m_io.modality_dir(root, s, t, modality), exts)
    except FileNotFoundError:
        return 0


def _npz_info(p: Path) -> dict:
    try:
        d = np.load(p, allow_pickle=False)
        k, names = d["kpts"], [str(n) for n in d["names"]]
        idx = [names.index(n) for n in WRIST_NAMES if n in names]
        valid = float(np.mean(np.any(np.nan_to_num(k[:, idx, 2]) >= 0.3, axis=1))) if idx else float("nan")
        return {"npz_frames": int(len(k)), "npz_wrist_valid": round(valid, 3)}
    except Exception as e:  # noqa: BLE001 — file hỏng vẫn phải báo được
        return {"npz_frames": 0, "npz_wrist_valid": float("nan"), "npz_error": str(e)[:80]}


def trial_stats(data_root: str, kp_dir: str | Path, pose_model: str = "mediapipe", fps: float = FPS) -> pd.DataFrame:
    """1 dòng / video: số frame từng nguồn, phủ nhãn, số bước thấy được, trạng thái train."""
    rows = []
    for s, t in ha4m_io.list_trials(data_root):
        stem = f"{s}_{t}"
        r: dict = {"stem": stem, "subject": s, "trial": t, "dir": str(ha4m_io.video_dir(data_root, s, t))}
        r["color_frames"] = _modality_count(data_root, s, t, "Color", ha4m_io.IMG_EXT)
        r["skeleton_files"] = _modality_count(data_root, s, t, "Skeletons", (".txt",))
        lp = ha4m_io.labels_path(data_root, s, t)
        r["has_labels"] = lp is not None
        warn: list[str] = []
        if lp is not None:
            fl = ha4m_io.load_frame_labels(data_root, s, t)
            act = fl[fl["action_id"] != 0]
            keys = {ds.class_key(a, rp) for a, rp in act[["action_id", "repetition"]].itertuples(index=False)}
            r.update({"label_frames": len(fl), "dont_care_pct": round(100 * (1 - len(act) / max(len(fl), 1)), 1),
                      "duration_s": round(len(fl) / fps, 1),
                      "cycle_s": round((act["frame"].max() - act["frame"].min() + 1) / fps, 1) if len(act) else 0.0,
                      "steps_found": f"{len(keys & set(EXPECTED_KEYS))}/{len(EXPECTED_KEYS)}"})
            miss = [k for k in EXPECTED_KEYS if k not in keys]
            if miss:
                warn.append(f"thiếu bước {','.join(miss)}")
            extra = sorted(keys - set(EXPECTED_KEYS))
            if extra:
                warn.append(f"khoá lạ {','.join(extra)}")
            if fl["frame"].diff().dropna().ne(1).any():
                warn.append("frame nhãn không liên tục")
            if r["color_frames"] and abs(r["color_frames"] - len(fl)) > 2:
                warn.append(f"ảnh Color {r['color_frames']} ≠ nhãn {len(fl)} frame (zip chưa giải nén đủ?)")
        npz = Path(kp_dir) / f"{stem}_{pose_model}.npz"
        r["has_npz"] = npz.exists()
        if npz.exists():
            r.update(_npz_info(npz))
            if r.get("label_frames") and abs(r["npz_frames"] - r["label_frames"]) > 2:
                warn.append(f".npz {r['npz_frames']} ≠ nhãn {r['label_frames']} frame")
            if r["npz_wrist_valid"] == r["npz_wrist_valid"] and r["npz_wrist_valid"] < 0.85:
                warn.append(f"cổ tay hợp lệ chỉ {100 * r['npz_wrist_valid']:.0f}% frame")
        if r["has_npz"] and r["has_labels"]:
            r["status"] = "READY"
        elif not r["has_labels"]:
            r["status"] = "NO_LABELS"
        elif r["color_frames"]:
            r["status"] = "NEED_EXTRACT"
        else:
            r["status"] = "NO_DATA"
        r["warnings"] = "; ".join(warn)
        rows.append(r)
    cols = ["stem", "subject", "trial", "status", "color_frames", "skeleton_files", "has_labels", "label_frames",
            "duration_s", "cycle_s", "dont_care_pct", "steps_found", "has_npz", "npz_frames", "npz_wrist_valid",
            "warnings", "dir"]
    df = pd.DataFrame(rows)
    return df.reindex(columns=cols) if len(df) else pd.DataFrame(columns=cols)


def zip_stats(data_root: str, patterns: str = "", known: set[str] | None = None, wait_s: float = 3.0) -> pd.DataFrame:
    """Zip HA4M trong data-root (+ glob --zips): còn đang tải không, đọc được không, đã giải nén chưa."""
    paths = {p for r in ha4m_io.roots_of(data_root) if r.is_dir() for p in r.glob("*.zip")}
    paths |= {Path(p) for pat in patterns.split(",") if pat.strip() for p in glob.glob(pat.strip())}
    paths = sorted(paths)
    if not paths:
        return pd.DataFrame(columns=["zip", "stem", "size_gb", "modified", "status", "note"])
    size0 = {p: p.stat().st_size for p in paths}
    if wait_s > 0:
        time.sleep(wait_s)
    rows = []
    for p in paths:
        st = p.stat()
        sub, tri = ha4m_io._ids_for(Path(p.stem))
        stem = f"{sub}_{tri}"
        r = {"zip": str(p), "stem": stem, "size_gb": round(st.st_size / 1e9, 2),
             "modified": dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"), "note": ""}
        if st.st_size != size0[p]:
            r["status"], r["note"] = "INCOMPLETE", f"đang tăng {(st.st_size - size0[p]) / 1e6 / wait_s:.0f} MB/s"
        else:
            try:
                with zipfile.ZipFile(p) as zf:
                    names = zf.namelist()
                has_lab = any(n.rsplit("/", 1)[-1].lower().startswith("labels") for n in names)
                n_color = sum(1 for n in names if "/color/" in n.lower() and n.lower().endswith(ha4m_io.IMG_EXT))
                r["note"] = f"{len(names)} file, {n_color} ảnh Color, nhãn: {'có' if has_lab else 'KHÔNG'}"
                r["status"] = "EXTRACTED" if known and stem in known else "TO_PREPARE"
            except (zipfile.BadZipFile, OSError) as e:
                r["status"], r["note"] = "INCOMPLETE", f"chưa đọc được zip ({type(e).__name__}) — tải chưa xong/hỏng"
        rows.append(r)
    return pd.DataFrame(rows)


def class_stats(data_root: str, stems: list[str], fps: float = FPS) -> pd.DataFrame:
    """Phân bố frame + thời lượng trung bình theo bước.lần trên các video có nhãn."""
    seg_rows, n_frames = [], {}
    for stem in stems:
        s, t = stem.split("_")
        fl = ha4m_io.load_frame_labels(data_root, s, t)
        for k, n in fl.apply(lambda x: ds.class_key(x.action_id, x.repetition), axis=1).value_counts().items():
            n_frames[k] = n_frames.get(k, 0) + int(n)
        seg = ha4m_io.labels_to_segments(fl)
        for x in seg.itertuples(index=False):
            seg_rows.append({"key": ds.class_key(x.action_id, x.repetition), "dur_s": (x.end_frame - x.start_frame + 1) / fps})
    seg = pd.DataFrame(seg_rows, columns=["key", "dur_s"])
    out = []
    total = sum(n_frames.values()) or 1
    for k in ["0"] + EXPECTED_KEYS:
        d = seg.loc[seg["key"] == k, "dur_s"]
        a, _ = ds.parse_key(k)
        out.append({"key": k, "action": ha4m_io.ACTION_NAMES.get(a, "?"), "frames": n_frames.get(k, 0),
                    "pct": round(100 * n_frames.get(k, 0) / total, 1), "n_segments": int(len(d)),
                    "mean_s": round(float(d.mean()), 2) if len(d) else None,
                    "std_s": round(float(d.std()), 2) if len(d) > 1 else None})
    return pd.DataFrame(out)


def recommend(n_subjects: int, n_trials: int) -> tuple[str, str]:
    """(strategy, giải thích) theo lượng dữ liệu có nhãn."""
    if n_trials == 0:
        return "none", "Chưa có video nào có nhãn → chưa train được."
    if n_trials == 1:
        return "overfit", ("Chỉ có 1 video (= 1 lần lắp, 1 chuỗi). KHÔNG chia train/val/test được: cắt 1 video theo thời gian "
                           "thì val/test chứa bước train chưa từng thấy (các bước nối tiếp nhau). Chỉ dùng để kiểm tra "
                           "code train chạy thông (overfit train=val). Cần thêm video.")
    if n_subjects >= 3:
        return "subject", (f"{n_subjects} subject → chia THEO NGƯỜI (train/val/test không chung người) — dùng được cho báo cáo.")
    return "trial", (f"{n_trials} video / {n_subjects} subject → chia THEO VIDEO. Số liệu chỉ phản ánh 'video mới' "
                     f"{'của cùng người (rò rỉ theo người)' if n_subjects == 1 else 'với rất ít người'}; "
                     "cần ≥ 3 subject (khuyến nghị ≥ 10) để kết quả có ý nghĩa.")


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(trống)_\n"
    d = df.fillna("").astype(str)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(v.replace("|", "/") for v in row) + " |" for row in d.values.tolist()]
    return "\n".join(lines) + "\n"


def cmd_stats(a) -> int:
    tr = trial_stats(a.data_root, a.keypoints_dir, a.pose_model, a.fps)
    zp = zip_stats(a.data_root, a.zips, set(tr["stem"]), 0 if a.no_wait else 3.0)
    labeled = tr[tr["has_labels"] == True]  # noqa: E712
    ready = tr[tr["status"] == "READY"]
    n_sub, n_tri = labeled["subject"].nunique(), len(labeled)
    strat, why = recommend(n_sub, n_tri)
    cls = class_stats(a.data_root, list(labeled["stem"]), a.fps) if n_tri else pd.DataFrame()
    per_sub = (tr.groupby("subject").agg(videos=("stem", "count"), labeled=("has_labels", "sum"),
                                         ready=("status", lambda x: int((x == "READY").sum())),
                                         minutes=("duration_s", lambda x: round(x.sum() / 60, 1))).reset_index()
               if len(tr) else pd.DataFrame())
    pending = zp[zp["status"] != "EXTRACTED"] if len(zp) else zp

    md = [f"# Thống kê dataset HA4M — {dt.datetime.now():%Y-%m-%d %H:%M}\n",
          f"- data_root: `{a.data_root}` · keypoints: `{a.keypoints_dir}` · pose: `{a.pose_model}` · fps {a.fps:g}",
          f"- Video tìm thấy: **{len(tr)}** · có nhãn: **{n_tri}** ({n_sub} subject) · "
          f"READY (có .npz + nhãn): **{len(ready)}**",
          f"- Tổng thời lượng có nhãn: **{labeled['duration_s'].sum() / 60:.1f} phút** "
          f"({int(labeled['label_frames'].sum()) if n_tri else 0} frame)",
          f"- Zip chưa giải nén / đang tải: **{len(pending)}**\n",
          f"## Kết luận: sẵn sàng train?\n\n- Chiến lược chia đề xuất: **`{strat}`** — {why}"]
    need = tr[tr["status"] == "NEED_EXTRACT"]
    if len(need):
        md.append(f"- {len(need)} video còn thiếu keypoints ({', '.join(need['stem'])}): "
                  f"`python src/01_extract.py --data-root {a.data_root} --subjects all --trials all --model {a.pose_model}`"
                  f" (~6 phút/video CPU)")
    inc = zp[zp["status"] == "INCOMPLETE"] if len(zp) else zp
    if len(inc):
        md.append(f"- Zip CHƯA tải xong — đừng giải nén: {', '.join(Path(z).name for z in inc['zip'])}")
    todo = zp[zp["status"] == "TO_PREPARE"] if len(zp) else zp
    if len(todo):
        md.append(f"- Zip đủ, chưa xử lý: `python src/tools/prepare_ha4m.py --zips \"{','.join(todo['zip'])}\" "
                  f"--data-root {a.data_root} --with-skeletons` (thêm `--keep-color` nếu muốn giữ ảnh)")
    bad = tr[tr["warnings"].fillna("") != ""]
    if len(bad):
        md.append(f"- Cảnh báo dữ liệu: {len(bad)} video (xem cột warnings)")
    md += ["\n## Theo video\n", _md_table(tr.drop(columns=["dir"])), "\n## Theo subject\n", _md_table(per_sub),
           "\n## Zip\n", _md_table(zp), "\n## Phân bố lớp (bước.lần) trên các video có nhãn\n", _md_table(cls)]
    text = "\n".join(md)
    print(text)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ha4m_stats.md").write_text(text, "utf-8")
    tr.to_csv(out / "ha4m_trials.csv", index=False)
    if len(cls):
        cls.to_csv(out / "ha4m_classes.csv", index=False)
    (out / "ha4m_stats.json").write_text(json.dumps(
        {"data_root": a.data_root, "n_videos": len(tr), "n_labeled": n_tri, "n_subjects": int(n_sub),
         "n_ready": len(ready), "recommended_strategy": strat, "reason": why,
         "trials": json.loads(tr.to_json(orient="records")), "zips": json.loads(zp.to_json(orient="records"))},
        ensure_ascii=False, indent=2), "utf-8")
    log(f"đã ghi {out}/ha4m_stats.md, ha4m_trials.csv, ha4m_classes.csv, ha4m_stats.json")
    return 0


# ---------------------------------------------------------------- chia split
def _take(items: list[str], pinned: set[str], n_val: int, n_test: int, seed: int,
          force_val: set[str], force_test: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Chia danh sách đơn vị (subject hoặc stem) → (train, val, test); pinned luôn train, force_* tường minh."""
    test = [x for x in items if x in force_test]
    val = [x for x in items if x in force_val and x not in test]
    free = [x for x in items if x not in pinned and x not in test and x not in val]
    random.Random(seed).shuffle(free)
    if not force_test:
        test, free = free[:n_test], free[n_test:]
    if not force_val:
        val, free = free[:n_val], free[n_val:]
    train = [x for x in items if x not in test and x not in val]
    return sorted(train), sorted(val), sorted(test)


def make_split(tr: pd.DataFrame, strategy: str = "auto", val_frac: float = 0.15, test_frac: float = 0.2,
               seed: int = 0, pin_train: set[str] | None = None, val: set[str] | None = None,
               test: set[str] | None = None) -> dict:
    """tr: bảng trial_stats (chỉ các dòng có nhãn). Trả dict split (stem từng tập + thông tin kiểm tra)."""
    tr = tr.sort_values("stem").reset_index(drop=True)
    stems, subjects = list(tr["stem"]), sorted(tr["subject"].unique())
    sub_of = dict(zip(tr["stem"], tr["subject"]))
    auto, why = recommend(len(subjects), len(stems))
    strategy = auto if strategy == "auto" else strategy
    pin_train, val, test = pin_train or set(), val or set(), test or set()
    warnings: list[str] = []
    if strategy == "none" or not stems:
        raise ValueError("không có video nào có nhãn để chia")
    if strategy == "overfit":
        keep = [s for s in stems if s in pin_train or sub_of[s] in pin_train] or stems[:1]
        tr_s, va_s, te_s = keep[:1], keep[:1], []
        warnings.append("OVERFIT: train = val = cùng 1 video, không có test — chỉ kiểm tra code, KHÔNG báo cáo số này")
    elif strategy == "subject":
        if len(subjects) < 2:
            raise ValueError("strategy=subject cần ≥ 2 subject")
        n = len(subjects)
        n_test = max(1, round(n * test_frac)) if test_frac > 0 and n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
        n_val = min(n_val, n - n_test - 1)
        pins = {s for s in subjects if s in pin_train}
        a, b, c = _take(subjects, pins, n_val, n_test, seed, {s for s in subjects if s in val},
                        {s for s in subjects if s in test})
        tr_s = [s for s in stems if sub_of[s] in a]
        va_s = [s for s in stems if sub_of[s] in b]
        te_s = [s for s in stems if sub_of[s] in c]
    elif strategy == "trial":
        n = len(stems)
        if n < 2:
            raise ValueError("strategy=trial cần ≥ 2 video")
        n_test = max(1, round(n * test_frac)) if test_frac > 0 and n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
        n_val = min(n_val, n - n_test - 1)
        pins = {s for s in stems if s in pin_train or sub_of[s] in pin_train}
        if len(pins) > n - n_val - n_test:      # vd chỉ có video của U001 → chỉ ghim video đầu
            pins = set(sorted(pins)[:max(1, n - n_val - n_test)])
        tr_s, va_s, te_s = _take(stems, pins, n_val, n_test, seed, {s for s in stems if s in val or sub_of[s] in val},
                                 {s for s in stems if s in test or sub_of[s] in test})
        if n_test == 0 and not test:
            warnings.append("chỉ có 2 video → không có tập test (val dùng tạm làm đánh giá); thêm video rồi chia lại")
    else:
        raise ValueError(f"strategy không hợp lệ: {strategy}")

    subs = {k: sorted({sub_of[s] for s in v}) for k, v in (("train", tr_s), ("val", va_s), ("test", te_s))}
    leak = sorted((set(subs["train"]) & (set(subs["val"]) | set(subs["test"]))) | (set(subs["val"]) & set(subs["test"])))
    if leak and strategy != "overfit":
        warnings.append(f"rò rỉ theo người: subject {leak} xuất hiện ở > 1 tập (chia theo video)")
    if not tr_s:
        raise ValueError("tập train rỗng")
    frames = dict(zip(tr["stem"], tr["label_frames"].fillna(0).astype(int)))
    split = {"strategy": strategy, "auto_reason": why, "seed": seed, "val_frac": val_frac, "test_frac": test_frac,
             "pin_train": sorted(pin_train), "train": tr_s, "val": va_s, "test": te_s,
             "subjects": subs, "leak_subjects": leak,
             "n_frames": {k: int(sum(frames[s] for s in v)) for k, v in (("train", tr_s), ("val", va_s), ("test", te_s))},
             "not_ready": sorted(tr.loc[tr["status"] != "READY", "stem"]), "warnings": warnings}
    return split


def _split_class_coverage(data_root: str, split: dict) -> list[str]:
    """Bước nào có trong val/test mà train không có (model không thể đoán đúng lớp chưa học)."""
    def keys(stems):
        out = set()
        for stem in stems:
            s, t = stem.split("_")
            fl = ha4m_io.load_frame_labels(data_root, s, t)
            out |= {ds.class_key(a, r) for a, r in fl[["action_id", "repetition"]].itertuples(index=False)}
        return out
    k_tr = keys(split["train"])
    warn = []
    for name in ("val", "test"):
        miss = sorted(keys(split[name]) - k_tr, key=ds.parse_key)
        if miss:
            warn.append(f"{name} có lớp {miss} không xuất hiện trong train")
    return warn


def load_split(path: str | Path) -> dict:
    sp = json.loads(Path(path).read_text("utf-8"))
    for k in ("train", "val", "test"):
        sp.setdefault(k, [])
    return sp


def print_split(sp: dict) -> None:
    print(f"split '{sp.get('name', '?')}' · strategy {sp['strategy']} · seed {sp.get('seed')}")
    for k in ("train", "val", "test"):
        print(f"  {k:5s} {len(sp[k]):3d} video · {sp.get('n_frames', {}).get(k, 0):7d} frame · "
              f"subject {','.join(sp.get('subjects', {}).get(k, [])) or '—'} · {', '.join(sp[k]) or '—'}")
    for w in sp.get("warnings", []):
        print(f"  ⚠ {w}")
    if sp.get("not_ready"):
        print(f"  ⚠ chưa có keypoints .npz (trích trước khi train): {', '.join(sp['not_ready'])}")


def _set(s: str) -> set[str]:
    return {x.strip().upper() for x in s.split(",") if x.strip()}


def cmd_make(a) -> int:
    tr = trial_stats(a.data_root, a.keypoints_dir, a.pose_model, a.fps)
    tr = tr[tr["has_labels"] == True]  # noqa: E712
    if a.only_ready:
        tr = tr[tr["status"] == "READY"]
    if a.subjects:
        tr = tr[tr["subject"].isin(_set(a.subjects))]
    if tr.empty:
        log("không có video có nhãn nào (xem: python -m kaizen.train.split stats)")
        return 1
    out = Path(a.out) if a.out else SPLITS_DIR / f"{a.name}.json"
    if out.exists() and not a.force:
        log(f"{out} đã có — split phải CỐ ĐỊNH trước khi xem kết quả test. Đổi --name hoặc --force.")
        return 1
    try:
        sp = make_split(tr, a.strategy, a.val_frac, a.test_frac, a.seed, _set(a.pin_train), _set(a.val), _set(a.test))
    except ValueError as e:
        log(f"✗ {e}")
        return 1
    sp["warnings"] += _split_class_coverage(a.data_root, sp)
    sp = {"name": out.stem, "created": dt.datetime.now().isoformat(timespec="seconds"), "data_root": a.data_root,
          "keypoints_dir": a.keypoints_dir, "pose_model": a.pose_model, **sp, "cmd": " ".join(sys.argv)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sp, ensure_ascii=False, indent=2), "utf-8")
    print_split(sp)
    log(f"đã ghi {out}")
    if sp["not_ready"]:
        log(f"trích keypoints: python src/01_extract.py --data-root {a.data_root} --subjects all --trials all "
            f"--model {a.pose_model}")
    log(f"train: python -m kaizen.train.segmenter train --split-file {out} --name mstcn_{out.stem}")
    return 0


def cmd_show(a) -> int:
    print_split(load_split(a.path))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data(p):
        p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="thư mục HA4M (nhiều thư mục: a,b)")
        p.add_argument("--keypoints-dir", default="data/keypoints")
        p.add_argument("--pose-model", default="mediapipe")
        p.add_argument("--fps", type=float, default=FPS)

    p = sub.add_parser("stats", help="thống kê dữ liệu + zip, đánh giá sẵn sàng train")
    data(p)
    p.add_argument("--zips", default="", help="glob zip thêm ngoài data-root (nhiều mẫu: a,b)")
    p.add_argument("--out-dir", default=str(STATS_DIR))
    p.add_argument("--no-wait", action="store_true", help="không chờ 3 s để kiểm tra zip còn đang tải")
    p = sub.add_parser("make", help="chia train/val/test → data/splits/<name>.json")
    data(p)
    p.add_argument("--name", default="v1")
    p.add_argument("--out", default="")
    p.add_argument("--strategy", default="auto", choices=["auto", "subject", "trial", "overfit"])
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pin-train", default="U001", help="subject/video luôn ở train (station.yaml chỉnh trên U001)")
    p.add_argument("--val", default="", help="subject hoặc video ép vào val (vd U008 hoặc U001_V002)")
    p.add_argument("--test", default="", help="subject hoặc video ép vào test")
    p.add_argument("--subjects", default="", help="chỉ dùng các subject này")
    p.add_argument("--only-ready", action="store_true", help="chỉ lấy video đã có .npz")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("show", help="in 1 file split")
    p.add_argument("path")
    a = ap.parse_args(argv)
    return {"stats": cmd_stats, "make": cmd_make, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

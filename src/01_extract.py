"""[01] Batch trích keypoints cho các video HA4M → data/keypoints/<subject>_<trial>_<model>.npz.

Tự quét --data-root (1 hoặc nhiều thư mục, cách nhau dấu phẩy) bằng ha4m_io.list_trials (quét phòng thủ:
Color* không phân biệt hoa thường, tên thư mục lệch nhẹ, Labels*.txt). Video thiếu nhãn vẫn được trích,
đánh dấu labels_available=false trong data/keypoints/manifest.json.

    python src/01_extract.py --data-root data/ha4m_new --subjects all --limit 1
    python src/01_extract.py --subjects U001,U002 --trials all --model mediapipe [--preview] [--force]
fps và ROI lấy từ config/station.yaml (fps, extract_roi). Mặc định bỏ qua trial đã có .npz (--force để chạy lại).
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, base_parser, load_config, select_stems, setup, stem  # noqa: E402
import ha4m_io  # noqa: E402

TAG = "01"
SUSPECT_DETECT = 0.85


def npz_meta(p: Path) -> dict:
    """meta + số frame của 1 .npz (đọc lười, không nạp kpts)."""
    try:
        with np.load(p, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
            meta["n_frames"] = int(d["t"].shape[0])
        return meta
    except Exception as e:  # file hỏng
        return {"error": str(e)}


def limit_per_subject(stems: list[str], n: int) -> list[str]:
    if n <= 0:
        return stems
    out, cnt = [], {}
    for st in stems:
        sub = st.split("_")[0]
        if cnt.get(sub, 0) < n:
            out.append(st)
            cnt[sub] = cnt.get(sub, 0) + 1
    return out


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--model", default="mediapipe", choices=["mediapipe", "mediapipe-heavy", "rtmw"])
    ap.add_argument("--out", default="data/keypoints")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N trial đầu mỗi subject (0 = tất cả)")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                    help="bỏ qua trial đã có .npz cùng model (mặc định bật)")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="chỉ để thử nhanh (npz sẽ thiếu frame)")
    a = ap.parse_args()
    setup(a, TAG, need_run=False)
    cfg = load_config(a.config)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = ha4m_io.list_trials(a.data_root)
    stems = limit_per_subject(select_stems([stem(s, t) for s, t in trials], a.subjects, a.trials), a.limit)
    if not stems:
        print(f"[{TAG}] không có trial nào khớp trong {a.data_root} (subjects={a.subjects or 'all'}, "
              f"trials={a.trials or 'all'})", file=sys.stderr)
        sys.exit(1)
    print(f"[{TAG}] {len(stems)} trial trong {a.data_root}: {', '.join(stems[:10])}{' ...' if len(stems) > 10 else ''}")

    man_path = out_dir / "manifest.json"
    try:
        kp_man = json.loads(man_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        kp_man = {}
    roi = cfg.get("extract_roi")
    rows, failed = [], []
    print(f"[{TAG}] | trial | trạng thái | số frame | proc_fps | detect_rate | nhãn |")
    for st in stems:
        sub, tri = st.split("_")
        npz = out_dir / f"{st}_{a.model}.npz"
        has_lab = ha4m_io.labels_path(a.data_root, sub, tri) is not None
        if not has_lab:
            print(f"[{TAG}] cảnh báo: {st} không có Labels*.txt → vẫn trích keypoints, labels_available=false",
                  file=sys.stderr)
        status, secs = "skip (đã có)", 0.0
        if not npz.exists() or a.force or not a.skip_existing:
            try:
                frames_dir = ha4m_io.modality_dir(a.data_root, sub, tri, "Color")
            except FileNotFoundError as e:
                print(f"[{TAG}] {st}: {e}", file=sys.stderr)
                failed.append(st)
                continue
            cmd = [sys.executable, str(ROOT / "extract_keypoints.py"), "--frames-dir", str(frames_dir),
                   "--fps", str(cfg["fps"]), "--name", st, "--model", a.model, "--out", str(out_dir),
                   "--stride", str(a.stride), "--max-frames", str(a.max_frames), "--force"]
            if roi:
                cmd += ["--roi", ",".join(str(int(v)) for v in roi)]
            if a.preview:
                cmd.append("--preview")
            print(f"[{TAG}] {st}: trích từ {frames_dir} ...", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd)
            secs = time.time() - t0
            if r.returncode != 0 or not npz.exists():
                print(f"[{TAG}] {st}: extract_keypoints lỗi (exit {r.returncode})", file=sys.stderr)
                failed.append(st)
                continue
            status = f"đã trích ({secs:.0f}s)"
        meta = npz_meta(npz)
        row = dict(trial=st, status=status, n_frames=meta.get("n_frames"), proc_fps=meta.get("proc_fps"),
                   detect_rate=meta.get("detect_rate"), labels=has_lab)
        rows.append(row)
        kp_man[f"{st}_{a.model}"] = {
            "trial": st, "model": a.model, "data_root": a.data_root,
            "video_dir": str(ha4m_io.video_dir(a.data_root, sub, tri)), "n_frames": meta.get("n_frames"),
            "proc_fps": meta.get("proc_fps"), "detect_rate": meta.get("detect_rate"),
            "labels_available": has_lab, "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            **({"partial_max_frames": a.max_frames} if a.max_frames and "đã trích" in status else {})}
        man_path.write_text(json.dumps(kp_man, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{TAG}] | {st} | {status} | {row['n_frames']} | {row['proc_fps']} | {row['detect_rate']} | "
              f"{'có' if has_lab else 'KHÔNG'} |", flush=True)

    # Tổng hợp
    print(f"\n[{TAG}] ===== Tổng hợp: {len(rows)} trial OK, {len(failed)} lỗi =====")
    print(f"[{TAG}] | trial | trạng thái | số frame | proc_fps | detect_rate | nhãn |")
    print(f"[{TAG}] |---|---|---|---|---|---|")
    for rw in rows:
        print(f"[{TAG}] | {rw['trial']} | {rw['status']} | {rw['n_frames']} | {rw['proc_fps']} | "
              f"{rw['detect_rate']} | {'có' if rw['labels'] else 'KHÔNG'} |")
    sus = [rw["trial"] for rw in rows if rw["detect_rate"] is not None and rw["detect_rate"] < SUSPECT_DETECT]
    if sus:
        print(f"[{TAG}] ⚠ detect_rate < {SUSPECT_DETECT}: {sus} — xem *_preview.mp4 (chạy lại với --preview --force)")
    nolab = [rw["trial"] for rw in rows if not rw["labels"]]
    if nolab:
        print(f"[{TAG}] ⚠ không có nhãn (oracle/eval sẽ bỏ qua): {nolab}")
    if failed:
        print(f"[{TAG}] ✗ lỗi: {failed}", file=sys.stderr)
    subs = sorted({rw["trial"].split("_")[0] for rw in rows})
    if rows:
        print(f"[{TAG}] Gợi ý chạy tiếp:\n    bash run_all.sh --data-root {a.data_root} --subjects {','.join(subs)} "
              f"--model {a.model} [--oracle] [--dry-run | --provider gemini]")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

"""Test toàn luồng với dữ liệu TỔNG HỢP (không cần dataset HA4M, không cần mạng).

Sinh trong thư mục tạm: cấu trúc HA4M giả (IDU900V001..V004, IDU901V001) với Labels.txt, vài ảnh Color,
1 file Skeleton; keypoints .npz hình sin theo nhãn (tay phải đi tới khay của bước → dừng → về vùng lắp).
Cài sẵn bất thường: V002 bước 10 chậm gấp 3 (R1), V003 lấy dư 1 lần ở khay bước 4 (R2),
V004 tay phải đứng yên ngoài zone 45s (R3, và CV cycle time > 0.25 → R6). Chạy 00, 02→08 (07 --dry-run) bằng subprocess rồi assert.

    python tests/synthetic.py            # in OK/FAIL, exit code ≠ 0 nếu lỗi
    python tests/synthetic.py --keep     # giữ thư mục tạm để xem report
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

FPS = 30
W, H = 320, 240
ASM = (160.0, 90.0)
BINS = {f"bin_{i}": (30 + 52 * (i - 1) + 20, 190.0) for i in range(1, 7)}   # tâm khay
REPS = {2: 3, 3: 3, 11: 2}
STEP_ZONE = {s: (None if s == 9 else f"bin_{(s - 1) % 6 + 1}") for s in range(1, 13)}


def station_cfg() -> dict:
    zones = {n: [[x - 20, y - 20], [x + 20, y - 20], [x + 20, y + 20], [x - 20, y + 20]] for n, (x, y) in BINS.items()}
    zones["assembly_area"] = [[120, 55], [200, 55], [200, 125], [120, 125]]
    steps = [{"id": s, "name": f"Step {s}", "zone": STEP_ZONE[s], "reps": REPS.get(s, 1), "va": s not in (1, 11)}
             for s in range(1, 13)]
    return {"fps": FPS, "px_per_cm": None, "moving_speed_px_s": 120, "min_event_s": 0.25, "idle_threshold_s": 3.0,
            "idle_still_s": 1.5, "pick_max_still_s": 2.0, "zones": zones, "steps": steps,
            "step_min_gap_s": 1.0, "muri_dev_deg": 30, "cycle_time_target_s": 60}


def seg_plan(variant: str, rng) -> list[tuple[int, int, int, dict]]:
    """[(action_id, rep, n_frames, extra)] — extra: {'extra_pick': zone} | {'idle_s': x}."""
    plan = [(0, 0, 20, {})]
    for s in range(1, 13):
        for rep in range(REPS.get(s, 1)):
            n = int(FPS * (3.0 + rng.uniform(-0.3, 0.3)))
            ex = {}
            if variant == "slow" and s == 10:
                n *= 3
            if variant == "extra_pick" and s == 4:
                n += FPS * 2
                ex = {"extra_pick": True}
            if variant == "idle" and s == 8:
                n += FPS * 45
                ex = {"idle_s": 45}
            plan.append((s, rep, n, ex))
    plan.append((0, 0, 20, {}))
    return plan


def move(a, b, n):
    """Nửa sóng cos từ a → b trong n frame."""
    u = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    return np.outer(1 - u, a) + np.outer(u, b)


def right_traj(action: int, n: int, ex: dict) -> np.ndarray:
    if action == 0:
        return np.tile(ASM, (n, 1))
    zone = STEP_ZONE[action]
    tgt = np.array(BINS[zone] if zone else (175.0, 140.0)) + [0, -5]
    parts = [move(ASM, tgt, 18), np.tile(tgt, (12, 1)), move(tgt, ASM, 18)]
    if ex.get("extra_pick"):
        parts += [move(ASM, tgt, 18), np.tile(tgt, (12, 1)), move(tgt, ASM, 18)]
    if ex.get("idle_s"):
        far = np.array([285.0, 40.0])
        k = int(ex["idle_s"] * FPS)
        parts += [move(ASM, far, 15), np.tile(far, (k, 1)), move(far, ASM, 15)]
    used = sum(len(p) for p in parts)
    rest = max(n - used, 1)
    tt = np.arange(rest) / FPS
    parts.append(np.stack([ASM[0] + 3 * np.sin(2 * np.pi * tt), ASM[1] + 2 * np.cos(2 * np.pi * tt)], 1))
    return np.concatenate(parts)[:n] if used < n else np.concatenate(parts)


def make_trial(root: Path, kp_dir: Path, sub: str, tri: str, variant: str, rng) -> None:
    import extract_keypoints as ek
    plan = seg_plan(variant, rng)
    labels, rw = [], []
    for a, rep, n, ex in plan:
        tr = right_traj(a, n, ex)
        rw.append(tr)
        labels += [(a, rep)] * len(tr)
    rw = np.concatenate(rw)
    T = len(rw)
    t = np.arange(T) / FPS
    lw = np.stack([140 + 4 * np.sin(2 * np.pi * 0.5 * t), 95 + 3 * np.cos(2 * np.pi * 0.5 * t)], 1)
    names = ek.MP_NAMES
    k = np.full((T, len(names), 3), np.nan, np.float32)

    def put(name, xy, s=0.9):
        i = names.index(name)
        k[:, i, :2], k[:, i, 2] = xy, s
    put("l_wrist", lw); put("r_wrist", rw)
    put("l_elbow", lw + [-10, 35]); put("r_elbow", rw + [10, 35])
    put("lh_middle_1", lw + [-2, -15] + np.stack([12 * np.sin(t), np.zeros(T)], 1))
    put("rh_middle_1", rw + [2, -15] + np.stack([12 * np.sin(1.3 * t), np.zeros(T)], 1))
    np.savez_compressed(kp_dir / f"{sub}_{tri}_mediapipe.npz", kpts=k, t=t.astype(np.float32),
                        frame=np.arange(T, dtype=np.int32), names=np.array(names),
                        meta=json.dumps({"model": "synthetic", "src_fps": FPS}))

    vd = root / f"ID{sub}{tri}"
    (vd / "Color" / "000000000000").mkdir(parents=True)
    (vd / "Skeletons" / "000000000000").mkdir(parents=True)
    with open(vd / "Labels.txt", "w", newline="") as f:
        for i, (a, rep) in enumerate(labels):
            f.write(f"{i} {a} {rep}\r\n")
    ts0 = 1_000_000
    for i in range(0, T, 10):   # chỉ ghi 1/10 ảnh cho nhanh (08 bỏ qua frame không có ảnh)
        img = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(img, (int(rw[i, 0]), int(rw[i, 1])), 6, (0, 200, 255), -1)
        cv2.imwrite(str(vd / "Color" / "000000000000" / f"FrameID{i:06d}_DeviceTimeStamp{ts0 + i * 33333}us.png"), img)
    hdr = "BodyID\tJointID\tConfidance\tX\tY\tZ\tQw\tQx\tQy\tQz\tX2DColor\tY2DColor\tX2DDepth\tY2DDepth\n"
    rows = "".join(f"1\t{j}\t0.666667\t0\t0\t1000\t1\t0\t0\t0\t{rw[0, 0]:.1f}\t{rw[0, 1]:.1f}\t0\t0\n" for j in range(32))
    (vd / "Skeletons" / "000000000000" / f"FrameID000000_DeviceTimeStamp{ts0}us.txt").write_text(hdr + rows)


def run(tmp: Path, *args: str) -> str:
    r = subprocess.run([sys.executable, *args], cwd=tmp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main() -> int:
    keep = "--keep" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="kaizen_synth_"))
    rng = np.random.default_rng(0)
    root, kp = tmp / "data" / "ha4m", tmp / "data" / "keypoints"
    root.mkdir(parents=True); kp.mkdir(parents=True)
    (tmp / "config").mkdir()
    for f in ("kaizen_kb.md", "sop.md"):
        shutil.copy(REPO / "config" / f, tmp / "config" / f)
    (tmp / "config" / "station.yaml").write_text(yaml.safe_dump(station_cfg(), sort_keys=False))
    for tri, var in (("V001", "normal"), ("V002", "slow"), ("V003", "extra_pick"), ("V004", "idle")):
        make_trial(root, kp, "U900", tri, var, rng)
    make_trial(root, kp, "U901", "V001", "normal", rng)

    S = lambda n: str(REPO / "src" / n)  # noqa: E731
    dr = ["--data-root", str(root)]
    run(tmp, S("00_inspect.py"), *dr)
    run(tmp, S("02_features.py"), *dr)
    run(tmp, S("03_events.py"), *dr, "--print-timeline")
    run(tmp, S("04_steps.py"), *dr, "--oracle")
    run(tmp, S("04_steps.py"), *dr)
    run(tmp, S("05_metrics.py"), *dr, "--source", "oracle")
    run(tmp, S("06_findings.py"))
    run(tmp, S("07_explain.py"), "--dry-run")
    run(tmp, S("08_report.py"), *dr)

    checks = []
    ok = lambda name, cond: checks.append((name, bool(cond)))  # noqa: E731
    out = tmp / "outputs"
    ok("inspect report", (out / "inspect_report.md").exists())
    for st in ("U900_V001", "U900_V002", "U900_V003", "U900_V004", "U901_V001"):
        ok(f"features {st}", (out / "features" / f"{st}.parquet").exists())
        ok(f"events {st}", (out / "events" / f"{st}.csv").exists())
    ev = json.loads((out / "steps" / "U900_V001_eval.json").read_text())
    ok(f"predicted F1@0.5 ≥ 0.5 (={ev['f1_iou05']})", ev["f1_iou05"] >= 0.5)
    ok(f"cycle time err < 2s (={ev['cycle_time_err_s']})", abs(ev["cycle_time_err_s"]) < 2)
    fd = json.loads((out / "findings" / "findings.json").read_text())
    rules = {(f["rule"], tuple(f["evidence"]["trials"])) for f in fd["findings"]}
    ok("R1 bước chậm ở V002", any(r == "R1" and "U900_V002" in t for r, t in rules))
    ok("R2 pick dư ở V003", any(r == "R2" and "U900_V003" in t for r, t in rules))
    ok("R3 idle dài ở V004", any(r == "R3" and "U900_V004" in t for r, t in rules))
    ok("R6 CV cao subject U900", any(r == "R6" for r, _ in rules))
    ok("không có R3 ở V001", not any(r == "R3" and "U900_V001" in t for r, t in rules))
    scores = [f["rank_score"] for f in fd["findings"]]
    ok("findings xếp hạng giảm dần", scores == sorted(scores, reverse=True))
    ex = json.loads((out / "findings" / "explained.json").read_text())
    ok(f"dry-run pass rate = 1 (={ex['pass_rate']})", ex["pass_rate"] == 1.0)
    ok("report.md", (out / "report.md").exists() and "Giả định & giới hạn" in (out / "report.md").read_text())

    # Validator phải bắt số bịa và citation bịa
    m7 = importlib.import_module("07_explain")
    ids = set(m7.load_ids(tmp / "config" / "kaizen_kb.md", tmp / "config" / "sop.md"))
    f0 = fd["findings"][0]
    good = {"observation": f"Giá trị {f0['evidence']['value']} {f0['evidence']['unit']}.", "hypothesis": "", "citations": ["KB-01"]}
    bad = {"observation": "Chậm 999.9 s so với chuẩn.", "hypothesis": "", "citations": ["KB-77"]}
    ok("validator pass câu đúng", m7.validate(good, f0, ids)["status"] == "passed")
    v = m7.validate(bad, f0, ids)
    ok("validator bắt số bịa + citation bịa", v["status"] == "failed" and v["bad_citations"] == ["KB-77"]
       and "999.9" in v["unsupported_numbers"])

    width = max(len(n) for n, _ in checks)
    for n, c in checks:
        print(f"{'OK  ' if c else 'FAIL'} {n:<{width}}")
    n_fail = sum(not c for _, c in checks)
    print(f"\n{len(checks) - n_fail}/{len(checks)} check đạt. Thư mục: {tmp}")
    if not keep and n_fail == 0:
        shutil.rmtree(tmp)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

"""[01] Wrapper: gọi extract_keypoints.py cho N trial HA4M → data/keypoints/<subject>_<trial>_<model>.npz.

    python src/01_extract.py --subjects U001 --model mediapipe [--preview] [--force]
fps và ROI lấy từ config/station.yaml (fps, extract_roi).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, base_parser, load_config, select_stems, stem  # noqa: E402
import ha4m_io  # noqa: E402

TAG = "01"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--model", default="mediapipe", choices=["mediapipe", "mediapipe-heavy", "rtmw"])
    ap.add_argument("--out", default="data/keypoints")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)

    trials = ha4m_io.list_trials(a.data_root)
    stems = select_stems([stem(s, t) for s, t in trials], a.subjects, a.trials)
    if not stems:
        sys.exit(f"[{TAG}] không có trial nào khớp trong {a.data_root}")
    roi = cfg.get("extract_roi")
    for st in stems:
        sub, tri = st.split("_")
        out = Path(a.out) / f"{st}_{a.model}.npz"
        if out.exists() and not a.force:
            print(f"[{TAG}] đã có {out}, bỏ qua (--force để chạy lại)")
            continue
        cmd = [sys.executable, str(ROOT / "extract_keypoints.py"),
               "--frames-dir", str(ha4m_io.modality_dir(a.data_root, sub, tri, "Color")),
               "--fps", str(cfg["fps"]), "--name", st, "--model", a.model, "--out", a.out,
               "--stride", str(a.stride), "--max-frames", str(a.max_frames), "--force"]
        if roi:
            cmd += ["--roi", ",".join(str(int(v)) for v in roi)]
        if a.preview:
            cmd.append("--preview")
        print(f"[{TAG}] {st}: {' '.join(cmd)}")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"[{TAG}] {st} xong trong {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

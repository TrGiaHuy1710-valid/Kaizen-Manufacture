"""Chuẩn bị dữ liệu train từ các file zip HA4M (IDU001V001.zip …) khi ĐĨA ÍT CHỖ.

Mỗi zip ~12 GB (giải nén đủ ~17 GB: Color, Depth, PointClouds…), nhưng train MS-TCN chỉ cần keypoints .npz (~1.3 MB)
+ Labels.txt. Với từng zip, lần lượt (tuần tự, đỉnh dung lượng ≈ 1 thư mục Color ~7.6 GB):
    1) giải nén CHỈ <ID>/Color/** + <ID>/Labels*.txt (+ Skeletons nếu --with-skeletons) vào --data-root
    2) chạy src/01_extract.py cho đúng trial đó → data/keypoints/<U>_<V>_<model>.npz (+ manifest.json)
    3) xoá ảnh Color (giữ thư mục rỗng để ha4m_io vẫn nhận trial + Labels.txt), tuỳ chọn xoá zip
Trial đã có .npz + nhãn → bỏ qua (chạy lại an toàn sau khi bị ngắt).

    python src/tools/prepare_ha4m.py --zips "/path/to/zips/IDU00*.zip" --data-root data/ha4m
    python src/tools/prepare_ha4m.py --zips "/path/IDU009*.zip,/path/IDU010*.zip" --keep-color   # giữ ảnh (để chạy run_all/overlay)
    python src/tools/prepare_ha4m.py --zips "..." --dry-run                                     # chỉ liệt kê
"""
from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kaizen.skeleton import ha4m_io  # noqa: E402

TAG = "prep"
MARGIN_GB = 2.0


def log(msg: str) -> None:
    print(f"[{TAG}] {msg}", flush=True)


def _top(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if "/" in n.strip("/")]
    tops = {n.split("/")[0] for n in names}
    if len(tops) != 1:
        raise ValueError(f"zip có {len(tops)} thư mục gốc ({sorted(tops)[:3]}…) — cần đúng 1 (vd IDU001V001/)")
    return tops.pop()


def _wanted(name: str, top: str, with_skel: bool) -> bool:
    rel = name[len(top) + 1:]
    head = rel.split("/")[0].lower()
    if "/" not in rel:
        return head.startswith("labels") and head.endswith(".txt")
    return head == "color" or (with_skel and head == "skeletons")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", required=True, help="glob, nhiều mẫu cách nhau dấu phẩy")
    ap.add_argument("--data-root", default="data/ha4m")
    ap.add_argument("--keypoints-dir", default="data/keypoints")
    ap.add_argument("--model", default="mediapipe")
    ap.add_argument("--keep-color", action="store_true", help="không xoá ảnh Color sau khi trích")
    ap.add_argument("--with-skeletons", action="store_true", help="giải nén thêm Skeletons (đánh giá pose vs Kinect)")
    ap.add_argument("--delete-zip", action="store_true", help="xoá file zip sau khi trích thành công")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    zips = sorted({p for pat in a.zips.split(",") if pat.strip() for p in glob.glob(pat.strip())})
    if not zips:
        log(f"không thấy zip nào khớp {a.zips}")
        return 1
    root, kp_dir = Path(a.data_root), Path(a.keypoints_dir)
    root.mkdir(parents=True, exist_ok=True)
    log(f"{len(zips)} zip → {root} · keypoints {kp_dir} · model {a.model}")
    ok, skipped, failed = [], [], []
    t_all = time.time()
    for i, zp in enumerate(zips, 1):
        try:
            zf = zipfile.ZipFile(zp)
            top = _top(zf)
        except (zipfile.BadZipFile, ValueError) as e:
            log(f"[{i}/{len(zips)}] {zp}: lỗi zip — {e}")
            failed.append(zp)
            continue
        sub, tri = ha4m_io._ids_for(Path(top))
        st = f"{sub}_{tri}"
        npz = kp_dir / f"{st}_{a.model}.npz"
        vdir = root / top
        has_lab = any(p.name.lower().startswith("labels") for p in vdir.glob("*.txt")) if vdir.exists() else False
        if npz.exists() and has_lab:
            log(f"[{i}/{len(zips)}] {st}: đã có {npz.name} + nhãn → bỏ qua")
            skipped.append(st)
            continue
        members = [m for m in zf.infolist() if not m.is_dir() and _wanted(m.filename, top, a.with_skeletons)]
        need_gb = sum(m.file_size for m in members) / 1e9
        free_gb = shutil.disk_usage(root).free / 1e9
        log(f"[{i}/{len(zips)}] {st}: {len(members)} file, {need_gb:.1f} GB cần giải nén · trống {free_gb:.1f} GB")
        if a.dry_run:
            continue
        if need_gb + MARGIN_GB > free_gb:
            log(f"✗ không đủ chỗ trống cho {st} (cần {need_gb + MARGIN_GB:.1f} GB) — dừng")
            failed.append(st)
            break
        t0 = time.time()
        if not npz.exists():
            for m in members:
                zf.extract(m, root)
        else:   # đã có keypoints, chỉ thiếu nhãn
            for m in members:
                if "/" not in m.filename[len(top) + 1:]:
                    zf.extract(m, root)
        zf.close()
        log(f"  giải nén {time.time() - t0:.0f}s")
        ha4m_io._SCAN_CACHE.clear()
        if not npz.exists():
            t1 = time.time()
            r = subprocess.run([sys.executable, str(ROOT / "src" / "01_extract.py"), "--data-root", str(root),
                                "--subjects", sub, "--trials", tri, "--model", a.model, "--out", str(kp_dir)])
            if r.returncode != 0 or not npz.exists():
                log(f"✗ {st}: 01_extract lỗi (exit {r.returncode}) — giữ nguyên ảnh để kiểm tra")
                failed.append(st)
                continue
            log(f"  trích keypoints {time.time() - t1:.0f}s")
        if not a.keep_color:
            color = ha4m_io._find_child(vdir, "Color")
            n = 0
            if color is not None:
                for p in color.rglob("*"):
                    if p.is_file():
                        p.unlink()
                        n += 1
            log(f"  xoá {n} ảnh Color (giữ thư mục + Labels.txt)")
        if a.delete_zip:
            Path(zp).unlink()
            log(f"  xoá {zp}")
        ok.append(st)
    log(f"xong {len(ok)} trial, bỏ qua {len(skipped)}, lỗi {len(failed)} · {(time.time() - t_all) / 60:.1f} phút")
    if failed:
        log(f"lỗi: {failed}")
    log("kiểm tra: python -m kaizen.train.segmenter status")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

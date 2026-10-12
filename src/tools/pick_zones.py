"""Vẽ polygon zone trên 1 frame HA4M bằng chuột → in YAML ra stdout.

Chạy local (cần màn hình):
    python src/tools/pick_zones.py --subject U001 --trial V001 --frame 1700
    Click trái: thêm đỉnh | Enter/n: đóng polygon, nhập tên zone ở terminal | u: xoá đỉnh cuối | q: xong
Không có màn hình (server): vẽ các zone hiện có trong station.yaml ra PNG để kiểm tra:
    python src/tools/pick_zones.py --subject U001 --trial V001 --frame 1700 --render outputs/zones.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ha4m_io  # noqa: E402

PALETTE = [(46, 134, 171), (209, 73, 91), (237, 174, 73), (0, 121, 140), (102, 46, 155),
           (60, 179, 113), (255, 127, 80), (106, 90, 205), (210, 105, 30), (70, 130, 180)]


def draw_zones(img: np.ndarray, zones: dict, thick: int = 3) -> np.ndarray:
    """Vẽ polygon + tên zone lên ảnh (tại chỗ)."""
    for i, (name, pts) in enumerate(zones.items()):
        if not pts:
            continue
        c = PALETTE[i % len(PALETTE)]
        p = np.array(pts, np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [p], c)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [p], True, c, thick)
        cv2.putText(img, name, tuple(p.min(0) + [4, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=ha4m_io.DEFAULT_ROOT)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--config", default="config/station.yaml")
    ap.add_argument("--render", default="", help="chỉ vẽ zone hiện có ra file PNG (không mở cửa sổ)")
    ap.add_argument("--scale", type=float, default=0.5, help="tỉ lệ hiển thị cửa sổ")
    a = ap.parse_args()

    img = ha4m_io.read_frame(a.data_root, a.subject, a.trial, a.frame)
    if img is None:
        sys.exit(f"không đọc được frame {a.frame}")
    zones = {}
    if Path(a.config).exists():
        zones = (yaml.safe_load(open(a.config, encoding="utf-8")) or {}).get("zones") or {}

    if a.render:
        Path(a.render).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.render, draw_zones(img.copy(), zones))
        print(f"đã ghi {a.render}")
        return

    s = a.scale
    new, cur = {}, []
    win = "pick_zones"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            cur.append([int(x / s), int(y / s)])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = draw_zones(img.copy(), {**zones, **new})
        if cur:
            cv2.polylines(vis, [np.array(cur, np.int32)], False, (0, 255, 255), 2)
            for p in cur:
                cv2.circle(vis, tuple(p), 5, (0, 255, 255), -1)
        cv2.imshow(win, cv2.resize(vis, None, fx=s, fy=s))
        k = cv2.waitKey(30) & 0xFF
        if k in (13, ord("n")) and len(cur) >= 3:
            name = input("Tên zone (vd bin_carrier, assembly_area): ").strip() or f"zone_{len(new) + 1}"
            new[name], cur = cur, []
        elif k == ord("u") and cur:
            cur.pop()
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    print(yaml.safe_dump({"zones": new}, default_flow_style=None, sort_keys=False))


if __name__ == "__main__":
    main()

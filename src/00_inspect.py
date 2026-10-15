"""[00] Khảo sát dữ liệu HA4M thật → outputs/inspect_report.md.

In cây thư mục 2 cấp, 10 dòng đầu Labels.txt + đoán schema, pattern tên frame, 1 file skeleton,
số frame, kích thước ảnh, fps ước lượng từ DeviceTimeStamp trong tên file.

    python src/00_inspect.py --data-root data/ha4m [--subject U001 --trial V001]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.core.cli import base_parser, setup  # noqa: E402
from kaizen.core.io import can_write  # noqa: E402
from kaizen.skeleton.inspect_ha4m import guess_label_schema, inspect_report, tree  # noqa: E402,F401

TAG = "00"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--subject", default="")
    ap.add_argument("--trial", default="")
    ap.add_argument("--out", default="outputs/inspect_report.md")
    a = ap.parse_args()
    setup(a, TAG, need_run=False)
    out = Path(a.out)
    if not can_write(out, a.force, TAG):
        return
    try:
        md = inspect_report(a.data_root, a.subject, a.trial)
    except FileNotFoundError as e:
        print(f"[{TAG}] {e}", file=sys.stderr)
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"[{TAG}] đã ghi {out}")


if __name__ == "__main__":
    main()

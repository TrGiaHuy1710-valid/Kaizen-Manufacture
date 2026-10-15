"""Sinh tests/fixtures/sop_sample.pdf bằng PyMuPDF: SOP mẫu 2 trang, 12 bước, 1 bảng (tất định → doc_id cố định).

    python tests/fixtures/make_sop_pdf.py [out.pdf] [--scanned-page]
Font DejaVuSans đi kèm matplotlib (đủ dấu tiếng Việt). Trang 1: tiêu đề, phạm vi, bước 1–6; trang 2: bước 7–12
+ bảng thời gian chuẩn / khay. --scanned-page thêm trang 3 chỉ có ảnh (giả trang scan → needs_ocr).
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pymupdf

TITLE = "HƯỚNG DẪN CÔNG VIỆC — LẮP BỘ BÁNH RĂNG HÀNH TINH (EGT)"
STEPS = [
    "Lấy giá đỡ Carrier từ khay phải trên, đặt vào đồ gá giữa bàn.",
    "Lấy và lắp 3 ổ bi bánh răng (Gear Bearing) lên 3 trục của Carrier.",
    "Lấy và lắp 3 bánh răng hành tinh (Planet Gear) lên các ổ bi, xoay nhẹ để khớp răng.",
    "Lấy và lắp trục Carrier (Carrier Shaft) vào lỗ giữa.",
    "Lấy và lắp trục bánh răng mặt trời (Sun Shaft).",
    "Lấy và lắp bánh răng mặt trời (Sun Gear) ăn khớp với 3 bánh răng hành tinh.",
    "Lấy và lắp ổ bi bánh răng mặt trời (Sun Gear Bearing).",
    "Lấy và lắp vành răng ngoài (Ring Bear). Lưu ý an toàn: đeo găng tay chống cắt khi cầm vành răng.",
    "Lấy Block 2 và đặt lên Block 1, ấn đều hai tay đến khi nghe tiếng khớp.",
    "Lấy và lắp nắp (Cover) đúng chiều mũi tên.",
    "Lấy và đặt 2 con vít vào lỗ trên nắp.",
    "Dùng lục giác siết cả 2 vít, trả lục giác về giá và đặt cụm EGT vào khay thành phẩm.",
]
STD_TIME = [4.0, 9.5, 10.0, 3.5, 3.0, 4.5, 3.5, 5.0, 6.0, 4.0, 7.5, 12.0]
TRAY = ["Khay phải trên", "Khay phải trên", "Khay trước số 5", "Khay trước số 4", "Khay trước số 3",
        "Khay trước số 2", "Khay trước số 1", "Khay trái trên", "Đồ gá giữa", "Khay trái trên", "Khay vít",
        "Khay vít"]


def _font_paths() -> tuple[str, str]:
    import matplotlib
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return str(d / "DejaVuSans.ttf"), str(d / "DejaVuSans-Bold.ttf")


class _Writer:
    def __init__(self, doc):
        self.doc, self.page, self.y = doc, None, 0.0
        self.reg, self.bold = _font_paths()

    def new_page(self):
        self.page = self.doc.new_page(width=595, height=842)
        self.page.insert_font(fontname="dv", fontfile=self.reg)
        self.page.insert_font(fontname="dvb", fontfile=self.bold)
        self.y = 60.0

    def line(self, text, size=10.0, bold=False, x=50.0, gap=4.0):
        self.y += size
        self.page.insert_text((x, self.y), text, fontname="dvb" if bold else "dv", fontsize=size)
        self.y += gap

    def para(self, text, size=10.0, width=92, x=50.0):
        for ln in textwrap.wrap(text, width):
            self.line(ln, size, x=x, gap=3.0)
        self.y += 6

    def table(self, header, rows, widths, size=9.5, row_h=17.0, x=50.0):
        y0 = self.y + 4
        allrows = [header] + rows
        for i, r in enumerate(allrows):
            cx = x
            top = y0 + i * row_h
            for w, cell in zip(widths, r):
                self.page.draw_rect(pymupdf.Rect(cx, top, cx + w, top + row_h), color=(0, 0, 0), width=0.6)
                self.page.insert_text((cx + 4, top + row_h - 5), str(cell), fontname="dvb" if i == 0 else "dv",
                                      fontsize=size)
                cx += w
        self.y = y0 + len(allrows) * row_h + 10


def build_sop_pdf(out: str | Path, scanned_page: bool = False) -> Path:
    """Ghi PDF mẫu ra `out` (tất định) và trả đường dẫn."""
    doc = pymupdf.open()
    w = _Writer(doc)
    w.new_page()
    w.line(TITLE, 15, bold=True, gap=8)
    w.line("Mã tài liệu: WI-EGT-01 · Trạm lắp tay số 3 · Phiên bản 1.0", 9.5, gap=10)
    w.line("1. Phạm vi", 12.5, bold=True, gap=6)
    w.para("Tài liệu mô tả trình tự chuẩn để một công nhân lắp trọn một bộ bánh răng hành tinh tại bàn lắp. "
           "Linh kiện đặt trong các khay trong suốt; cụm lắp được giữ trên đồ gá giữa bàn. Một chu kỳ tính từ lúc "
           "bắt đầu bước đầu tiên đến khi đặt cụm thành phẩm vào khay.")
    w.line("2. Trình tự lắp ráp", 12.5, bold=True, gap=6)
    for i, s in enumerate(STEPS[:6], 1):
        w.para(f"Bước {i}: {s}")
    w.new_page()
    for i, s in enumerate(STEPS[6:], 7):
        w.para(f"Bước {i}: {s}")
    w.line("3. Thời gian chuẩn và vị trí khay", 12.5, bold=True, gap=6)
    w.table(["Bước", "Thời gian chuẩn (s)", "Khay"], [[i + 1, f"{t:g}", k] for i, (t, k) in enumerate(zip(STD_TIME, TRAY))],
            [60, 140, 180])
    if scanned_page:
        p = doc.new_page(width=595, height=842)
        rng = np.random.default_rng(0)
        img = (rng.random((300, 220, 3)) * 255).astype(np.uint8)
        pix = pymupdf.Pixmap(pymupdf.csRGB, 220, 300, img.tobytes(), False)
        p.insert_image(p.rect, pixmap=pix)
    doc.set_metadata({"title": "Hướng dẫn công việc — Lắp bộ bánh răng hành tinh", "author": "Kaizen test",
                      "creator": "tests/fixtures/make_sop_pdf.py", "producer": "PyMuPDF",
                      "creationDate": "D:20260101000000", "modDate": "D:20260101000000"})
    doc.subset_fonts()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out), garbage=4, deflate=True, no_new_id=True)
    doc.close()
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    p = build_sop_pdf(args[0] if args else Path(__file__).with_name("sop_sample.pdf"), "--scanned-page" in sys.argv)
    print(p, os.path.getsize(p), "bytes")

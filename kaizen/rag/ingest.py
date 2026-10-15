"""Document understanding: PDF / DOCX / MD / TXT → danh sách Block có cấu trúc (heading / text / table + trang + bbox).

PDF (PyMuPDF):
  - text theo block (page.get_text("dict")) giữ số trang + bbox; heading nhận theo cỡ chữ (≥ 1.15 × cỡ chữ thân
    trung vị của tài liệu) hoặc dòng ngắn in đậm; cấp heading theo thứ hạng cỡ chữ.
  - bảng: page.find_tables() → markdown (table.to_markdown()); text nằm trong bbox bảng bị bỏ để không lặp.
  - trang scan (ít text, có ảnh phủ phần lớn trang) → needs_ocr, bỏ qua ở v1 (TODO: OCR), KHÔNG crash.
DOCX: đọc thẳng word/document.xml (zip) — Heading*/Title → heading, w:tbl → markdown; ngắt trang (w:br type=page /
  lastRenderedPageBreak) → tăng số trang (xấp xỉ).
MD: '#' → heading theo cấp, dòng '|' liên tiếp → bảng, đoạn cách bởi dòng trống. TXT: đoạn; '\\f' = sang trang.
doc_id = "DOC-" + sha1(bytes file)[:6] → cùng file = cùng doc_id (ingest lại idempotent).
"""
from __future__ import annotations

import hashlib
import re
import statistics
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".md": "md", ".markdown": "md", ".txt": "txt"}
OCR_TEXT_MIN_CHARS = 30          # trang có ít hơn số ký tự này …
OCR_IMAGE_MIN_FRAC = 0.3         # … và ảnh phủ ≥ 30% diện tích → coi là trang scan


@dataclass
class Block:
    """1 khối nội dung theo thứ tự đọc."""
    kind: str                       # heading | text | table
    text: str
    page: int                       # 1-based
    level: int = 0                  # cấp heading (1 = lớn nhất); 0 cho text/table
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ParsedDoc:
    """Kết quả parse 1 tài liệu."""
    doc_id: str
    title: str
    filename: str
    kind: str
    pages: int
    blocks: list[Block] = field(default_factory=list)
    needs_ocr_pages: list[int] = field(default_factory=list)
    sha1: str = ""


class UnsupportedDocument(ValueError):
    """Định dạng file không hỗ trợ."""


def doc_id_for(data: bytes) -> tuple[str, str]:
    """(doc_id, sha1) từ nội dung file."""
    h = hashlib.sha1(data).hexdigest()
    return f"DOC-{h[:6]}", h


def parse_document(path: str | Path | None = None, *, data: bytes | None = None, filename: str | None = None,
                   title: str | None = None) -> ParsedDoc:
    """Parse file (theo đuôi) từ đường dẫn hoặc bytes + filename."""
    if data is None:
        if path is None:
            raise ValueError("cần path hoặc data")
        data = Path(path).read_bytes()
    filename = filename or (Path(path).name if path else "document")
    kind = SUPPORTED.get(Path(filename).suffix.lower())
    if kind is None:
        raise UnsupportedDocument(f"không hỗ trợ '{Path(filename).suffix}' (nhận: {', '.join(sorted(SUPPORTED))})")
    doc_id, sha = doc_id_for(data)
    if kind == "pdf":
        blocks, pages, ocr, meta_title = _parse_pdf(data)
    elif kind == "docx":
        blocks, pages, ocr, meta_title = _parse_docx(data)
    else:
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        blocks, pages, ocr, meta_title = (_parse_md(text) if kind == "md" else _parse_txt(text)), 0, [], None
        pages = max((b.page for b in blocks), default=1)
    first_heading = next((b.text for b in blocks if b.kind == "heading"), None)
    t = title or meta_title or first_heading or Path(filename).stem
    return ParsedDoc(doc_id, t.strip(), filename, kind, pages, blocks, ocr, sha)


# ---------------------------------------------------------------- PDF
def _span_stats(pdf) -> float:
    """Cỡ chữ thân (trung vị theo số ký tự) của cả tài liệu."""
    sizes = []
    for page in pdf:
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for sp in ln["spans"]:
                    n = len(sp["text"].strip())
                    if n:
                        sizes += [round(sp["size"], 1)] * min(n, 200)
    return statistics.median(sizes) if sizes else 11.0


def _inside(b: tuple, boxes: list[tuple]) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any(x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1 for x0, y0, x1, y1 in boxes)


def _is_scanned(page) -> bool:
    text = page.get_text("text").strip()
    if len(text) >= OCR_TEXT_MIN_CHARS:
        return False
    area = abs(page.rect.width * page.rect.height) or 1.0
    img_area = 0.0
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        img_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return img_area / area >= OCR_IMAGE_MIN_FRAC


def _parse_pdf(data: bytes) -> tuple[list[Block], int, list[int], str | None]:
    import pymupdf
    if hasattr(pymupdf, "no_recommend_layout"):
        pymupdf.no_recommend_layout()             # tắt lời gợi ý cài pymupdf_layout in ra stdout
    pdf = pymupdf.open(stream=data, filetype="pdf")
    body = _span_stats(pdf)
    raw: list[tuple[Block, float]] = []          # (block, cỡ chữ lớn nhất) để xếp cấp heading
    ocr: list[int] = []
    for pno, page in enumerate(pdf, 1):
        if _is_scanned(page):
            ocr.append(pno)                       # TODO v2: OCR (vd PaddleOCR/Tesseract) cho trang scan
            continue
        items: list[tuple[float, float, Block, float]] = []
        tboxes = []
        try:
            tables = page.find_tables().tables
        except Exception:                         # find_tables lỗi trên PDF lạ → bỏ qua bảng, giữ text
            tables = []
        for t in tables:
            md = t.to_markdown(clean=False).strip()
            if md:
                tboxes.append(tuple(t.bbox))
                items.append((t.bbox[1], t.bbox[0], Block("table", md, pno, 0, tuple(t.bbox)), 0.0))
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0 or _inside(b["bbox"], tboxes):
                continue
            lines, max_size, bold_chars, n_chars = [], 0.0, 0, 0
            for ln in b["lines"]:
                s = "".join(sp["text"] for sp in ln["spans"]).strip()
                if s:
                    lines.append(s)
                for sp in ln["spans"]:
                    n = len(sp["text"].strip())
                    n_chars += n
                    max_size = max(max_size, sp["size"])
                    if sp["flags"] & 16 or "bold" in sp["font"].lower():
                        bold_chars += n
            text = " ".join(lines).strip()
            if not text:
                continue
            short = len(text) <= 90 and not text.endswith((".", ";", ","))
            is_head = max_size >= body * 1.15 or (short and n_chars and bold_chars / n_chars > 0.8)
            items.append((b["bbox"][1], b["bbox"][0],
                          Block("heading" if is_head else "text", text, pno, 0, tuple(b["bbox"])), max_size))
        items.sort(key=lambda x: (round(x[0], 1), x[1]))
        raw += [(blk, size) for _, _, blk, size in items]
    head_sizes = sorted({round(s, 1) for blk, s in raw if blk.kind == "heading"}, reverse=True)
    for blk, s in raw:
        if blk.kind == "heading":
            blk.level = min(head_sizes.index(round(s, 1)) + 1, 6)
    meta_title = (pdf.metadata or {}).get("title") or None
    return [blk for blk, _ in raw], pdf.page_count, ocr, meta_title


# ---------------------------------------------------------------- DOCX
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(el) -> str:
    return "".join(t.text or "" for t in el.iter(f"{_W}t")).strip()


def _md_table(rows: list[list[str]]) -> str:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    clean = lambda c: c.replace("|", "/").replace("\n", " ").strip()  # noqa: E731
    out = ["| " + " | ".join(clean(c) for c in rows[0]) + " |", "|" + "---|" * w]
    out += ["| " + " | ".join(clean(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _parse_docx(data: bytes) -> tuple[list[Block], int, list[int], str | None]:
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        title = None
        if "docProps/core.xml" in z.namelist():
            core = ET.fromstring(z.read("docProps/core.xml"))
            t = core.find("{http://purl.org/dc/elements/1.1/}title")
            title = (t.text or "").strip() or None if t is not None else None
    body = root.find(f"{_W}body")
    blocks, page = [], 1
    for el in list(body) if body is not None else []:
        if el.tag == f"{_W}p":
            page += sum(1 for _ in el.iter(f"{_W}lastRenderedPageBreak"))     # đoạn bắt đầu ở trang mới
            text = _docx_text(el)
            style = el.find(f"{_W}pPr/{_W}pStyle")
            sval = ((style.get(f"{_W}val") if style is not None else "") or "").replace(" ", "")
            m = re.match(r"(?i)^(heading|title|tiêuđề)(\d*)$", sval)
            if text:
                lvl = (int(m.group(2)) if m.group(2) else 1) if m else 0
                blocks.append(Block("heading" if m else "text", text, page, lvl))
            page += sum(1 for br in el.iter(f"{_W}br") if br.get(f"{_W}type") == "page")   # ngắt sau đoạn
        elif el.tag == f"{_W}tbl":
            rows = [[_docx_text(tc) for tc in tr.iter(f"{_W}tc")] for tr in el.iter(f"{_W}tr")]
            md = _md_table(rows)
            if md:
                blocks.append(Block("table", md, page))
    return blocks, page, [], title


# ---------------------------------------------------------------- MD / TXT
def _parse_md(text: str) -> list[Block]:
    blocks, buf, tbl, page = [], [], [], 1

    def flush():
        if buf:
            blocks.append(Block("text", " ".join(x.strip() for x in buf).strip(), page))
            buf.clear()
        if tbl:
            blocks.append(Block("table", "\n".join(tbl), page))
            tbl.clear()

    for ln in text.split("\n"):
        if "\f" in ln:
            flush()
            page += ln.count("\f")
            ln = ln.replace("\f", "")
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush()
            blocks.append(Block("heading", m.group(2).strip(), page, len(m.group(1))))
        elif ln.strip().startswith("|"):
            if buf:
                blocks.append(Block("text", " ".join(x.strip() for x in buf).strip(), page))
                buf.clear()
            tbl.append(ln.strip())
        elif not ln.strip():
            flush()
        else:
            if tbl:
                flush()
            buf.append(ln)
    flush()
    return [b for b in blocks if b.text]


def _parse_txt(text: str) -> list[Block]:
    blocks = []
    for pno, page in enumerate(text.split("\f"), 1):
        for para in re.split(r"\n\s*\n", page):
            p = " ".join(x.strip() for x in para.split("\n")).strip()
            if p:
                blocks.append(Block("text", p, pno))
    return blocks

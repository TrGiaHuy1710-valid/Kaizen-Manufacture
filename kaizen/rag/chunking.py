"""Chunk theo cấu trúc: heading-aware, mục tiêu min_tokens–max_tokens (mặc định 300–600), overlap 50.

Quy tắc:
  - Sang trang → luôn cắt (mỗi chunk thuộc 1 trang → trích dẫn chính xác trang).
  - Gặp heading → cắt nếu chunk hiện tại đã ≥ min_tokens; nếu chưa thì gộp tiếp (mục nhỏ không thành chunk vụn).
  - Bảng → chunk riêng (kind=table), giữ markdown; bảng quá dài → tách theo dòng, lặp lại header.
  - Khối vượt max_tokens → tách theo câu/từ; chunk kế tiếp mở đầu bằng `overlap` token cuối của chunk trước.
Token ≈ từ cách nhau bởi khoảng trắng (tiếng Việt: âm tiết) — xấp xỉ đủ cho kích thước chunk.
chunk_id = f"{doc_id}:p{page}:c{k}" (k = thứ tự chunk trong trang) → ổn định khi ingest lại cùng file.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from kaizen.rag.ingest import Block, ParsedDoc

MIN_TOKENS, MAX_TOKENS, OVERLAP = 300, 600, 50


@dataclass
class Chunk:
    """1 chunk + metadata trích dẫn. score chỉ có khi là kết quả search."""
    chunk_id: str
    doc_id: str
    title: str
    page: int
    section: str
    text: str
    order: int
    kind: str = "text"
    n_tokens: int = 0
    score: float = 0.0

    def to_dict(self) -> dict:
        """dict JSON-able."""
        return asdict(self)

    @property
    def metadata(self) -> dict:
        """Metadata cho vector store (không None)."""
        return {"doc_id": self.doc_id, "title": self.title, "page": self.page, "section": self.section or "",
                "chunk_id": self.chunk_id, "order": self.order, "kind": self.kind, "n_tokens": self.n_tokens}


def n_tokens(text: str) -> int:
    """Số token xấp xỉ (từ cách nhau bởi khoảng trắng)."""
    return len(text.split())


def _split_long(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Tách text dài theo câu, mỗi phần ≤ max_tokens, phần sau mở đầu bằng `overlap` token cuối của phần trước."""
    sents = [s for s in re.split(r"(?<=[.!?;:])\s+", text) if s.strip()]
    words: list[str] = []
    for s in sents:
        w = s.split()
        if len(w) > max_tokens:                       # câu siêu dài → cắt cứng theo từ
            for i in range(0, len(w), max_tokens - overlap):
                words.append(" ".join(w[i:i + max_tokens - overlap]))
        else:
            words.append(s)
    out, cur = [], []
    for s in words:
        if cur and len(cur) + len(s.split()) > max_tokens:
            out.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur += s.split()
    if cur:
        out.append(" ".join(cur))
    return out


def _split_table(md: str, max_tokens: int) -> list[str]:
    lines = md.split("\n")
    head, rows = lines[:2], lines[2:]
    out, cur = [], []
    for r in rows:
        if cur and n_tokens("\n".join(head + cur + [r])) > max_tokens:
            out.append("\n".join(head + cur))
            cur = []
        cur.append(r)
    out.append("\n".join(head + cur))
    return out


def chunk_document(doc: ParsedDoc, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS,
                   overlap: int = OVERLAP) -> list[Chunk]:
    """ParsedDoc → list Chunk có chunk_id ổn định + metadata {doc_id, title, page, section}."""
    chunks: list[Chunk] = []
    per_page: dict[int, int] = {}
    stack: list[tuple[int, str]] = []                     # (level, heading) đường dẫn mục hiện tại
    cur: list[str] = []
    cur_page, cur_section = None, ""

    def section_path() -> str:
        return " > ".join(h for _, h in stack)

    def emit(text: str, page: int, section: str, kind: str = "text") -> None:
        k = per_page.get(page, 0)
        per_page[page] = k + 1
        chunks.append(Chunk(f"{doc.doc_id}:p{page}:c{k}", doc.doc_id, doc.title, page, section, text.strip(),
                            len(chunks), kind, n_tokens(text)))

    def flush() -> None:
        nonlocal cur
        if cur and cur_page is not None:
            body = "\n".join(cur)
            if n_tokens(body) > max_tokens:
                for part in _split_long(body, max_tokens, overlap):
                    emit(part, cur_page, cur_section)
            else:
                emit(body, cur_page, cur_section)
        cur = []

    for b in doc.blocks:
        if cur_page is not None and b.page != cur_page:
            flush()
        if b.kind == "heading":
            if cur and n_tokens("\n".join(cur)) >= min_tokens:
                flush()
            while stack and stack[-1][0] >= max(b.level, 1):
                stack.pop()
            stack.append((max(b.level, 1), b.text))
            if not cur:
                cur_section = section_path()
            cur.append(b.text)
            cur_page = b.page
            continue
        if b.kind == "table":
            flush()
            sec = section_path()
            for part in (_split_table(b.text, max_tokens) if n_tokens(b.text) > max_tokens else [b.text]):
                emit(part, b.page, sec, "table")
            cur_page, cur_section = b.page, sec
            continue
        if not cur:
            cur_section = section_path()
        bt = n_tokens(b.text)
        # khối tự nó > max_tokens: nối vào chunk hiện tại, flush() sẽ tách cả khối (giữ heading đi cùng nội dung);
        # khối vừa max_tokens nhưng cộng dồn bị tràn: cắt, chunk mới mở đầu bằng overlap của chunk trước
        if cur and bt <= max_tokens and n_tokens("\n".join(cur)) + bt > max_tokens:
            tail = " ".join("\n".join(cur).split()[-overlap:]) if overlap else ""
            flush()
            cur = [tail] if tail else []
            cur_section = section_path()
        cur.append(b.text)
        cur_page = b.page
    flush()
    return chunks

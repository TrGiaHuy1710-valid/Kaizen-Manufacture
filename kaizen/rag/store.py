"""Document store: Chroma persistent (không server) + registry JSON + chunks JSONL theo tài liệu.

Bố cục (mặc định data/rag_store/, đổi bằng env KAIZEN_RAG_STORE):
    chroma/                         vector index — 1 collection / embedder (chunks_<slug>), cosine
    documents.json                  registry {doc_id: {title, filename, kind, pages, chunks, needs_ocr_pages, ...}}
    docs/<doc_id>/chunks.jsonl      toàn bộ chunk theo thứ tự (nguồn sự thật để liệt kê / re-index)
    docs/<doc_id>/<filename>        bản sao file gốc
Ingest idempotent theo doc_id (= hash nội dung). Đổi embedder → chunk của tài liệu cũ được index lại từ JSONL.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from kaizen.rag.chunking import Chunk, chunk_document
from kaizen.rag.embed import Embedder, get_embedder
from kaizen.rag.ingest import ParsedDoc, parse_document

DEFAULT_STORE = "data/rag_store"


@dataclass
class IngestResult:
    """Kết quả ingest 1 tài liệu (payload của POST /documents)."""
    doc_id: str
    title: str
    filename: str
    pages: int
    chunks: int
    needs_ocr_pages: list[int] = field(default_factory=list)
    already_present: bool = False

    def to_dict(self) -> dict:
        """dict JSON-able."""
        return dict(self.__dict__)


def default_store_path() -> str:
    """env KAIZEN_RAG_STORE hoặc data/rag_store."""
    return os.environ.get("KAIZEN_RAG_STORE") or DEFAULT_STORE


class DocumentStore:
    """Ingest / liệt kê / truy xuất chunk. Thoả interface Retriever của kaizen.llm.explain (search + has_chunk)."""

    def __init__(self, path: str | Path | None = None, embedder: Embedder | str | None = None):
        self.path = Path(path or default_store_path())
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder if embedder is not None and not isinstance(embedder, str) else get_embedder(embedder)
        self._client = None
        self._col = None
        self._has_cache: dict[str, bool] = {}

    # ------------------------------------------------------------ chroma
    @property
    def collection(self):
        """Collection Chroma của embedder hiện tại (tạo + đồng bộ lười)."""
        if self._col is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(path=str(self.path / "chroma"),
                                                     settings=Settings(anonymized_telemetry=False))
            self._col = self._client.get_or_create_collection(f"chunks_{self.embedder.slug}",
                                                              metadata={"hnsw:space": "cosine"},
                                                              embedding_function=None)
            self._sync()
        return self._col

    def _sync(self) -> None:
        """Index lại từ JSONL các tài liệu chưa có trong collection của embedder hiện tại."""
        reg = self._registry()
        for doc_id, info in reg.items():
            if self.embedder.slug not in info.get("embedders", []):
                self._index(self.get_chunks(doc_id))
                info.setdefault("embedders", []).append(self.embedder.slug)
        self._save_registry(reg)

    def _index(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), 128):
            part = chunks[i:i + 128]
            vecs = self.embedder.embed_documents([self._embed_text(c) for c in part])
            self._col.upsert(ids=[c.chunk_id for c in part], embeddings=vecs, documents=[c.text for c in part],
                             metadatas=[c.metadata for c in part])

    @staticmethod
    def _embed_text(c: Chunk) -> str:
        return f"{c.title} | {c.section}\n{c.text}" if c.section else f"{c.title}\n{c.text}"

    # ------------------------------------------------------------ registry
    def _registry(self) -> dict:
        p = self.path / "documents.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _save_registry(self, reg: dict) -> None:
        (self.path / "documents.json").write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_documents(self) -> list[dict]:
        """Tài liệu đã ingest (theo thời điểm ingest)."""
        return sorted(self._registry().values(), key=lambda d: d.get("ingested_at", ""))

    def get_document(self, doc_id: str) -> dict | None:
        """Thông tin 1 tài liệu (None nếu không có)."""
        return self._registry().get(doc_id)

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Mọi chunk của tài liệu theo thứ tự đọc (từ JSONL)."""
        p = self.path / "docs" / doc_id / "chunks.jsonl"
        if not p.exists():
            return []
        return [Chunk(**json.loads(ln)) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # ------------------------------------------------------------ ingest
    def ingest(self, path: str | Path | None = None, *, data: bytes | None = None, filename: str | None = None,
               title: str | None = None, force: bool = False) -> IngestResult:
        """Parse → chunk → embed → upsert. Cùng nội dung (doc_id) đã có → bỏ qua (trừ force)."""
        if data is None:
            data = Path(path).read_bytes()
        doc = parse_document(path, data=data, filename=filename or (Path(path).name if path else None), title=title)
        reg = self._registry()
        if doc.doc_id in reg and not force:
            info = reg[doc.doc_id]
            _ = self.collection                                  # đảm bảo đã index cho embedder hiện tại
            return IngestResult(doc.doc_id, info["title"], info["filename"], info["pages"], info["chunks"],
                                info.get("needs_ocr_pages", []), already_present=True)
        chunks = chunk_document(doc)
        self._write_doc(doc, chunks, data)
        col = self.collection
        if force:
            col.delete(where={"doc_id": doc.doc_id})
        self._col = col
        self._index(chunks)
        reg = self._registry()
        reg[doc.doc_id] = {"doc_id": doc.doc_id, "title": doc.title, "filename": doc.filename, "kind": doc.kind,
                           "pages": doc.pages, "chunks": len(chunks), "needs_ocr_pages": doc.needs_ocr_pages,
                           "sha1": doc.sha1, "ingested_at": dt.datetime.now().isoformat(timespec="seconds"),
                           "embedders": [self.embedder.slug]}
        self._save_registry(reg)
        self._has_cache.clear()
        return IngestResult(doc.doc_id, doc.title, doc.filename, doc.pages, len(chunks), doc.needs_ocr_pages)

    def _write_doc(self, doc: ParsedDoc, chunks: list[Chunk], data: bytes) -> None:
        d = self.path / "docs" / doc.doc_id
        d.mkdir(parents=True, exist_ok=True)
        (d / Path(doc.filename).name).write_bytes(data)
        (d / "chunks.jsonl").write_text("".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in chunks),
                                        encoding="utf-8")

    # ------------------------------------------------------------ retrieval
    def search(self, query: str, k: int = 6, filter: dict | None = None) -> list[Chunk]:
        """Top-k chunk theo cosine (score = 1 − distance), kèm metadata đầy đủ. filter: {doc_id: ...}."""
        col = self.collection
        if col.count() == 0:
            return []
        where = {"doc_id": filter["doc_id"]} if filter and filter.get("doc_id") else None
        r = col.query(query_embeddings=[self.embedder.embed_query(query)], n_results=max(1, int(k)), where=where,
                      include=["documents", "metadatas", "distances"])
        out = []
        for cid, text, md, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            out.append(Chunk(cid, md["doc_id"], md.get("title", ""), int(md["page"]), md.get("section", ""), text,
                             int(md.get("order", 0)), md.get("kind", "text"), int(md.get("n_tokens", 0)),
                             round(1.0 - float(dist), 4)))
        return out

    def has_chunk(self, chunk_id: str) -> bool:
        """chunk_id có trong store (collection hiện tại) không — dùng cho validator citation."""
        if chunk_id not in self._has_cache:
            self._has_cache[chunk_id] = bool(self.collection.get(ids=[chunk_id])["ids"])
        return self._has_cache[chunk_id]

    def count(self) -> int:
        """Số chunk trong collection của embedder hiện tại."""
        return self.collection.count()

    def describe(self) -> dict:
        """Tóm tắt store (cho manifest / health)."""
        return {"path": str(self.path), "embedder": self.embedder.slug, "documents": len(self._registry()),
                "chunks": self.count()}


def count_chunks_fast(path: str | Path | None = None) -> int:
    """Tổng số chunk theo registry (không nạp embedder/Chroma) — cho /health."""
    p = Path(path or default_store_path()) / "documents.json"
    if not p.exists():
        return 0
    return sum(int(d.get("chunks", 0)) for d in json.loads(p.read_text(encoding="utf-8")).values())

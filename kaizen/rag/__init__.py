"""RAG tài liệu: parse (PDF/DOCX/MD/TXT) → chunk theo cấu trúc → embed → Chroma → truy xuất / trích bước chuẩn.

    upload → chunks → (query | extract-steps) → explain có trích dẫn (chunk_id "DOC-xxxxxx:pN:cM")

    DocumentStore(path, embedder).ingest(file) / .search(query, k, filter) / .has_chunk(id) / .get_chunks(doc_id)
    extract_steps(doc_id, provider, store) -> StepsSpec;  to_station(spec) -> bản nháp station.yaml
Import nhẹ: chromadb / sentence-transformers / pymupdf chỉ nạp khi dùng.
"""
from kaizen.rag.chunking import Chunk, chunk_document
from kaizen.rag.embed import Embedder, HashEmbedder, SentenceTransformerEmbedder, get_embedder
from kaizen.rag.extract import ExtractedStep, StepsSpec, extract_steps, to_station
from kaizen.rag.ingest import ParsedDoc, UnsupportedDocument, parse_document
from kaizen.rag.store import DocumentStore, IngestResult

__all__ = ["Chunk", "DocumentStore", "Embedder", "ExtractedStep", "HashEmbedder", "IngestResult", "ParsedDoc",
           "SentenceTransformerEmbedder", "StepsSpec", "UnsupportedDocument", "chunk_document", "extract_steps",
           "get_embedder", "parse_document", "to_station"]

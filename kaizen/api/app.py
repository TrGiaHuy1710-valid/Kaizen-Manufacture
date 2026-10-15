"""FastAPI service — chạy từ repo root:  uvicorn kaizen.api.app:app [--port 8000]

    POST /documents            multipart file (PDF/DOCX/MD/TXT) → {doc_id, title, pages, chunks, needs_ocr_pages}
    GET  /documents            danh sách đã ingest
    POST /rag/query            {query, k, doc_id?} → chunks + scores (+ metadata trích dẫn)
    POST /rag/extract-steps    {doc_id, provider?, llm_model?} → StepsSpec (mỗi bước có trang nguồn)
    POST /analyze/explain      {run_id, provider?, llm_model?, use_rag, k, save} → nội dung explained.json
    GET  /runs                 các run trong outputs/runs (qua run_io)
    GET  /runs/{id}/metrics | /findings | /explained
    GET  /health               {status, providers_available, store_chunks}
Env: KAIZEN_RAG_STORE (data/rag_store), KAIZEN_EMBEDDER (e5 | hash | ...), LLM_PROVIDER, LLM_MODEL, *_API_KEY.
Lỗi provider/thiếu key → HTTP 400 nêu tên biến env. CORS mở cho localhost (Streamlit gọi được).
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import kaizen
from kaizen.core import run_io
from kaizen.core.io import read_json, write_json
from kaizen.llm.explain import KnowledgeBase, explain
from kaizen.llm.prompts import EXPLAIN_RAG_V1, EXPLAIN_V1
from kaizen.llm.providers import LLMError, MissingAPIKeyError, available_providers, get_provider
from kaizen.rag.extract import extract_steps
from kaizen.rag.ingest import UnsupportedDocument
from kaizen.rag.store import DocumentStore, count_chunks_fast, default_store_path

RUN_ID_RE = re.compile(r"^(latest|[\w.\-]+)$")
MAX_UPLOAD_MB = 50


class QueryIn(BaseModel):
    """Body POST /rag/query."""
    query: str = Field(min_length=1)
    k: int = Field(6, ge=1, le=50)
    doc_id: Optional[str] = None


class ExtractIn(BaseModel):
    """Body POST /rag/extract-steps."""
    doc_id: str
    provider: Optional[str] = None
    llm_model: Optional[str] = None


class ExplainIn(BaseModel):
    """Body POST /analyze/explain."""
    run_id: str = "latest"
    provider: Optional[str] = None
    llm_model: Optional[str] = None
    use_rag: bool = False
    k: int = Field(4, ge=1, le=20)
    save: bool = Field(False, description="ghi đè <run>/explained/explained.json + manifest (như 07 --force)")


def _provider(name: str | None, model: str | None):
    try:
        return get_provider(name, model)
    except MissingAPIKeyError as e:
        raise HTTPException(400, f"{e}") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _run(run_id: str) -> run_io.RunContext:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, f"run_id không hợp lệ: {run_id!r}")
    try:
        return run_io.open_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


def _artifact(run_id: str, stage: str, name: str) -> dict:
    p = _run(run_id).root / stage / name
    if not p.exists():
        raise HTTPException(404, f"run {run_id} chưa có {stage}/{name}")
    return read_json(p)


def create_app(store_path: str | Path | None = None, embedder: str | None = None,
               kb_path: str = "config/kaizen_kb.md", sop_path: str = "config/sop.md",
               extractions_dir: str | Path = "outputs/extractions") -> FastAPI:
    """App factory (store nạp lười: embedder chỉ tải khi cần)."""
    app = FastAPI(title="AI Kaizen Copilot API", version=kaizen.__version__,
                  description="RAG tài liệu + phân tích Kaizen. Mọi con số do code tính; LLM chỉ giải thích.")
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                       allow_methods=["*"], allow_headers=["*"])
    holder: dict = {}
    lock = threading.Lock()
    path = Path(store_path or default_store_path())

    def store() -> DocumentStore:
        if "store" not in holder:
            holder["store"] = DocumentStore(path, embedder)
        return holder["store"]

    @app.get("/health")
    def health() -> dict:
        """Trạng thái + provider có key + số chunk trong store (không nạp model)."""
        return {"status": "ok", "version": kaizen.__version__, "providers_available": available_providers(),
                "store_chunks": count_chunks_fast(path), "store": str(path),
                "embedder": embedder or os.environ.get("KAIZEN_EMBEDDER") or "e5"}

    @app.post("/documents")
    def upload(file: UploadFile = File(...)) -> dict:
        """Ingest 1 tài liệu (idempotent theo nội dung)."""
        data = file.file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"file > {MAX_UPLOAD_MB} MB")
        if not data:
            raise HTTPException(400, "file rỗng")
        try:
            with lock:
                r = store().ingest(data=data, filename=Path(file.filename or "upload").name)
        except UnsupportedDocument as e:
            raise HTTPException(415, str(e)) from e
        except Exception as e:                       # file hỏng / parse lỗi → không crash server
            raise HTTPException(422, f"không đọc được tài liệu: {type(e).__name__}: {e}") from e
        return r.to_dict()

    @app.get("/documents")
    def documents() -> list[dict]:
        """Tài liệu đã ingest."""
        return store().list_documents()

    @app.post("/rag/query")
    def rag_query(q: QueryIn) -> dict:
        """Top-k chunk theo truy vấn (lọc doc_id tuỳ chọn)."""
        if q.doc_id and store().get_document(q.doc_id) is None:
            raise HTTPException(404, f"không có tài liệu {q.doc_id}")
        hits = store().search(q.query, q.k, {"doc_id": q.doc_id} if q.doc_id else None)
        return {"query": q.query, "k": q.k, "results": [c.to_dict() for c in hits]}

    @app.post("/rag/extract-steps")
    def rag_extract(body: ExtractIn) -> dict:
        """Trích bước chuẩn + trang nguồn (lưu outputs/extractions/<doc_id>.json)."""
        if store().get_document(body.doc_id) is None:
            raise HTTPException(404, f"không có tài liệu {body.doc_id}")
        prov = _provider(body.provider, body.llm_model)
        try:
            spec = extract_steps(body.doc_id, prov, store(), out_dir=extractions_dir, log=lambda *_: None)
        except LLMError as e:
            raise HTTPException(502, f"lỗi gọi LLM: {e}") from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return spec.model_dump()

    @app.post("/analyze/explain")
    def analyze_explain(body: ExplainIn) -> dict:
        """Giải thích findings của 1 run (tuỳ chọn RAG: citation = chunk_id)."""
        ctx = _run(body.run_id)
        fp = ctx.root / "findings" / "findings.json"
        if not fp.exists():
            raise HTTPException(404, f"run {ctx.run_id} chưa có findings/findings.json (chạy 06)")
        prov = _provider(body.provider, body.llm_model)
        retriever = None
        if body.use_rag:
            with lock:
                for f in (kb_path, sop_path):
                    store().ingest(f)
            retriever = store()
        try:
            res = explain(read_json(fp)["findings"], KnowledgeBase.from_files(kb_path, sop_path), prov,
                          retriever=retriever, k=body.k, log=lambda *_: None)
        except LLMError as e:
            raise HTTPException(502, f"lỗi gọi LLM: {e}") from e
        if body.save:
            write_json(ctx.path("explained", "explained.json"), res)
            ctx.update_manifest(llm={"provider": prov.name, "model": res["llm_model"]})
            ctx.record_stage("07_explain", via="api", provider=prov.name, model=res["llm_model"],
                             prompt=(EXPLAIN_RAG_V1 if retriever else EXPLAIN_V1).id, pass_rate=res["pass_rate"])
        return {"run_id": ctx.run_id, "saved": body.save, **res}

    @app.get("/runs")
    def runs() -> list[dict]:
        """Các run (mới nhất cuối), tóm tắt từ manifest."""
        return [{"run_id": m.get("run_id"), "created_at": m.get("created_at"), "source": m.get("source"),
                 "pose_model": m.get("pose_model"), "llm": m.get("llm"), "tag": m.get("tag"),
                 "n_trials": len(m.get("trials") or {})} for m in run_io.list_runs()]

    @app.get("/runs/{run_id}/metrics")
    def run_metrics(run_id: str) -> dict:
        """metrics.json của run."""
        return _artifact(run_id, "metrics", "metrics.json")

    @app.get("/runs/{run_id}/findings")
    def run_findings(run_id: str) -> dict:
        """findings.json của run."""
        return _artifact(run_id, "findings", "findings.json")

    @app.get("/runs/{run_id}/explained")
    def run_explained(run_id: str) -> dict:
        """explained.json của run."""
        return _artifact(run_id, "explained", "explained.json")

    return app


app = create_app()

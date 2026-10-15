"""CLI RAG (store mặc định data/rag_store, env KAIZEN_RAG_STORE; embedder env KAIZEN_EMBEDDER, mặc định e5).

    python -m kaizen.rag ingest tests/fixtures/sop_sample.pdf config/kaizen_kb.md [--force]
    python -m kaizen.rag list
    python -m kaizen.rag query "lắp bánh răng hành tinh" [-k 6] [--doc DOC-xxxxxx]
    python -m kaizen.rag extract-steps DOC-xxxxxx [--provider dry-run|gemini|anthropic|openai] [--llm-model M]
    python -m kaizen.rag to-station DOC-xxxxxx [--base config/station.yaml] [--out draft.yaml] [--provider ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kaizen.llm.providers import PROVIDER_NAMES, MissingAPIKeyError, get_provider
from kaizen.rag.extract import OUT_DIR, extract_steps, load_extraction, to_station
from kaizen.rag.store import DocumentStore


def main(argv: list[str] | None = None) -> int:
    """Entry point `python -m kaizen.rag`."""
    ap = argparse.ArgumentParser(prog="python -m kaizen.rag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help="thư mục store (mặc định env KAIZEN_RAG_STORE / data/rag_store)")
    ap.add_argument("--embedder", default=None, help="e5 | hash | st:<model> | openai | gemini")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("files", nargs="+")
    i.add_argument("--force", action="store_true")
    sub.add_parser("list")
    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=6)
    q.add_argument("--doc", default=None)
    for name in ("extract-steps", "to-station"):
        e = sub.add_parser(name)
        e.add_argument("doc_id")
        e.add_argument("--provider", choices=PROVIDER_NAMES, default=None,
                       help="mặc định env LLM_PROVIDER, rồi gemini" + (" (to-station: chỉ khi chưa có extraction)"
                                                                       if name == "to-station" else ""))
        e.add_argument("--llm-model", default=None)
        if name == "to-station":
            e.add_argument("--base", default="config/station.yaml")
            e.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    store = DocumentStore(a.store, a.embedder)

    if a.cmd == "ingest":
        for f in a.files:
            r = store.ingest(f, force=a.force)
            print(json.dumps(r.to_dict(), ensure_ascii=False))
        return 0
    if a.cmd == "list":
        for d in store.list_documents():
            print(f"{d['doc_id']}  {d['pages']:>3} trang  {d['chunks']:>4} chunk  {d['filename']}  — {d['title']}"
                  + (f"  (needs_ocr: {d['needs_ocr_pages']})" if d.get("needs_ocr_pages") else ""))
        return 0
    if a.cmd == "query":
        for c in store.search(a.query, a.k, {"doc_id": a.doc} if a.doc else None):
            print(f"{c.score:.3f}  {c.chunk_id}  p{c.page}  [{c.section}]\n        {c.text[:160]!r}")
        return 0
    try:
        if a.cmd == "to-station" and Path(OUT_DIR, f"{a.doc_id}.json").exists() and not a.provider:
            spec = load_extraction(a.doc_id)
        else:
            spec = extract_steps(a.doc_id, get_provider(a.provider, a.llm_model), store)
    except (MissingAPIKeyError, KeyError, ValueError) as e:
        print(f"[rag] {e}", file=sys.stderr)
        return 1
    if a.cmd == "extract-steps":
        for s in spec.steps:
            print(f"{s.order:>2}. {s.name}  (trang {s.page}, std {s.std_time_s}, khay {s.zone_hint})"
                  + (f"  ⚠ {s.warnings}" if s.warnings else ""))
        for w in spec.warnings:
            print(f"[rag] ⚠ {w}")
        return 0
    out = to_station(spec, a.base, a.out)
    print(f"[rag] bản nháp station.yaml ({len(spec.steps)} bước) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

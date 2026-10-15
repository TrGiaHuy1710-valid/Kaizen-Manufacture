"""[07] findings + KB/SOP → explained.json (LLM giải thích, code kiểm chứng). Wrapper mỏng → kaizen.llm.explain.

Provider: --provider anthropic|gemini|openai|dry-run (mặc định env LLM_PROVIDER, không có thì gemini);
  model: --llm-model > env LLM_MODEL > env ANTHROPIC_MODEL / GEMINI_MODEL / OPENAI_MODEL > mặc định
  (claude-sonnet-4-6 / gemini-3.8-flash / gpt-4.1-mini). Key chỉ qua env: ANTHROPIC_API_KEY, GEMINI_API_KEY,
  OPENAI_API_KEY — thiếu key → báo tên biến và dừng ngay. dry-run: explanation giả, không cần key/mạng.
LLM nhận toàn bộ kaizen_kb.md + sop.md + findings (prompt EXPLAIN_V1), trả JSON array
[{finding_id, observation, hypothesis, recommendation, citations}]. LLM CHỈ giải thích — không tạo số.

Parse lượng thứ (kaizen/llm/parsing.py, chung mọi provider): bỏ code fence/lời dẫn, bọc object đơn, điền key thiếu.
Không parse được → hỏi lại 1 lần "chỉ trả JSON array"; vẫn hỏng → validation: parse_failed (pipeline KHÔNG dừng).
VALIDATOR (kaizen/llm/validator.py): citation (chuẩn hoá KB13 → KB-13) ∈ KB/SOP; mọi số trong
observation+hypothesis+recommendation ∈ whitelist của finding. Fail → gọi lại 1 lần kèm thông báo lỗi.

    python src/07_explain.py --provider dry-run
    GEMINI_API_KEY=... python src/07_explain.py --run latest --provider gemini [--llm-model gemini-3.8-flash]
    ANTHROPIC_API_KEY=... python src/07_explain.py --run latest --provider anthropic
    OPENAI_API_KEY=... python src/07_explain.py --run latest --provider openai
RAG (--use-rag): kaizen_kb.md + sop.md được ingest vào RAG store như tài liệu (idempotent); với mỗi finding truy xuất
k chunk (cả tài liệu đã upload, vd SOP PDF), prompt EXPLAIN_RAG_V1 bắt buộc trích dẫn bằng chunk_id
("DOC-xxxxxx:pN:cM"); validator kiểm chunk_id TỒN TẠI trong store + số liệu như cũ.
    python src/07_explain.py --run latest --use-rag --dry-run --force          # store data/rag_store, embedder e5
    KAIZEN_EMBEDDER=hash python src/07_explain.py --run latest --use-rag --provider gemini --force
So sánh pass-rate 3 provider: mỗi provider 1 run (02–06 tất định, nhanh) rồi compare_runs:
    for p in openai gemini anthropic; do bash run_all.sh --provider $p --tag $p; done
    python src/tools/compare_runs.py --last 3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen import llm  # noqa: E402
from kaizen.core.cli import base_parser, setup, stage_file  # noqa: E402
from kaizen.core.io import can_write, read_json, write_json  # noqa: E402
from kaizen.llm.explain import (KnowledgeBase, ask_parsed, dry_run_explain, explanation_fields,  # noqa: E402,F401
                                pick)
from kaizen.llm.prompts import DRY_CITES, EXPLAIN_RAG_V1, EXPLAIN_V1, ONLY_JSON, SYSTEM, user_prompt  # noqa: E402,F401
from kaizen.llm.validator import FIELDS, ParseError, load_ids, parse_explanations, validate  # noqa: E402,F401

TAG = "07"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--findings", default=None, help="mặc định <run>/findings/findings.json")
    ap.add_argument("--kb", default="config/kaizen_kb.md")
    ap.add_argument("--sop", default="config/sop.md")
    ap.add_argument("--out", default=None, help="mặc định <run>/explained/explained.json")
    ap.add_argument("--provider", choices=llm.PROVIDER_NAMES, default=os.environ.get("LLM_PROVIDER") or "gemini")
    ap.add_argument("--llm-model", "--model", dest="llm_model", default=None,
                    help="mặc định: env LLM_MODEL, rồi env <PROVIDER>_MODEL, rồi model mặc định của provider")
    ap.add_argument("--dry-run", action="store_true", help="= --provider dry-run")
    ap.add_argument("--no-retry", action="store_true", help="không gọi lại khi validator fail")
    ap.add_argument("--use-rag", action="store_true", help="truy xuất chunk từ RAG store, citation = chunk_id")
    ap.add_argument("--rag-store", default=None, help="mặc định env KAIZEN_RAG_STORE hoặc data/rag_store")
    ap.add_argument("--rag-k", type=int, default=4, help="số chunk truy xuất mỗi finding")
    ap.add_argument("--embedder", default=None, help="e5 (mặc định, env KAIZEN_EMBEDDER) | hash | st:<model> | openai")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    try:
        provider = llm.get_provider("dry-run" if a.dry_run else a.provider, a.llm_model)
    except llm.MissingAPIKeyError as e:
        print(f"[{TAG}] {e}", file=sys.stderr)
        sys.exit(1)
    out = stage_file(ctx, a.out, "explained", "explained.json", "outputs/findings/explained.json")
    if not can_write(out, a.force, TAG):
        return
    doc = read_json(stage_file(ctx, a.findings, "findings", "findings.json", "outputs/findings/findings.json"))
    findings = doc["findings"]
    kb = KnowledgeBase.from_files(a.kb, a.sop)
    store = None
    if a.use_rag:
        from kaizen.rag.store import DocumentStore
        store = DocumentStore(a.rag_store, a.embedder)
        for f in (a.kb, a.sop):
            r = store.ingest(f)
            print(f"[{TAG}] RAG: {r.doc_id} {r.filename} ({r.chunks} chunk{', đã có' if r.already_present else ''})")
        print(f"[{TAG}] RAG store: {store.describe()}")
    try:
        res = llm.explain(findings, kb, provider, retriever=store, k=a.rag_k, retry=not a.no_retry)
    except llm.LLMError as e:
        print(f"[{TAG}] lỗi gọi LLM: {e}", file=sys.stderr)
        sys.exit(1)
    items = res["explanations"]
    n, n_pass, n_pass1 = res["n_findings"], res["n_passed"], res["n_passed_first_attempt"]
    write_json(out, res)
    if ctx and not a.out:   # chỉ ghi manifest khi output nằm trong run
        ctx.update_manifest(llm={"provider": provider.name, "model": res["llm_model"]})
        ctx.record_stage("07_explain", provider=provider.name, model=res["llm_model"],
                         prompt=(EXPLAIN_RAG_V1 if store else EXPLAIN_V1).id,
                         **({"rag": store.describe()} if store else {}),
                         pass_rate=res["pass_rate"], pass_rate_first_attempt=res["pass_rate_first_attempt"],
                         n_parse_failed=res["n_parse_failed"])
    for x in items:
        msg = x["final_check"].get("message") if x["validation"] != "passed" else ""
        tries = len(x["attempts"])
        print(f"[{TAG}] {x['finding_id']}: {x['validation']}" + (f" (sau {tries} lần)" if tries > 1 else "")
              + (f" — {msg}" if msg else ""))
    print(f"[{TAG}] provider={provider.name} model={res['llm_model']}: pass {n_pass}/{n} "
          f"(lần 1: {n_pass1}/{n}) → {out}")


if __name__ == "__main__":
    main()

"""[07] findings + KB/SOP → explained.json (LLM giải thích, code kiểm chứng).

Provider: --provider anthropic|gemini|dry-run (mặc định env LLM_PROVIDER, không có thì gemini).
  anthropic: claude-sonnet-4-6, env ANTHROPIC_API_KEY · gemini: env GEMINI_MODEL / gemini-3.8-flash, env GEMINI_API_KEY
  dry-run  : explanation giả, không cần key/mạng.
LLM nhận toàn bộ kaizen_kb.md + sop.md + findings, trả JSON array
[{finding_id, observation, hypothesis, recommendation, citations}]. LLM CHỈ giải thích — không tạo số.

Parse lượng thứ (src/llm_validate.py): bỏ code fence/lời dẫn, bọc object đơn, điền key thiếu (format_warning).
Không parse được → hỏi lại 1 lần "chỉ trả JSON array"; vẫn hỏng → validation: parse_failed (pipeline KHÔNG dừng).
VALIDATOR: citation (chuẩn hoá KB13 → KB-13) ∈ KB/SOP; mọi số trong observation+hypothesis+recommendation
∈ whitelist của finding. Fail → gọi lại 1 lần kèm thông báo lỗi; kết quả cuối: passed | failed_after_retry.

    python src/07_explain.py --provider dry-run
    GEMINI_API_KEY=... python src/07_explain.py --provider gemini [--llm-model gemini-3.8-flash]
    ANTHROPIC_API_KEY=... python src/07_explain.py --provider anthropic
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, read_json, setup, stage_file, write_json  # noqa: E402
import llm_providers as llm  # noqa: E402
from llm_validate import FIELDS, ParseError, load_ids, parse_explanations, validate  # noqa: E402

TAG = "07"

SYSTEM = """Bạn là kỹ sư Kaizen/IE hỗ trợ phân tích trạm lắp ráp. Bạn CHỈ giải thích các finding đã được code tính sẵn.
Quy tắc bắt buộc:
1. observation chỉ được dùng các con số CÓ SẴN trong JSON của finding đó, chép đúng giá trị (được làm tròn 1 chữ số thập phân). KHÔNG tự tính, cộng, trừ, quy đổi hay tạo số mới. Không có số phù hợp thì diễn đạt bằng lời.
2. hypothesis là giả thuyết nguyên nhân, nêu rõ là giả thuyết; cũng không được đưa số mới.
3. recommendation: đề xuất cải tiến cụ thể, ưu tiên theo ECRS; không đưa số mới.
4. citations: chỉ dùng ID có trong tài liệu (dạng KB-xx hoặc SOP-xx), ít nhất 1 ID cho mỗi finding.
5. Viết tiếng Việt, ngắn gọn. Dùng dấu chấm cho số thập phân.
6. Khi nhắc tên chỉ số thống kê hãy viết P90/IoU/CV nguyên dạng, không diễn giải thành số.
7. Chỉ trả về JSON array, không kèm giải thích hay markdown:
[{"finding_id": "F01", "observation": "...", "hypothesis": "...", "recommendation": "...", "citations": ["KB-03"]}]"""

ONLY_JSON = "Response trước không parse được thành JSON. Chỉ trả JSON array, không markdown, không lời dẫn."

DRY_CITES = {"slow_step": ["KB-10", "KB-12"], "extra_pick": ["KB-04", "KB-07"], "long_idle": ["KB-03", "KB-16"],
             "sequence_deviation": ["KB-14", "SOP-20"], "hand_imbalance": ["KB-05", "KB-15"],
             "high_variation": ["KB-10", "KB-01"], "wrist_posture": ["KB-13", "KB-01"]}


def dry_run_explain(findings: list[dict]) -> list[dict]:
    """Explanation giả, chỉ dùng số có trong finding → test luồng không cần mạng."""
    out = []
    for f in findings:
        e = f["evidence"]
        out.append({"finding_id": f["id"],
                    "observation": f"[dry-run] {e['metric']} = {e['value']} {e['unit']} (rule {f['rule']}).",
                    "hypothesis": "[dry-run] Giả thuyết nguyên nhân sẽ do LLM sinh khi chạy thật.",
                    "recommendation": "[dry-run] Kỹ sư IE xem lại đoạn video bằng chứng và áp dụng ECRS.",
                    "citations": DRY_CITES.get(f["type"], ["KB-01"])})
    return out


def user_prompt(findings: list[dict], kb_text: str, sop_text: str) -> str:
    return (f"<kaizen_kb>\n{kb_text}\n</kaizen_kb>\n\n<sop>\n{sop_text}\n</sop>\n\n"
            f"<findings>\n{json.dumps(findings, ensure_ascii=False, indent=1)}\n</findings>\n\n"
            "Giải thích TỪNG finding theo đúng định dạng JSON đã quy định.")


def ask_parsed(provider: str, model: str, messages: list[dict], calls: list[dict]) -> tuple[list[dict] | None, str, list[str]]:
    """Gọi LLM + parse; không parse được thì hỏi lại 1 lần. Trả (items|None, raw_text_cuối, lỗi_parse)."""
    errs = []
    text, usage = llm.chat(provider, model, SYSTEM, messages)
    calls.append(usage)
    try:
        return parse_explanations(text), text, errs
    except ParseError as e:
        errs.append(str(e))
        print(f"[{TAG}] parse lỗi ({e}) → hỏi lại 1 lần chỉ JSON", file=sys.stderr)
    text2, usage = llm.chat(provider, model, SYSTEM, messages + [{"role": "assistant", "content": text},
                                                                 {"role": "user", "content": ONLY_JSON}])
    calls.append(usage)
    try:
        return parse_explanations(text2), text2, errs
    except ParseError as e:
        errs.append(str(e))
        return None, text2, errs


def pick(items: list[dict] | None, fid: str) -> dict | None:
    return next((x for x in items or [] if x.get("finding_id") == fid), None)


def explanation_fields(x: dict | None) -> dict:
    x = x or {}
    out = {k: x.get(k, "") for k in ("finding_id",) + FIELDS}
    out["citations"] = x.get("citations", [])
    if x.get("format_warning"):
        out["format_warning"] = x["format_warning"]
    return out


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--findings", default=None, help="mặc định <run>/findings/findings.json")
    ap.add_argument("--kb", default="config/kaizen_kb.md")
    ap.add_argument("--sop", default="config/sop.md")
    ap.add_argument("--out", default=None, help="mặc định <run>/explained/explained.json")
    ap.add_argument("--provider", choices=llm.PROVIDERS, default=os.environ.get("LLM_PROVIDER") or "gemini")
    ap.add_argument("--llm-model", "--model", dest="llm_model", default=None,
                    help="mặc định: claude-sonnet-4-6 (anthropic) / env GEMINI_MODEL hoặc gemini-3.8-flash (gemini)")
    ap.add_argument("--dry-run", action="store_true", help="= --provider dry-run")
    ap.add_argument("--no-retry", action="store_true", help="không gọi lại khi validator fail")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    provider = "dry-run" if a.dry_run else a.provider
    model = a.llm_model or llm.default_model(provider)
    out = stage_file(ctx, a.out, "explained", "explained.json", "outputs/findings/explained.json")
    if not can_write(out, a.force, TAG):
        return
    doc = read_json(stage_file(ctx, a.findings, "findings", "findings.json", "outputs/findings/findings.json"))
    findings = doc["findings"]
    kb_ids = set(load_ids(Path(a.kb), Path(a.sop)))
    by_id = {f["id"]: f for f in findings}

    calls: list[dict] = []
    raw = {}
    if not findings:
        first = []
    elif provider == "dry-run":
        first = dry_run_explain(findings)
    else:
        err = llm.check_credentials(provider)
        if err:
            print(f"[{TAG}] {err} — export key hoặc chạy --provider dry-run", file=sys.stderr)
            sys.exit(1)
        messages = [{"role": "user", "content": user_prompt(findings, Path(a.kb).read_text(encoding="utf-8"),
                                                            Path(a.sop).read_text(encoding="utf-8"))}]
        try:
            first, raw["attempt1"], perr = ask_parsed(provider, model, messages, calls)
        except llm.LLMError as e:
            print(f"[{TAG}] lỗi gọi LLM: {e}", file=sys.stderr)
            sys.exit(1)
        if perr:
            raw["parse_errors_attempt1"] = perr

    # Lần 1: validate
    results = {}
    for f in findings:
        x = pick(first, f["id"])
        if first is None:
            results[f["id"]] = {"attempts": [], "final": explanation_fields(None), "validation": "parse_failed",
                                "final_check": {"message": "không parse được response (xem raw_responses)"}}
            continue
        v = validate(x, f, kb_ids) if x else {"status": "failed", "unknown_numbers": [], "bad_citations": [],
                                               "citations_normalized": [], "message": "LLM không trả lời finding này"}
        results[f["id"]] = {"attempts": [{"attempt": 1, "explanation": explanation_fields(x), "check": v}],
                            "final": explanation_fields(x), "final_check": v,
                            "validation": "passed" if v["status"] == "passed" else "failed"}

    # Lần 2: retry có phản hồi cho finding fail (chỉ khi gọi LLM thật)
    failed = [fid for fid, r_ in results.items() if r_["validation"] == "failed"]
    if failed and provider != "dry-run" and not a.no_retry:
        fb = "\n".join(f"- {fid}: {results[fid]['final_check']['message']}" for fid in failed)
        retry_msg = (f"Validator phát hiện lỗi trong các explanation sau:\n{fb}\n\n"
                     "Hãy sửa đúng các lỗi trên, chỉ dùng số có trong JSON của finding tương ứng, chỉ dùng citation có "
                     f"trong tài liệu. Trả JSON thuần (array) chỉ gồm các finding: {', '.join(failed)}.")
        msgs = messages + [{"role": "assistant", "content": raw["attempt1"]}, {"role": "user", "content": retry_msg}]
        print(f"[{TAG}] retry {len(failed)} finding fail: {failed}")
        try:
            second, raw["attempt2"], perr = ask_parsed(provider, model, msgs, calls)
            if perr:
                raw["parse_errors_attempt2"] = perr
        except llm.LLMError as e:
            print(f"[{TAG}] lỗi gọi LLM khi retry: {e}", file=sys.stderr)
            second = None
        for fid in failed:
            x = pick(second, fid)
            v = validate(x, by_id[fid], kb_ids) if x else {
                "status": "failed", "unknown_numbers": [], "bad_citations": [], "citations_normalized": [],
                "message": "retry không trả lời finding này" if second is not None else "retry không parse được"}
            r_ = results[fid]
            r_["attempts"].append({"attempt": 2, "explanation": explanation_fields(x), "check": v})
            if x:
                r_["final"], r_["final_check"] = explanation_fields(x), v
            r_["validation"] = "passed" if v["status"] == "passed" else "failed_after_retry"

    items = [{"finding_id": fid, **r_} for fid, r_ in results.items()]
    n = len(items)
    n_pass = sum(x["validation"] == "passed" for x in items)
    n_pass1 = sum(bool(x["attempts"]) and x["attempts"][0]["check"]["status"] == "passed" for x in items)
    res = {"provider": provider, "mode": "dry-run" if provider == "dry-run" else ("none" if not findings else "llm"),
           "llm_model": model if provider != "dry-run" else None, "calls": calls,
           "n_findings": n, "n_passed": n_pass, "pass_rate": round(n_pass / n, 3) if n else None,
           "n_passed_first_attempt": n_pass1, "pass_rate_first_attempt": round(n_pass1 / n, 3) if n else None,
           "n_retried": sum(len(x["attempts"]) > 1 for x in items),
           "n_parse_failed": sum(x["validation"] == "parse_failed" for x in items),
           "validator": "citation (chuẩn hoá) ∈ KB/SOP; số trong observation+hypothesis+recommendation ∈ whitelist "
                        "của finding (giá trị, số trong tên trường/chuỗi, số phổ quát, bước 1–12; làm tròn 0.1)",
           "raw_responses": raw, "explanations": items}
    write_json(out, res)
    if ctx and not a.out:   # chỉ ghi manifest khi output nằm trong run
        ctx.update_manifest(llm={"provider": provider, "model": res["llm_model"]})
        ctx.record_stage("07_explain", provider=provider, model=res["llm_model"], pass_rate=res["pass_rate"],
                         pass_rate_first_attempt=res["pass_rate_first_attempt"], n_parse_failed=res["n_parse_failed"])
    for x in items:
        msg = x["final_check"].get("message") if x["validation"] != "passed" else ""
        tries = len(x["attempts"])
        print(f"[{TAG}] {x['finding_id']}: {x['validation']}" + (f" (sau {tries} lần)" if tries > 1 else "")
              + (f" — {msg}" if msg else ""))
    print(f"[{TAG}] provider={provider} model={res['llm_model']}: pass {n_pass}/{n} "
          f"(lần 1: {n_pass1}/{n}) → {out}")


if __name__ == "__main__":
    main()

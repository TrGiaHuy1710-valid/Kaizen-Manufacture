"""Giải thích findings bằng LLM (stage 07) — LLM CHỈ giải thích, validator (code) kiểm số + trích dẫn.

Luồng: prompt (KB + SOP + findings, hoặc chunk RAG truy xuất theo từng finding) → parse lượng thứ (hỏng → hỏi lại
1 lần "chỉ JSON") → validate từng finding → finding fail được gọi lại 1 lần kèm thông báo lỗi validator →
explained.json {attempts, final, validation: passed | failed_after_retry | parse_failed}, pass_rate,
pass_rate_first_attempt. Chế độ RAG: citation = chunk_id, hợp lệ khi tồn tại trong store (retriever.has_chunk).
version="v2" (07 mặc định): prompt EXPLAIN_V2 / EXPLAIN_V2_KB + <context> (bảng bước rút gọn id/dur/std/lệch + zone
— số trong context của trial được phép dùng), schema thêm ecrs + remeasure, validator kiểm thêm định dạng (FORMAT_V2).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from kaizen.llm.prompts import (DRY_CITES, EXPLAIN_RAG_V1, EXPLAIN_V1, EXPLAIN_V2, EXPLAIN_V2_KB, ONLY_JSON_V1,
                                RETRY_FEEDBACK_V1, findings_json)
from kaizen.llm.providers import MAX_TOKENS, LLMError, LLMProvider
from kaizen.llm.validator import FIELDS, FORMAT_V2, V2_FIELDS, ParseError, load_ids, parse_explanations, validate

TAG = "07"
VALIDATOR_DESC = ("citation (chuẩn hoá) ∈ KB/SOP; số trong observation+hypothesis+recommendation ∈ whitelist "
                  "của finding (giá trị, số trong tên trường/chuỗi, số phổ quát, bước 1–12; làm tròn 0.1)")
VALIDATOR_DESC_RAG = ("citation = chunk_id tồn tại trong RAG store (hoặc KB/SOP-xx); số trong "
                      "observation+hypothesis+recommendation ∈ whitelist của finding (như chế độ thường)")
TYPE_QUERY = {
    "slow_step": "bước chậm bất thường, biến động thời gian bước, chuẩn hoá công việc, median MAD",
    "extra_pick": "lấy dư linh kiện, thao tác thừa, khay cấp liệu, số lần lấy chuẩn",
    "long_idle": "chờ đợi, tay đứng yên, lãng phí chờ, idle",
    "sequence_deviation": "sai thứ tự, thiếu bước, poka-yoke, thứ tự bước bắt buộc",
    "hand_imbalance": "cân bằng tải hai tay, một tay rảnh, đồ gá giữ chi tiết",
    "high_variation": "biến động cycle time, Mura, không đều, chuẩn hoá công việc",
    "wrist_posture": "tư thế cổ tay trung tính gập 30 độ, Muri, ergonomic, quá tải",
    "std_time_slow": "thời gian chuẩn của bước, chênh lệch so với chuẩn, standardized work",
    "std_time_fast": "thời gian chuẩn của bước, bỏ sót thao tác, cập nhật chuẩn",
}
VALIDATOR_DESC_V2 = " + định dạng: quan sát ≤2 câu, giả thuyết 1 câu, đề xuất ≤2 câu, đo lại ≤12 từ (≤20 tiếng), ecrs ∈ E|C|R|S"
DRY_ECRS = {"extra_pick": "E", "long_idle": "R", "hand_imbalance": "R", "wrist_posture": "R", "std_time_slow": "S",
            "std_time_fast": "S", "slow_step": "S", "high_variation": "S", "sequence_deviation": "E"}


@dataclass
class KnowledgeBase:
    """KB tĩnh: {ID: đoạn} của [KB-xx]/[SOP-xx] + toàn văn để đưa vào prompt EXPLAIN_V1."""
    ids: dict[str, str] = field(default_factory=dict)
    kb_text: str = ""
    sop_text: str = ""

    @classmethod
    def from_files(cls, kb: str | Path = "config/kaizen_kb.md", sop: str | Path = "config/sop.md") -> "KnowledgeBase":
        """Đọc kaizen_kb.md + sop.md."""
        return cls(load_ids(Path(kb), Path(sop)), Path(kb).read_text(encoding="utf-8"),
                   Path(sop).read_text(encoding="utf-8"))


class Retriever(Protocol):
    """Nguồn chunk cho chế độ RAG (kaizen.rag.store.DocumentStore thoả interface này)."""

    def search(self, query: str, k: int = 6, filter: dict | None = None) -> list[Any]:
        """Top-k chunk (có chunk_id, title, page, section, text, score)."""

    def has_chunk(self, chunk_id: str) -> bool:
        """chunk_id có trong store không."""


def dry_run_explain(findings: list[dict], rag_cites: dict[str, list[str]] | None = None,
                    v2: bool = False) -> list[dict]:
    """Explanation giả, chỉ dùng số có trong finding → test luồng không cần mạng (RAG: trích chunk top đầu)."""
    out = []
    for f in findings:
        e = f["evidence"]
        cites = (rag_cites or {}).get(f["id"])[:2] if rag_cites and rag_cites.get(f["id"]) else \
            DRY_CITES.get(f["type"], ["KB-01"])
        out.append({"finding_id": f["id"],
                    "observation": f"[dry-run] {e['metric']} = {e['value']} {e['unit']} (rule {f['rule']}).",
                    "hypothesis": "[dry-run] Giả thuyết nguyên nhân sẽ do LLM sinh khi chạy thật.",
                    "recommendation": "[dry-run] Kỹ sư IE xem lại đoạn video bằng chứng và áp dụng ECRS.",
                    **({"ecrs": DRY_ECRS.get(f["type"], "S"),
                        "remeasure": "[dry-run] Đo lại chỉ số này sau cải tiến"} if v2 else {}),
                    "citations": cites})
    return out


def ask_parsed(provider: LLMProvider, system: str, user: str, history: list[dict],
               calls: list[dict], extra: tuple[str, ...] = ()) -> tuple[list[dict] | None, str, list[str]]:
    """Gọi LLM + parse; không parse được thì hỏi lại 1 lần. Trả (items|None, raw_text_cuối, lỗi_parse)."""
    errs = []
    res = provider.complete(system, user, history=history, max_tokens=MAX_TOKENS)
    calls.append(res.usage)
    try:
        return parse_explanations(res.text, extra), res.text, errs
    except ParseError as e:
        errs.append(str(e))
        print(f"[{TAG}] parse lỗi ({e}) → hỏi lại 1 lần chỉ JSON", file=sys.stderr)
    res2 = provider.complete(system, ONLY_JSON_V1.user, max_tokens=MAX_TOKENS,
                             history=history + [{"role": "user", "content": user},
                                                {"role": "assistant", "content": res.text}])
    calls.append(res2.usage)
    try:
        return parse_explanations(res2.text, extra), res2.text, errs
    except ParseError as e:
        errs.append(str(e))
        return None, res2.text, errs


def pick(items: list[dict] | None, fid: str) -> dict | None:
    """Explanation của finding_id trong list (None nếu thiếu)."""
    return next((x for x in items or [] if x.get("finding_id") == fid), None)


def explanation_fields(x: dict | None, extra: tuple[str, ...] = ()) -> dict:
    """Chuẩn hoá explanation về đúng key (finding_id, 3 trường[, ecrs, remeasure], citations[, format_warning])."""
    x = x or {}
    out = {k: x.get(k, "") for k in ("finding_id",) + FIELDS + tuple(extra)}
    out["citations"] = x.get("citations", [])
    if x.get("format_warning"):
        out["format_warning"] = x["format_warning"]
    return out


def finding_query(f: dict) -> str:
    """Câu truy vấn RAG cho 1 finding: chủ đề theo type + metric + ngữ cảnh (tên bước, zone, tay)."""
    e = f.get("evidence", {})
    parts = [TYPE_QUERY.get(f.get("type", ""), f.get("type", "")), str(e.get("metric", ""))]
    parts += [str(e[k]) for k in ("step_name", "zone", "hand") if e.get(k)]
    return " · ".join(p for p in parts if p)


def retrieve_for_findings(findings: list[dict], retriever: Retriever, k: int = 4) -> tuple[dict, list]:
    """{finding_id: [chunk...]} + danh sách chunk khử trùng theo thứ tự xuất hiện."""
    per, seen, docs = {}, set(), []
    for f in findings:
        hits = retriever.search(finding_query(f), k=k)
        per[f["id"]] = hits
        for c in hits:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                docs.append(c)
    return per, docs


def render_documents(chunks: list) -> str:
    """Khối <documents>: mỗi chunk '[chunk_id] tiêu đề · trang · mục' + nội dung."""
    out = []
    for c in chunks:
        head = f"[{c.chunk_id}] {c.title} · trang {c.page}" + (f" · mục: {c.section}" if c.section else "")
        out.append(f"{head}\n{c.text}")
    return "\n\n".join(out)


def step_context(metrics: dict, standards: dict | None = None, trials: list[str] | None = None,
                 zones: list[str] | None = None) -> dict:
    """<context> cho EXPLAIN_V2: {steps: {trial: [{id, name, dur_s, std_s, dev_pct}]}, zones} — số do code tính."""
    std = {int(k): (v or {}).get("std_s") for k, v in ((standards or {}).get("std_times") or {}).items()}
    out = {}
    for st in trials or list(metrics.get("trials", {})):
        m = metrics.get("trials", {}).get(st)
        if not m:
            continue
        rows = []
        for row in sorted(m["steps"]["rows"], key=lambda x: x["step_id"]):
            s = std.get(int(row["step_id"]))
            rows.append({"id": row["step_id"], "name": row["name"], "dur_s": row["dur_s"], "std_s": s,
                         "dev_pct": round((row["dur_s"] - s) / s * 100, 1) if s else None})
        out[st] = rows
    return {"steps": out, "zones": list(zones or [])}


def _context_for(context: dict | None, f: dict) -> dict:
    """Phần context của 1 finding (bảng bước các trial của nó, tối đa 3) — dùng cho prompt + whitelist số."""
    if not context:
        return {}
    trials = (f.get("evidence", {}).get("trials") or [])[:3]
    return {"steps": {t: context.get("steps", {}).get(t, []) for t in trials}, "zones": context.get("zones", [])}


def explain(findings: list[dict], kb: KnowledgeBase | None, provider: LLMProvider, *,
            retriever: Retriever | None = None, k: int = 4, retry: bool = True,
            log: Callable[[str], None] = print, version: str = "v1", context: dict | None = None) -> dict:
    """findings → nội dung explained.json. retriever ≠ None → chế độ RAG (citation chunk_id).

    version: "v1" (EXPLAIN_V1 / EXPLAIN_RAG_V1, 3 trường) | "v2" (EXPLAIN_V2 / EXPLAIN_V2_KB + context + ecrs/remeasure,
    validator kiểm định dạng). context: step_context(...) — chỉ dùng ở v2.
    Raise LLMError nếu lần gọi đầu thất bại (lỗi khi retry chỉ ghi nhận, không dừng).
    """
    if version not in ("v1", "v2"):
        raise ValueError(f"version không hỗ trợ: {version!r} (v1 | v2)")
    v2 = version == "v2"
    extra = V2_FIELDS if v2 else ()
    fmt = FORMAT_V2 if v2 else None
    kb = kb or KnowledgeBase()
    kb_ids = set(kb.ids)
    by_id = {f["id"]: f for f in findings}
    citation_exists = retriever.has_chunk if retriever is not None else None
    ctx_trials = sorted({t for f in findings for t in (f.get("evidence", {}).get("trials") or [])[:3]})
    ctx_json = findings_json({"steps": {t: (context or {}).get("steps", {}).get(t, []) for t in ctx_trials},
                              "zones": (context or {}).get("zones", [])}) if v2 else ""
    rag, per = None, {}
    if retriever is not None and findings:
        per, docs = retrieve_for_findings(findings, retriever, k)
        tmpl = EXPLAIN_V2 if v2 else EXPLAIN_RAG_V1
        rag = {"prompt": tmpl.id, "k": k,
               "per_finding": {fid: [{"chunk_id": c.chunk_id, "score": round(float(c.score), 4), "page": c.page,
                                      "title": c.title} for c in cs] for fid, cs in per.items()}}
        prompt_findings = [{**f, "relevant_chunks": [c.chunk_id for c in per[f["id"]]]} for f in findings]
        kw = {"context_json": ctx_json} if v2 else {}
        user0 = tmpl.render(documents=render_documents(docs), findings_json=findings_json(prompt_findings), **kw)
    else:
        tmpl = EXPLAIN_V2_KB if v2 else EXPLAIN_V1
        kw = {"context_json": ctx_json} if v2 else {}
        user0 = tmpl.render(kb_text=kb.kb_text, sop_text=kb.sop_text, findings_json=findings_json(findings), **kw)

    calls: list[dict] = []
    raw = {}
    if not findings:
        first = []
    elif provider.name == "dry-run":
        first = dry_run_explain(findings, {fid: [c.chunk_id for c in cs] for fid, cs in per.items()} if rag else None,
                                v2=v2)
    else:
        first, raw["attempt1"], perr = ask_parsed(provider, tmpl.system, user0, [], calls, extra)
        if perr:
            raw["parse_errors_attempt1"] = perr

    def check(x, f):
        vf = {**f, "context": _context_for(context, f)} if v2 and context else f
        return validate(x, vf, kb_ids, citation_exists, fmt)

    # Lần 1: validate
    results = {}
    for f in findings:
        x = pick(first, f["id"])
        if first is None:
            results[f["id"]] = {"attempts": [], "final": explanation_fields(None, extra), "validation": "parse_failed",
                                "final_check": {"message": "không parse được response (xem raw_responses)"}}
            continue
        v = check(x, f) if x else {"status": "failed", "unknown_numbers": [], "bad_citations": [],
                                   "citations_normalized": [], "message": "LLM không trả lời finding này"}
        results[f["id"]] = {"attempts": [{"attempt": 1, "explanation": explanation_fields(x, extra), "check": v}],
                            "final": explanation_fields(x, extra), "final_check": v,
                            "validation": "passed" if v["status"] == "passed" else "failed"}

    # Lần 2: retry có phản hồi cho finding fail (chỉ khi gọi LLM thật)
    failed = [fid for fid, r_ in results.items() if r_["validation"] == "failed"]
    if failed and provider.name != "dry-run" and retry:
        fb = "\n".join(f"- {fid}: {results[fid]['final_check']['message']}" for fid in failed)
        retry_msg = RETRY_FEEDBACK_V1.render(feedback=fb, finding_ids=", ".join(failed))
        hist = [{"role": "user", "content": user0}, {"role": "assistant", "content": raw["attempt1"]}]
        log(f"[{TAG}] retry {len(failed)} finding fail: {failed}")
        try:
            second, raw["attempt2"], perr = ask_parsed(provider, tmpl.system, retry_msg, hist, calls, extra)
            if perr:
                raw["parse_errors_attempt2"] = perr
        except LLMError as e:
            print(f"[{TAG}] lỗi gọi LLM khi retry: {e}", file=sys.stderr)
            second = None
        for fid in failed:
            x = pick(second, fid)
            v = check(x, by_id[fid]) if x else {
                "status": "failed", "unknown_numbers": [], "bad_citations": [], "citations_normalized": [],
                "message": "retry không trả lời finding này" if second is not None else "retry không parse được"}
            r_ = results[fid]
            r_["attempts"].append({"attempt": 2, "explanation": explanation_fields(x, extra), "check": v})
            if x:
                r_["final"], r_["final_check"] = explanation_fields(x, extra), v
            r_["validation"] = "passed" if v["status"] == "passed" else "failed_after_retry"

    items = [{"finding_id": fid, **r_} for fid, r_ in results.items()]
    n = len(items)
    n_pass = sum(x["validation"] == "passed" for x in items)
    n_pass1 = sum(bool(x["attempts"]) and x["attempts"][0]["check"]["status"] == "passed" for x in items)
    dry = provider.name == "dry-run"
    res = {"provider": provider.name, "mode": "dry-run" if dry else ("none" if not findings else "llm"),
           "llm_model": provider.model if not dry else None, "calls": calls,
           "n_findings": n, "n_passed": n_pass, "pass_rate": round(n_pass / n, 3) if n else None,
           "n_passed_first_attempt": n_pass1, "pass_rate_first_attempt": round(n_pass1 / n, 3) if n else None,
           "n_retried": sum(len(x["attempts"]) > 1 for x in items),
           "n_parse_failed": sum(x["validation"] == "parse_failed" for x in items),
           "validator": (VALIDATOR_DESC_RAG if retriever is not None else VALIDATOR_DESC) + (VALIDATOR_DESC_V2 if v2 else ""),
           "raw_responses": raw, "explanations": items}
    if v2:
        res["prompt"] = tmpl.id
    if retriever is not None:
        res["rag"] = rag or {"prompt": (EXPLAIN_V2 if v2 else EXPLAIN_RAG_V1).id, "k": k, "per_finding": {}}
    return res

"""[07] findings + KB/SOP → outputs/findings/explained.json (LLM giải thích, code kiểm chứng).

LLM (Gemini API qua google-genai, mặc định gemini-3.8-flash, env GEMINI_API_KEY) nhận toàn bộ kaizen_kb.md + sop.md +
findings.json, trả JSON thuần: [{finding_id, observation, hypothesis, recommendation, citations}].
LLM CHỈ giải thích — không tạo số.

VALIDATOR (bằng code, luôn chạy):
  a) mọi citation phải tồn tại trong KB/SOP ([KB-xx] / [SOP-xx]);
  b) mọi token số trong observation + hypothesis (bỏ số nằm trong mã citation) phải xuất hiện trong JSON
     của chính finding đó (so khớp sau khi làm tròn 1 chữ số thập phân).
Finding fail → validation: "failed" (giữ lại, báo tỷ lệ pass).

    python src/07_explain.py --dry-run        # explanation giả, không cần key/mạng
    export GEMINI_API_KEY=...
    python src/07_explain.py [--llm-model gemini-3.8-flash]   # gọi API thật
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, read_json, write_json  # noqa: E402

TAG = "07"
CITE_RE = re.compile(r"\[((?:KB|SOP)-\d+)\]")
CITE_TOKEN_RE = re.compile(r"\b(?:KB|SOP)-\d+\b")
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
DEFAULT_MODEL = "gemini-3.8-flash"

SYSTEM = """Bạn là kỹ sư Kaizen/IE hỗ trợ phân tích trạm lắp ráp. Bạn CHỈ giải thích các finding đã được code tính sẵn.
Quy tắc bắt buộc:
1. observation chỉ được dùng các con số CÓ SẴN trong JSON của finding đó, chép đúng giá trị (được làm tròn 1 chữ số thập phân). KHÔNG tự tính, cộng, trừ, quy đổi hay tạo số mới. Không có số phù hợp thì diễn đạt bằng lời.
2. hypothesis là giả thuyết nguyên nhân, nêu rõ là giả thuyết; cũng không được đưa số mới.
3. recommendation: đề xuất cải tiến cụ thể, ưu tiên theo ECRS; tránh con số.
4. citations: chỉ dùng ID có trong tài liệu (dạng KB-xx hoặc SOP-xx), ít nhất 1 ID cho mỗi finding.
5. Viết tiếng Việt, ngắn gọn. Dùng dấu chấm cho số thập phân.
6. Trả về DUY NHẤT một mảng JSON, không markdown, không giải thích ngoài JSON:
[{"finding_id": "F01", "observation": "...", "hypothesis": "...", "recommendation": "...", "citations": ["KB-03"]}]"""


def load_ids(*paths: Path) -> dict[str, str]:
    """{ID: đoạn văn} cho mọi đoạn đánh dấu [KB-xx]/[SOP-xx] ở đầu đoạn."""
    ids = {}
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for block in re.split(r"\n\s*\n", text):
            m = CITE_RE.search(block)
            if m and block.strip().startswith(f"[{m.group(1)}]"):
                ids[m.group(1)] = block.strip()
    return ids


def numbers_in(obj) -> set[float]:
    """Mọi số trong 1 object JSON (kể cả trong chuỗi), làm tròn 1 chữ số."""
    s = json.dumps(obj, ensure_ascii=False)
    s = CITE_TOKEN_RE.sub(" ", s)
    return {round(float(x), 1) for x in NUM_RE.findall(s)}


def validate(exp: dict, finding: dict | None, kb_ids: set[str]) -> dict:
    """Trả {status, bad_citations, unsupported_numbers}."""
    if finding is None:
        return {"status": "failed", "reason": "finding_id không tồn tại", "bad_citations": [], "unsupported_numbers": []}
    cites = exp.get("citations") or []
    bad_c = [c for c in cites if c not in kb_ids]
    allowed = numbers_in(finding)
    text = CITE_TOKEN_RE.sub(" ", f"{exp.get('observation', '')} {exp.get('hypothesis', '')}")
    bad_n = sorted({x for x in NUM_RE.findall(text) if round(float(x), 1) not in allowed}, key=float)
    ok = not bad_c and not bad_n and len(cites) > 0
    return {"status": "passed" if ok else "failed", "bad_citations": bad_c, "unsupported_numbers": bad_n,
            **({"reason": "không có citation"} if not cites else {})}


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


EXPLAIN_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "finding_id": {"type": "string"},
            "observation": {"type": "string"},
            "hypothesis": {"type": "string"},
            "recommendation": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["finding_id", "observation", "hypothesis", "recommendation", "citations"],
    },
}


def call_llm(findings_doc: dict, kb_text: str, sop_text: str, model: str) -> tuple[list[dict], dict]:
    """Gọi Gemini API (google-genai), ép đầu ra JSON theo EXPLAIN_SCHEMA."""
    from google import genai
    from google.genai import types
    client = genai.Client()   # đọc GEMINI_API_KEY (hoặc GOOGLE_API_KEY) từ env
    user = (f"<kaizen_kb>\n{kb_text}\n</kaizen_kb>\n\n<sop>\n{sop_text}\n</sop>\n\n"
            f"<findings>\n{json.dumps(findings_doc['findings'], ensure_ascii=False, indent=1)}\n</findings>\n\n"
            "Giải thích TỪNG finding theo đúng định dạng JSON đã quy định.")
    resp = client.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                                           response_json_schema=EXPLAIN_SCHEMA, max_output_tokens=16000))
    text = (resp.text or "").strip()
    if not text:
        fb = getattr(resp, "prompt_feedback", None)
        reason = resp.candidates[0].finish_reason if resp.candidates else None
        raise RuntimeError(f"Gemini không trả nội dung (finish_reason={reason}, prompt_feedback={fb})")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    i, j = text.find("["), text.rfind("]")
    data = json.loads(text[i:j + 1])
    um = resp.usage_metadata
    usage = {"model": getattr(resp, "model_version", None) or model,
             "input_tokens": getattr(um, "prompt_token_count", None),
             "output_tokens": getattr(um, "candidates_token_count", None),
             "finish_reason": str(resp.candidates[0].finish_reason) if resp.candidates else None,
             "response_id": getattr(resp, "response_id", None)}
    return data, usage


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--findings", default="outputs/findings/findings.json")
    ap.add_argument("--kb", default="config/kaizen_kb.md")
    ap.add_argument("--sop", default="config/sop.md")
    ap.add_argument("--out", default="outputs/findings/explained.json")
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="không gọi API, sinh explanation giả")
    a = ap.parse_args()
    out = Path(a.out)
    if not can_write(out, a.force, TAG):
        return
    doc = read_json(Path(a.findings))
    findings = doc["findings"]
    kb_ids = load_ids(Path(a.kb), Path(a.sop))
    by_id = {f["id"]: f for f in findings}

    usage = {}
    if not findings:
        exps, mode = [], "none"
    elif a.dry_run:
        exps, mode = dry_run_explain(findings), "dry-run"
    else:
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            sys.exit(f"[{TAG}] thiếu GEMINI_API_KEY — export GEMINI_API_KEY=... hoặc chạy với --dry-run")
        exps, usage = call_llm(doc, Path(a.kb).read_text(encoding="utf-8"),
                               Path(a.sop).read_text(encoding="utf-8"), a.llm_model)
        mode = "llm"

    seen, items = set(), []
    for e in exps:
        fid = e.get("finding_id")
        seen.add(fid)
        items.append({**e, "validation": validate(e, by_id.get(fid), set(kb_ids))})
    for f in findings:
        if f["id"] not in seen:
            items.append({"finding_id": f["id"], "observation": "", "hypothesis": "", "recommendation": "",
                          "citations": [], "validation": {"status": "failed", "reason": "LLM không trả lời finding này",
                                                          "bad_citations": [], "unsupported_numbers": []}})
    n_pass = sum(1 for x in items if x["validation"]["status"] == "passed")
    res = {"mode": mode, "llm_model": a.llm_model if mode == "llm" else None, "usage": usage,
           "n_findings": len(findings), "n_passed": n_pass,
           "pass_rate": round(n_pass / len(items), 3) if items else None,
           "validator": "citation ∈ KB/SOP; mọi số trong observation+hypothesis ∈ số của finding (làm tròn 0.1)",
           "explanations": items}
    write_json(out, res)
    for x in items:
        v = x["validation"]
        extra = f" bad_cite={v['bad_citations']} bad_num={v['unsupported_numbers']}" if v["status"] != "passed" else ""
        print(f"[{TAG}] {x['finding_id']}: {v['status']}{extra}")
    print(f"[{TAG}] mode={mode}, pass {n_pass}/{len(items)} → {out}")


if __name__ == "__main__":
    main()

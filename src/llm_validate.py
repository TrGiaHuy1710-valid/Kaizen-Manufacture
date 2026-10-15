"""Validator + parser lượng thứ cho explanation của LLM (thuần Python, không gọi mạng — dùng trong 07 và tests).

Validator:
  - Whitelist số của 1 finding = (1) mọi giá trị số trong finding (cả |x|), làm tròn 0 và 1 chữ số;
    (2) mọi số xuất hiện trong TÊN TRƯỜNG và chuỗi của finding (dev_p90_deg → 90, "lệch > 30°" → 30,
    U001_V001 → 1); (3) số phổ quát 0, 1, 2, 3, 100 và số bước SOP 1–12; (4) số trong citation id.
  - Token số trong observation + hypothesis + recommendation (bỏ số dính chữ như P90/U001/F01, bỏ mã citation,
    hiểu dấu phần nghìn 13,715.4) phải ∈ whitelist (so sau khi làm tròn 1 chữ số).
  - Citation chấp nhận KB-13 / KB13 / kb-13 / [KB-13] → chuẩn hoá "KB-13" rồi đối chiếu KB/SOP.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CITE_ANY_RE = re.compile(r"\[?\s*\b(KB|SOP)\s*-?\s*(\d+)\b\s*\]?", re.I)
KB_HEAD_RE = re.compile(r"^\[((?:KB|SOP)-\d+)\]")
# số trong văn bản: không dính chữ/số phía TRƯỚC (loại P90, U001, F01, R7) nhưng được dính đơn vị phía sau
# (8.2s, 65.4%, 30°, 120px vẫn bị kiểm); hỗ trợ phần nghìn "13,715.4"
TEXT_NUM_RE = re.compile(r"(?<![\w.,])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
RAW_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
UNIVERSAL = {0.0, 1.0, 2.0, 3.0, 100.0} | {float(i) for i in range(1, 13)}
FIELDS = ("observation", "hypothesis", "recommendation")


# ---------------- KB / citation ----------------
def load_ids(*paths: Path) -> dict[str, str]:
    """{ID: đoạn văn} cho mọi đoạn bắt đầu bằng [KB-xx]/[SOP-xx]."""
    ids = {}
    for p in paths:
        for block in re.split(r"\n\s*\n", Path(p).read_text(encoding="utf-8")):
            m = KB_HEAD_RE.match(block.strip())
            if m:
                ids[normalize_citation(m.group(1))] = block.strip()
    return ids


def normalize_citation(c) -> str | None:
    """'kb13' / '[KB-13]' / 'KB 13' → 'KB-13'; None nếu không phải mã KB/SOP."""
    m = CITE_ANY_RE.fullmatch(str(c).strip())
    if not m:
        return None
    return f"{m.group(1).upper()}-{int(m.group(2)):02d}"


# ---------------- Số ----------------
def _to_float(tok: str) -> float:
    return float(tok.replace("−", "-").replace(",", ""))


def _walk(obj, nums: list[float], texts: list[str]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            texts.append(str(k))
            _walk(v, nums, texts)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, nums, texts)


def whitelist(finding: dict, citations: list[str] | None = None) -> set[float]:
    """Tập số được phép nhắc tới khi giải thích finding (xem docstring module)."""
    nums, texts = [], []
    _walk(finding, nums, texts)
    for t in texts + list(citations or []):
        nums += [float(x) for x in RAW_NUM_RE.findall(t)]
    allowed = set(UNIVERSAL)
    for v in nums:
        for x in (v, abs(v)):
            allowed.add(round(x, 1))
            allowed.add(float(round(x)))
    return allowed


def number_tokens(text: str) -> list[str]:
    """Token số trong văn bản (đã bỏ mã citation)."""
    text = CITE_ANY_RE.sub(" ", text)
    return TEXT_NUM_RE.findall(text)


def validate(exp: dict, finding: dict | None, kb_ids: set[str]) -> dict:
    """{status: passed|failed, unknown_numbers, bad_citations, citations_normalized, message}."""
    if finding is None:
        return {"status": "failed", "unknown_numbers": [], "bad_citations": [], "citations_normalized": [],
                "message": f"finding_id không tồn tại: {exp.get('finding_id')!r}"}
    raw = exp.get("citations") or []
    if isinstance(raw, str):
        raw = re.split(r"[,;\s]+", raw)
    norm = [normalize_citation(c) for c in raw if str(c).strip()]
    bad_c = [str(c) for c, n in zip([c for c in raw if str(c).strip()], norm) if n is None or n not in kb_ids]
    good = [n for n in norm if n and n in kb_ids]
    allowed = whitelist(finding, good)
    text = " ".join(str(exp.get(k, "")) for k in FIELDS)
    unknown = []
    for tok in number_tokens(text):
        if round(_to_float(tok), 1) not in allowed and tok not in unknown:
            unknown.append(tok)
    parts = []
    if unknown:
        parts.append(f"số lạ: {unknown}")
    if bad_c:
        parts.append(f"citation không tồn tại: {bad_c}")
    if not good:
        parts.append("thiếu citation hợp lệ")
    return {"status": "passed" if not parts else "failed", "unknown_numbers": unknown, "bad_citations": bad_c,
            "citations_normalized": good, "message": "; ".join(parts)}


# ---------------- Parser lượng thứ ----------------
class ParseError(ValueError):
    pass


def _outer_json(text: str) -> str:
    """Đoạn JSON ngoài cùng cân bằng ngoặc, bắt đầu từ '[' hoặc '{' đầu tiên."""
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
    if start < 0:
        raise ParseError("không thấy '[' hoặc '{'")
    stack, in_str, esc = [], False, False
    pairs = {"[": "]", "{": "}"}
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "]}":
            if not stack or ch != stack.pop():
                raise ParseError(f"ngoặc lệch ở vị trí {i}")
            if not stack:
                return text[start:i + 1]
    raise ParseError("JSON không đóng ngoặc")


def parse_explanations(text: str) -> list[dict]:
    """Parse response LLM thành list explanation. Bỏ code fence/lời dẫn, bọc object đơn thành list,
    bỏ key thừa, điền key thiếu (đánh dấu format_warning). Raise ParseError nếu không cứu được."""
    if not text or not text.strip():
        raise ParseError("response rỗng")
    t = re.sub(r"```(?:json|JSON)?", "", text)
    try:
        data = json.loads(_outer_json(t))
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON lỗi: {e}") from e
    if isinstance(data, dict):
        inner = next((v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
        data = inner if inner is not None and "finding_id" not in data else [data]
    if not isinstance(data, list):
        raise ParseError(f"kiểu gốc không hợp lệ: {type(data).__name__}")
    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        item, warn = {}, []
        for k in ("finding_id",) + FIELDS:
            v = it.get(k)
            if v is None:
                warn.append(f"thiếu '{k}'")
                v = ""
            item[k] = str(v)
        c = it.get("citations")
        if c is None:
            warn.append("thiếu 'citations'")
            c = []
        item["citations"] = re.split(r"[,;\s]+", c.strip()) if isinstance(c, str) else [str(x) for x in c]
        extra = sorted(set(it) - set(item))
        if extra:
            warn.append(f"bỏ key thừa {extra}")
        if warn:
            item["format_warning"] = warn
        out.append(item)
    if not out:
        raise ParseError("không có explanation nào dạng object")
    return out

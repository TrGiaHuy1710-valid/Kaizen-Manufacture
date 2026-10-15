"""Parse lượng thứ response LLM — dùng CHUNG cho mọi provider (anthropic / gemini / openai / fake).

Bỏ code fence ```json … ``` và lời dẫn, lấy đoạn JSON ngoài cùng cân bằng ngoặc (bỏ qua ngoặc trong chuỗi).
"""
from __future__ import annotations

import json
import re
from typing import Any

FENCE_RE = re.compile(r"```(?:json|JSON)?")


class ParseError(ValueError):
    """Response không cứu được thành JSON."""


def strip_fences(text: str) -> str:
    """Bỏ mọi ``` / ```json."""
    return FENCE_RE.sub("", text)


def outer_json(text: str) -> str:
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


def parse_json_loose(text: str) -> Any:
    """Response LLM → object JSON (bỏ fence/lời dẫn). Raise ParseError nếu không cứu được."""
    if not text or not text.strip():
        raise ParseError("response rỗng")
    try:
        return json.loads(outer_json(strip_fences(text)))
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON lỗi: {e}") from e

"""Fixture dùng chung: PDF SOP mẫu (sinh tất định bằng PyMuPDF) + provider LLM giả có kịch bản."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(FIXTURES))


@pytest.fixture(scope="session")
def sop_pdf() -> Path:
    """tests/fixtures/sop_sample.pdf — 2 trang, 12 bước, 1 bảng (sinh lại mỗi phiên, bytes tất định)."""
    pytest.importorskip("pymupdf")
    from make_sop_pdf import build_sop_pdf
    return build_sop_pdf(FIXTURES / "sop_sample.pdf")


class ScriptedProvider:
    """LLMProvider giả: trả lần lượt `responses` (str hoặc Exception), ghi lại mọi lời gọi. Không mạng, không key."""

    def __init__(self, responses, name="fake", model="fake-1"):
        self.name, self.model, self.calls = name, model, []
        self._seq = iter(responses)

    def complete(self, system, user, *, json_mode=True, max_tokens=2000, history=None):
        from kaizen.llm.providers import LLMResult
        self.calls.append({"system": system, "user": user, "history": list(history or [])})
        r = next(self._seq)
        if isinstance(r, Exception):
            raise r
        return LLMResult(r, {"provider": self.name})


@pytest.fixture
def scripted():
    """Factory: scripted([...responses]) → ScriptedProvider."""
    return ScriptedProvider

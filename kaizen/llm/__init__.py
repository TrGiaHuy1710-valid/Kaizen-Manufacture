"""LLM: provider + prompt (có version) + validator. LLM CHỈ giải thích — mọi số do code tính, validator chặn số/trích dẫn bịa.

    get_provider(name=None, model=None) -> LLMProvider   # anthropic | gemini | openai | dry-run (env LLM_PROVIDER/LLM_MODEL)
    provider.complete(system, user, json_mode=True, max_tokens=2000) -> LLMResult(.text, .parsed, .usage)
    explain(findings, kb, provider, retriever=None) -> dict (nội dung explained.json)
"""
from kaizen.llm.explain import KnowledgeBase, explain
from kaizen.llm.parsing import ParseError, parse_json_loose
from kaizen.llm.providers import (PROVIDER_NAMES, PROVIDERS, LLMError, LLMProvider, LLMResult, MissingAPIKeyError,
                                  available_providers, get_provider)

__all__ = ["KnowledgeBase", "LLMError", "LLMProvider", "LLMResult", "MissingAPIKeyError", "PROVIDERS",
           "PROVIDER_NAMES", "ParseError", "available_providers", "explain", "get_provider", "parse_json_loose"]

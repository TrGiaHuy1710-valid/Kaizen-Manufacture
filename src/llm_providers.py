"""Một interface gọi LLM cho 07: chat(provider, model, system, messages) -> (text, usage).

messages: [{"role": "user"|"assistant", "content": str}, ...] (hội thoại nhiều lượt để retry có phản hồi).
  anthropic : SDK `anthropic`, env ANTHROPIC_API_KEY, model mặc định claude-sonnet-4-6.
  gemini    : SDK `google-genai` (đã có trong requirements), env GEMINI_API_KEY (hoặc GOOGLE_API_KEY),
              model mặc định env GEMINI_MODEL hoặc gemini-3.8-flash; bật response_mime_type=application/json.
"""
from __future__ import annotations

import os

PROVIDERS = ("anthropic", "gemini", "dry-run")
DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "gemini": "gemini-3.8-flash"}
MAX_TOKENS = 16000


class LLMError(RuntimeError):
    pass


def default_model(provider: str) -> str | None:
    if provider == "gemini":
        return os.environ.get("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
    return DEFAULT_MODELS.get(provider)


def check_credentials(provider: str) -> str | None:
    """Thông báo lỗi nếu thiếu key, None nếu ổn."""
    if provider == "anthropic" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return "thiếu ANTHROPIC_API_KEY"
    if provider == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return "thiếu GEMINI_API_KEY"
    return None


def chat(provider: str, model: str, system: str, messages: list[dict]) -> tuple[str, dict]:
    if provider == "anthropic":
        return _anthropic(model, system, messages)
    if provider == "gemini":
        return _gemini(model, system, messages)
    raise LLMError(f"provider không hỗ trợ gọi API: {provider}")


def _anthropic(model: str, system: str, messages: list[dict]) -> tuple[str, dict]:
    import anthropic
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=system, messages=messages)
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Anthropic: lỗi mạng: {e}") from e
    if resp.stop_reason == "refusal":
        raise LLMError(f"Anthropic từ chối: {getattr(resp, 'stop_details', None)}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = {"provider": "anthropic", "model": resp.model, "input_tokens": resp.usage.input_tokens,
             "output_tokens": resp.usage.output_tokens, "stop_reason": resp.stop_reason,
             "request_id": getattr(resp, "_request_id", None)}
    return text, usage


def _gemini(model: str, system: str, messages: list[dict]) -> tuple[str, dict]:
    from google import genai
    from google.genai import errors, types
    client = genai.Client()   # GEMINI_API_KEY / GOOGLE_API_KEY từ env
    contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                              parts=[types.Part(text=m["content"])]) for m in messages]
    try:
        resp = client.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                               max_output_tokens=MAX_TOKENS))
    except errors.APIError as e:
        raise LLMError(f"Gemini API {getattr(e, 'code', '?')}: {getattr(e, 'message', e)}") from e
    text = resp.text or ""
    um = resp.usage_metadata
    usage = {"provider": "gemini", "model": getattr(resp, "model_version", None) or model,
             "input_tokens": getattr(um, "prompt_token_count", None),
             "output_tokens": getattr(um, "candidates_token_count", None),
             "finish_reason": str(resp.candidates[0].finish_reason) if resp.candidates else None,
             "response_id": getattr(resp, "response_id", None)}
    if not text.strip():
        raise LLMError(f"Gemini không trả nội dung (finish_reason={usage['finish_reason']}, "
                       f"prompt_feedback={getattr(resp, 'prompt_feedback', None)})")
    return text, usage

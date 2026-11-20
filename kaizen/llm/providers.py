"""Provider LLM sau 1 interface: `complete(system, user, *, json_mode, max_tokens, history) -> LLMResult`.

    get_provider(name=None, model=None)
        name : tham số > env LLM_PROVIDER > "gemini"           ("anthropic" | "gemini" | "openai" | "dry-run")
        model: tham số (--llm-model) > env LLM_MODEL > env riêng (ANTHROPIC_MODEL / GEMINI_MODEL / OPENAI_MODEL)
               > mặc định của provider
    Key CHỈ qua env: ANTHROPIC_API_KEY (hoặc ANTHROPIC_AUTH_TOKEN), GEMINI_API_KEY (hoặc GOOGLE_API_KEY),
    OPENAI_API_KEY. Thiếu key → MissingAPIKeyError ngay khi khởi tạo (thông báo nêu tên biến env).

    anthropic : SDK `anthropic` (messages API); không có JSON mode native → dựa vào prompt + parse lượng thứ.
    gemini    : SDK `google-genai`; json_mode → response_mime_type=application/json.
    openai    : SDK `openai` (chat.completions); json_mode → response_format={"type": "json_object"}
                (trả 1 object — parse_explanations tự bóc list bên trong).
    dry-run   : không gọi mạng; trả "[]" (các tác vụ có nhánh dry-run riêng, vd explain/extract_steps).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

from kaizen.llm.parsing import ParseError, parse_json_loose

DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "gemini": "gemini-3.8-flash", "openai": "gpt-4.1-mini"}
MODEL_ENV = {"anthropic": "ANTHROPIC_MODEL", "gemini": "GEMINI_MODEL", "openai": "OPENAI_MODEL"}
KEY_ENV = {"anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"), "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
           "openai": ("OPENAI_API_KEY",)}
MAX_TOKENS = 16000          # explain cần output dài (nhiều finding); mặc định complete() là 2000


class LLMError(RuntimeError):
    """Lỗi gọi API (HTTP, mạng, từ chối, rỗng)."""


class MissingAPIKeyError(LLMError):
    """Thiếu API key trong env — message nêu đúng tên biến cần export."""

    def __init__(self, provider: str):
        self.provider, self.env = provider, KEY_ENV[provider][0]
        super().__init__(f"thiếu {self.env} cho provider {provider} (export {self.env}=... hoặc dùng --provider dry-run)")


@dataclass
class LLMResult:
    """Kết quả 1 lần gọi: text thô, usage (provider, model, token...), parsed = JSON parse lượng thứ (None nếu hỏng)."""
    text: str
    usage: dict = field(default_factory=dict)

    @cached_property
    def parsed(self) -> Any:
        """JSON trong text (bỏ fence/lời dẫn); None nếu không parse được (xem parse_error)."""
        try:
            return parse_json_loose(self.text)
        except ParseError:
            return None

    @property
    def parse_error(self) -> str | None:
        """Lý do không parse được (None nếu parse OK)."""
        try:
            parse_json_loose(self.text)
            return None
        except ParseError as e:
            return str(e)


@runtime_checkable
class LLMProvider(Protocol):
    """Interface chung. history = các lượt trước [{role: user|assistant, content}] (retry có phản hồi)."""
    name: str
    model: str | None

    def complete(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int = 2000,
                 history: list[dict] | None = None) -> LLMResult:
        """1 lượt hoàn thành: system + (history) + user → LLMResult."""


def has_credentials(provider: str) -> bool:
    """True nếu env có key cho provider (dry-run luôn True)."""
    return provider == "dry-run" or any(os.environ.get(k) for k in KEY_ENV.get(provider, ()))


def check_credentials(provider: str) -> str | None:
    """Thông báo lỗi nếu thiếu key, None nếu ổn."""
    return None if has_credentials(provider) else f"thiếu {KEY_ENV[provider][0]}"


def default_model(provider: str) -> str | None:
    """Model mặc định: env LLM_MODEL > env riêng của provider > DEFAULT_MODELS."""
    if provider == "dry-run":
        return None
    return os.environ.get("LLM_MODEL") or os.environ.get(MODEL_ENV.get(provider, ""), "") or DEFAULT_MODELS.get(provider)


def _messages(user: str, history: list[dict] | None) -> list[dict]:
    return [*(history or []), {"role": "user", "content": user}]


class AnthropicProvider:
    """Claude qua SDK `anthropic` (env ANTHROPIC_API_KEY)."""
    name = "anthropic"

    def __init__(self, model: str | None = None):
        if not has_credentials(self.name):
            raise MissingAPIKeyError(self.name)
        import anthropic
        self._sdk = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or default_model(self.name)

    def complete(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int = 2000,
                 history: list[dict] | None = None) -> LLMResult:
        """messages.create; json_mode chỉ dựa vào prompt (parse lượng thứ phía sau)."""
        a = self._sdk
        try:
            resp = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                               messages=_messages(user, history))
        except a.APIStatusError as e:
            raise LLMError(f"Anthropic API {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"Anthropic: lỗi mạng: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError(f"Anthropic từ chối: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = {"provider": "anthropic", "model": resp.model, "input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens, "stop_reason": resp.stop_reason,
                 "request_id": getattr(resp, "_request_id", None)}
        return LLMResult(text, usage)


class GeminiProvider:
    """Gemini qua SDK `google-genai` (env GEMINI_API_KEY / GOOGLE_API_KEY)."""
    name = "gemini"

    def __init__(self, model: str | None = None):
        if not has_credentials(self.name):
            raise MissingAPIKeyError(self.name)
        from google import genai
        self.client = genai.Client()
        self.model = model or default_model(self.name)

    def complete(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int = 2000,
                 history: list[dict] | None = None) -> LLMResult:
        """generate_content; json_mode → response_mime_type=application/json."""
        from google.genai import errors, types
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])]) for m in _messages(user, history)]
        cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                          **({"response_mime_type": "application/json"} if json_mode else {}))
        try:
            resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
        except errors.APIError as e:
            raise LLMError(f"Gemini API {getattr(e, 'code', '?')}: {getattr(e, 'message', e)}") from e
        text = resp.text or ""
        um = resp.usage_metadata
        usage = {"provider": "gemini", "model": getattr(resp, "model_version", None) or self.model,
                 "input_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None),
                 "finish_reason": str(resp.candidates[0].finish_reason) if resp.candidates else None,
                 "response_id": getattr(resp, "response_id", None)}
        if not text.strip():
            raise LLMError(f"Gemini không trả nội dung (finish_reason={usage['finish_reason']}, "
                           f"prompt_feedback={getattr(resp, 'prompt_feedback', None)})")
        return LLMResult(text, usage)


class OpenAIProvider:
    """OpenAI qua SDK `openai` (env OPENAI_API_KEY; OPENAI_BASE_URL nếu dùng endpoint tương thích)."""
    name = "openai"

    def __init__(self, model: str | None = None):
        if not has_credentials(self.name):
            raise MissingAPIKeyError(self.name)
        import openai
        self._sdk = openai
        self.client = openai.OpenAI()
        self.model = model or default_model(self.name)

    def complete(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int = 2000,
                 history: list[dict] | None = None) -> LLMResult:
        """chat.completions.create; json_mode → response_format json_object."""
        o = self._sdk
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            resp = self.client.chat.completions.create(
                model=self.model, max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, *_messages(user, history)], **kw)
        except o.APIStatusError as e:
            raise LLMError(f"OpenAI API {e.status_code}: {e.message}") from e
        except o.APIConnectionError as e:
            raise LLMError(f"OpenAI: lỗi mạng: {e}") from e
        ch = resp.choices[0] if resp.choices else None
        text = (ch.message.content if ch and ch.message else None) or ""
        if ch is not None and getattr(ch.message, "refusal", None):
            raise LLMError(f"OpenAI từ chối: {ch.message.refusal}")
        u = resp.usage
        usage = {"provider": "openai", "model": resp.model, "input_tokens": getattr(u, "prompt_tokens", None),
                 "output_tokens": getattr(u, "completion_tokens", None),
                 "finish_reason": ch.finish_reason if ch else None, "response_id": resp.id}
        if not text.strip():
            raise LLMError(f"OpenAI không trả nội dung (finish_reason={usage['finish_reason']})")
        return LLMResult(text, usage)


class DryRunProvider:
    """Không gọi mạng/không cần key. complete() trả '[]'; tác vụ dùng nhánh dry-run riêng để sinh kết quả giả."""
    name = "dry-run"
    model = None

    def __init__(self, model: str | None = None):
        self.model = None

    def complete(self, system: str, user: str, *, json_mode: bool = True, max_tokens: int = 2000,
                 history: list[dict] | None = None) -> LLMResult:
        """Trả JSON rỗng."""
        return LLMResult("[]", {"provider": "dry-run"})


PROVIDERS: dict[str, type] = {"anthropic": AnthropicProvider, "gemini": GeminiProvider, "openai": OpenAIProvider,
                              "dry-run": DryRunProvider}
PROVIDER_NAMES = tuple(PROVIDERS)


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Khởi tạo provider theo tên/env (xem docstring module). MissingAPIKeyError nếu thiếu key."""
    name = (name or os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    if name == "dry":
        name = "dry-run"
    if name not in PROVIDERS:
        raise ValueError(f"provider không hỗ trợ: {name!r} (có: {', '.join(PROVIDER_NAMES)})")
    return PROVIDERS[name](model or default_model(name))


def available_providers() -> list[str]:
    """Provider có key trong env (luôn gồm dry-run) — dùng cho /health."""
    return [p for p in PROVIDER_NAMES if has_credentials(p)]

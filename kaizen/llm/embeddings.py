"""Embedding qua API provider (tuỳ chọn cho RAG): OpenAI / Gemini. Key chỉ qua env; thiếu key → MissingAPIKeyError.

    KAIZEN_EMBEDDER=openai   model env OPENAI_EMBED_MODEL (mặc định text-embedding-3-small)
    KAIZEN_EMBEDDER=gemini   model env GEMINI_EMBED_MODEL (mặc định gemini-embedding-001)
"""
from __future__ import annotations

import math
import os
import re

from kaizen.llm.providers import LLMError, MissingAPIKeyError, has_credentials

DEFAULT_EMBED = {"openai": "text-embedding-3-small", "gemini": "gemini-embedding-001"}


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class OpenAIEmbedder:
    """client.embeddings.create (env OPENAI_API_KEY)."""

    def __init__(self, model: str | None = None):
        if not has_credentials("openai"):
            raise MissingAPIKeyError("openai")
        import openai
        self._sdk, self.client = openai, openai.OpenAI()
        self.model = model or os.environ.get("OPENAI_EMBED_MODEL") or DEFAULT_EMBED["openai"]
        self.slug = "openai-" + re.sub(r"[^a-z0-9]+", "-", self.model.lower())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vector cho từng đoạn (batch 64)."""
        out = []
        for i in range(0, len(texts), 64):
            try:
                resp = self.client.embeddings.create(model=self.model, input=texts[i:i + 64])
            except self._sdk.OpenAIError as e:
                raise LLMError(f"OpenAI embeddings: {e}") from e
            out += [_norm(d.embedding) for d in resp.data]
        return out

    def embed_query(self, text: str) -> list[float]:
        """Vector câu truy vấn."""
        return self.embed_documents([text])[0]


class GeminiEmbedder:
    """client.models.embed_content (env GEMINI_API_KEY)."""

    def __init__(self, model: str | None = None):
        if not has_credentials("gemini"):
            raise MissingAPIKeyError("gemini")
        from google import genai
        self.client = genai.Client()
        self.model = model or os.environ.get("GEMINI_EMBED_MODEL") or DEFAULT_EMBED["gemini"]
        self.slug = "gemini-" + re.sub(r"[^a-z0-9]+", "-", self.model.lower())

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        from google.genai import errors, types
        out = []
        for i in range(0, len(texts), 64):
            try:
                resp = self.client.models.embed_content(model=self.model, contents=texts[i:i + 64],
                                                        config=types.EmbedContentConfig(task_type=task))
            except errors.APIError as e:
                raise LLMError(f"Gemini embeddings: {e}") from e
            out += [_norm(list(e.values)) for e in resp.embeddings]
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vector cho từng đoạn (RETRIEVAL_DOCUMENT)."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Vector câu truy vấn (RETRIEVAL_QUERY)."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def get_provider_embedder(name: str) -> "OpenAIEmbedder | GeminiEmbedder":
    """'openai' | 'gemini' → embedder (MissingAPIKeyError nếu thiếu key)."""
    return {"openai": OpenAIEmbedder, "gemini": GeminiEmbedder}[name]()

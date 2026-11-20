"""Embedding sau interface `Embedder` — đổi model = đổi 1 tham số / env, không đổi code gọi.

    get_embedder(name=None)   name hoặc env KAIZEN_EMBEDDER:
        "e5" (mặc định)  sentence-transformers intfloat/multilingual-e5-small (local, đa ngôn ngữ, có tiếng Việt;
                         lần đầu tải ~470 MB về cache HuggingFace — đặt HF_HOME để đổi chỗ lưu)
        "st:<model_id>"  model sentence-transformers bất kỳ
        "hash"           băm n-gram ký tự (offline, tất định, không tải gì) — dùng cho test/CI
        "openai" / "gemini"  embedding qua API provider (kaizen.llm.embeddings; cần key env)
Mỗi embedder có `slug` riêng → store dùng collection riêng theo slug (không trộn vector khác không gian).
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
import zlib
from typing import Protocol, runtime_checkable

DEFAULT_ST_MODEL = "intfloat/multilingual-e5-small"


@runtime_checkable
class Embedder(Protocol):
    """texts → vector (đã chuẩn hoá L2). Tài liệu và câu truy vấn có thể được mã hoá khác nhau (vd e5 prefix)."""
    slug: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vector cho từng đoạn tài liệu."""

    def embed_query(self, text: str) -> list[float]:
        """Vector cho câu truy vấn."""


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn")).replace("đ", "d")


class HashEmbedder:
    """Băm từ + n-gram ký tự (có và không dấu) vào `dim` chiều (crc32, tất định giữa các process), chuẩn hoá L2.

    Không hiểu ngữ nghĩa — chỉ trùng từ vựng; đủ cho test/CI và tài liệu cùng thuật ngữ.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.slug = f"hash{dim}"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        low = unicodedata.normalize("NFC", text.lower())
        for variant in (low, _strip_accents(low)):
            words = re.findall(r"\w+", variant)
            feats = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
            for w in words:
                pw = f" {w} "
                feats += [pw[i:i + 3] for i in range(len(pw) - 2)]
            for f in feats:
                h = zlib.crc32(f.encode("utf-8"))
                v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vector cho từng đoạn."""
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Vector cho câu truy vấn."""
        return self._vec(text)


class SentenceTransformerEmbedder:
    """sentence-transformers local (CPU/GPU). Model họ e5 tự thêm tiền tố 'query: ' / 'passage: '."""

    def __init__(self, model: str = DEFAULT_ST_MODEL, device: str | None = None, batch_size: int = 32):
        self.model_name, self.device, self.batch_size = model, device, batch_size
        self.slug = "st-" + re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower()).strip("-")
        self._model = None
        self._e5 = "e5" in model.lower()

    @property
    def model(self):
        """Nạp model lười (lần đầu dùng)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _enc(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vector cho từng đoạn ('passage: ' với e5)."""
        return self._enc([("passage: " + t) if self._e5 else t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Vector câu truy vấn ('query: ' với e5)."""
        return self._enc([("query: " + text) if self._e5 else text])[0]


def get_embedder(name: str | None = None) -> Embedder:
    """Embedder theo tên / env KAIZEN_EMBEDDER (xem docstring module)."""
    name = (name or os.environ.get("KAIZEN_EMBEDDER") or "e5").strip()
    low = name.lower()
    if low in ("e5", "default"):
        return SentenceTransformerEmbedder(DEFAULT_ST_MODEL)
    if low.startswith("st:"):
        return SentenceTransformerEmbedder(name[3:])
    if low.startswith("hash"):
        return HashEmbedder(int(low[4:]) if low[4:].isdigit() else 512)
    if low in ("openai", "gemini"):
        from kaizen.llm.embeddings import get_provider_embedder
        return get_provider_embedder(low)
    raise ValueError(f"embedder không hỗ trợ: {name!r} (e5 | st:<model> | hash | openai | gemini)")

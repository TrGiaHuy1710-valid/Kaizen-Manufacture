"""Shim tương thích ngược: module thật ở kaizen/llm/providers.py (`import llm_providers` ≡ `kaizen.llm.providers`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.llm import providers as _real  # noqa: E402

sys.modules[__name__] = _real

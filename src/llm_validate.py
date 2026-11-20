"""Shim tương thích ngược: module thật ở kaizen/llm/validator.py (`import llm_validate` ≡ `kaizen.llm.validator`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.llm import validator as _real  # noqa: E402

sys.modules[__name__] = _real

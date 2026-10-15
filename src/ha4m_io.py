"""Shim tương thích ngược: module thật ở kaizen/skeleton/ha4m_io.py (`import ha4m_io` ≡ `kaizen.skeleton.ha4m_io`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.skeleton import ha4m_io as _real  # noqa: E402

sys.modules[__name__] = _real

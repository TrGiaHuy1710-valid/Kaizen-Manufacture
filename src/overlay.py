"""Shim tương thích ngược: module thật ở kaizen/skeleton/overlay.py (`import overlay` ≡ `kaizen.skeleton.overlay`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.skeleton import overlay as _real  # noqa: E402

sys.modules[__name__] = _real

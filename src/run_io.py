"""Shim tương thích ngược: module thật ở kaizen/core/run_io.py (`import run_io` ≡ `kaizen.core.run_io`).

CLI giữ nguyên: python src/run_io.py new|latest|path ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.core import run_io as _real  # noqa: E402

if __name__ == "__main__":
    _real.main()
else:
    sys.modules[__name__] = _real

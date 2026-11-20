"""Shim tương thích ngược cho các script/UI cũ: tiện ích nay nằm trong kaizen.core (config, io, cli)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kaizen.core.cli import base_parser, setup, stage_dir, stage_file  # noqa: E402,F401
from kaizen.core.config import DEFAULT_CONFIG, OUT, load_config  # noqa: E402,F401
from kaizen.core.io import (_json_default, _split, can_write, parse_stem, r, read_json, select_stems,  # noqa: E402,F401
                            stem, stems_from_dir, write_json)
from kaizen.skeleton import ha4m_io  # noqa: E402,F401

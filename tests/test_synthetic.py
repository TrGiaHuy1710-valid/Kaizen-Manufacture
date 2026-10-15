"""Chạy tests/synthetic.py (pipeline 00→08 trên dữ liệu tổng hợp) trong pytest."""
import importlib.util
from pathlib import Path


def test_synthetic_pipeline():
    p = Path(__file__).with_name("synthetic.py")
    spec = importlib.util.spec_from_file_location("synthetic_pipeline", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0

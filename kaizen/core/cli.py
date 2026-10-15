"""Args chung cho các script stage (src/0X_*.py): --run/--data-root/--config/..., resolve run, đường dẫn stage."""
from __future__ import annotations

import argparse
from pathlib import Path

from kaizen.core import run_io
from kaizen.core.config import DEFAULT_CONFIG, DEFAULT_DATA_ROOT


def base_parser(desc: str) -> argparse.ArgumentParser:
    """Args chung: --run, --data-root, --config, --subjects, --trials, --force."""
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="run_id hoặc 'latest' (mặc định: outputs/runs/LATEST; chưa có run → đường dẫn phẳng cũ)")
    ap.add_argument("--data-root", default=None,
                    help=f"thư mục dataset, nhiều thư mục cách nhau dấu phẩy (mặc định: data_root trong manifest "
                         f"của run, rồi env HA4M_ROOT, rồi {DEFAULT_DATA_ROOT})")
    ap.add_argument("--config", default=None, help="mặc định: station.yaml chụp trong run, không có thì " + DEFAULT_CONFIG)
    ap.add_argument("--subjects", default="", help="vd U001,U002 (rỗng = tất cả)")
    ap.add_argument("--trials", default="", help="vd V001,V002 (rỗng = tất cả)")
    ap.add_argument("--force", action="store_true", help="ghi đè output đã có")
    return ap


def setup(a: argparse.Namespace, tag: str, need_run: bool = True) -> run_io.RunContext | None:
    """Resolve run (--run/LATEST/legacy) và điền a.data_root, a.config mặc định. Trả RunContext | None."""
    ctx = run_io.resolve(a.run, tag) if need_run else None
    man = ctx.manifest if ctx else {}
    if not a.data_root:
        a.data_root = man.get("data_root") or DEFAULT_DATA_ROOT
    if not a.config:
        a.config = str(ctx.station_path) if ctx and ctx.station_path.exists() else DEFAULT_CONFIG
    return ctx


def stage_dir(ctx: run_io.RunContext | None, override: str | None, stage: str, legacy: str) -> Path:
    """Đường dẫn stage: --xxx-dir tường minh > thư mục trong run > đường dẫn phẳng cũ."""
    if override:
        return Path(override)
    if ctx is not None:
        return ctx.dir(stage)
    return Path(legacy)


def stage_file(ctx: run_io.RunContext | None, override: str | None, stage: str, filename: str, legacy: str) -> Path:
    """Như stage_dir nhưng cho 1 file trong stage."""
    if override:
        return Path(override)
    if ctx is not None:
        return ctx.path(stage, filename)
    return Path(legacy)

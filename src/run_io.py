"""Quản lý output theo run: outputs/runs/<run_id>/{features,events,steps,metrics,findings,explained,report}/ + manifest.json.

run_id = <YYYYmmdd-HHMMSS>_<pred|oracle>_<posemodel>_<anthropic|gemini|dry>[_<tag>]  (trùng → thêm -2, -3...)
outputs/runs/LATEST = file text chứa run_id gần nhất (không dùng symlink để chạy được cả Windows).
Keypoints (.npz) KHÔNG nằm trong run — vẫn ở data/keypoints/ để tái dùng giữa các run.

CLI (dùng trong run_all.sh):
    python src/run_io.py new --source pred --pose mediapipe --llm gemini [--llm-model ...] [--tag demo] \
        [--data-root ...] [--subjects ...] [--trials ...] [--config config/station.yaml] [--cmd "<args run_all>"]
        → in run_id ra stdout
    python src/run_io.py latest        → in run_id gần nhất
    python src/run_io.py path <run|latest> [stage]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

RUNS = Path("outputs/runs")
LATEST = RUNS / "LATEST"
STAGES = ("features", "events", "steps", "metrics", "findings", "explained", "report")
LLM_TAG = {"anthropic": "anthropic", "gemini": "gemini", "dry-run": "dry", "dry": "dry"}


@dataclass
class RunContext:
    run_id: str
    root: Path

    def dir(self, stage: str) -> Path:
        """Thư mục của 1 stage trong run (tạo nếu chưa có)."""
        p = self.root / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    def path(self, stage: str, filename: str) -> Path:
        return self.dir(stage) / filename

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @property
    def station_path(self) -> Path:
        return self.root / "station.yaml"

    def update_manifest(self, **kw) -> None:
        """Gộp kw vào manifest (dict con được merge 1 cấp)."""
        m = self.manifest
        for k, v in kw.items():
            if isinstance(v, dict) and isinstance(m.get(k), dict):
                m[k].update(v)
            else:
                m[k] = v
        self.manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def record_stage(self, stage: str, **info) -> None:
        """Ghi lại 1 lần chạy stage (argv, thời điểm, thông tin thêm) vào manifest['stages']."""
        self.update_manifest(stages={stage: {"at": now_iso(), "argv": sys.argv, **info}})

    def set_trials(self, **per_trial: dict) -> None:
        """manifest['trials'][stem] cập nhật thêm khoá."""
        m = self.manifest
        tr = m.get("trials", {})
        for st, info in per_trial.items():
            tr.setdefault(st, {}).update(info)
        self.update_manifest(trials=tr)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def git_hash() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in s).strip("-")


def new_run(cfg_path: str | Path, *, source: str, pose_model: str, llm: str, llm_model: str | None = None,
            tag: str = "", data_root: str | None = None, subjects: str = "", trials: str = "",
            cmd: list[str] | None = None) -> RunContext:
    """Tạo outputs/runs/<run_id>/ + manifest.json + bản sao station.yaml, cập nhật LATEST."""
    base = f"{dt.datetime.now():%Y%m%d-%H%M%S}_{source}_{_safe(pose_model)}_{LLM_TAG.get(llm, _safe(llm))}"
    if tag:
        base += f"_{_safe(tag)}"
    RUNS.mkdir(parents=True, exist_ok=True)
    run_id, k = base, 2
    while (RUNS / run_id).exists():
        run_id, k = f"{base}-{k}", k + 1
    ctx = RunContext(run_id, RUNS / run_id)
    ctx.root.mkdir(parents=True)
    for s in STAGES:
        ctx.dir(s)
    cfg_path = Path(cfg_path)
    station = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    if cfg_path.exists():
        shutil.copy(cfg_path, ctx.station_path)
    ctx.update_manifest(
        run_id=run_id, created_at=now_iso(), git_hash=git_hash(), cwd=str(Path.cwd()),
        python=platform.python_version(), cmd=cmd or sys.argv, source=source, pose_model=pose_model,
        llm={"provider": llm, "model": llm_model}, tag=tag, data_root=data_root,
        selection={"subjects": subjects or "all", "trials": trials or "all"},
        config={"path": str(cfg_path), "snapshot": "station.yaml", "station": station},
        trials={}, stages={})
    LATEST.write_text(run_id + "\n", encoding="utf-8")
    return ctx


def latest_id() -> str | None:
    try:
        rid = LATEST.read_text(encoding="utf-8").strip()
        return rid if rid and (RUNS / rid).is_dir() else None
    except FileNotFoundError:
        return None


def open_run(run: str) -> RunContext:
    rid = latest_id() if run == "latest" else run
    if not rid or not (RUNS / rid).is_dir():
        raise FileNotFoundError(f"không thấy run '{run}' trong {RUNS}")
    return RunContext(rid, RUNS / rid)


def resolve(run: str | None, tag: str) -> RunContext | None:
    """--run <id|latest>. Không truyền: dùng LATEST nếu có, nếu chưa có run nào → None (đường dẫn phẳng cũ, cảnh báo)."""
    if run:
        try:
            ctx = open_run(run)
        except FileNotFoundError as e:
            print(f"[{tag}] {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[{tag}] run: {ctx.run_id}")
        return ctx
    rid = latest_id()
    if rid:
        print(f"[{tag}] run: {rid} (latest)")
        return RunContext(rid, RUNS / rid)
    print(f"[{tag}] cảnh báo: chưa có outputs/runs/ — dùng đường dẫn phẳng cũ outputs/<stage>/ "
          "(tạo run bằng run_all.sh hoặc `python src/run_io.py new ...`)", file=sys.stderr)
    return None


def list_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS.glob("*/")) if RUNS.exists() else []:
        m = RunContext(d.name, d).manifest
        if m:
            out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    n = sub.add_parser("new")
    n.add_argument("--source", choices=["pred", "oracle"], required=True)
    n.add_argument("--pose", default="mediapipe")
    n.add_argument("--llm", default="dry-run")
    n.add_argument("--llm-model", default=None)
    n.add_argument("--tag", default="")
    n.add_argument("--data-root", default=None)
    n.add_argument("--subjects", default="")
    n.add_argument("--trials", default="")
    n.add_argument("--config", default="config/station.yaml")
    n.add_argument("--cmd", default="", help="dòng lệnh gốc (vd args của run_all.sh) để tái lập")
    sub.add_parser("latest")
    p = sub.add_parser("path")
    p.add_argument("run")
    p.add_argument("stage", nargs="?", default="")
    a = ap.parse_args()
    if a.cmd_name == "new":
        ctx = new_run(a.config, source=a.source, pose_model=a.pose, llm=a.llm, llm_model=a.llm_model or None,
                      tag=a.tag, data_root=a.data_root, subjects=a.subjects, trials=a.trials,
                      cmd=a.cmd.split() if a.cmd else sys.argv)
        print(ctx.run_id)
    elif a.cmd_name == "latest":
        rid = latest_id()
        if not rid:
            sys.exit(1)
        print(rid)
    else:
        ctx = open_run(a.run)
        print(ctx.dir(a.stage) if a.stage else ctx.root)


if __name__ == "__main__":
    main()

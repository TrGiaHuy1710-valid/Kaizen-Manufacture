"""Chạy full luồng cho 1 dự án video, 6 bước có tiến độ (status.json của project):

    1 Keypoints → 2 Features/Events → 3 Steps → 4 Metrics/Findings → 5 Giải thích (RAG+LLM) → 6 Render video overlay

Mỗi bước gọi đúng stage thật của pipeline (src/01…08 — cùng code, cùng schema artifact như run_all.sh), với
KAIZEN_RUNS_DIR = <project>/runs nên run nằm trong dự án. Stage chạy trong tiến trình con: lỗi/thiếu RAM ở pose
không kéo sập UI, và UI Streamlit rerun không ngắt job. Lỗi ở bước nào dừng ở đó: status = error + log.
Provider LLM: tham số > env LLM_PROVIDER (đọc cả .env) > gemini; thiếu key → dry-run + ghi chú (không chạy nửa chừng).

    python -m kaizen.core.pipeline run <project_id> [--provider gemini] [--only render] [--force-keypoints]
    start_job(project_id)  # UI: chạy nền, giữ khoá toàn app (projects.current_job)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from kaizen.core import projects, run_io
from kaizen.core.config import DEFAULT_CONFIG, load_config

REPO = Path(__file__).resolve().parents[2]
STEPS = [("keypoints", "Keypoints"), ("features", "Features/Events"), ("steps", "Steps"),
         ("metrics", "Metrics/Findings"), ("explain", "Giải thích (RAG+LLM)"), ("render", "Render video overlay")]
CALLS: list[str] = []            # project_id mỗi lần run_project/start_job được gọi (spy cho test demo mode)


class PipelineError(RuntimeError):
    """Stage lỗi: step = key bước, log_tail = vài dòng log cuối."""

    def __init__(self, step: str, msg: str, log_tail: str = ""):
        super().__init__(msg)
        self.step, self.log_tail = step, log_tail


def load_env_file(path: Path = REPO / ".env") -> dict[str, str]:
    """KEY=VALUE trong .env (bỏ comment/export/nháy) — không ghi đè biến đã có trong môi trường."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.removeprefix("export ").split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            out[k] = v
    return out


def resolve_provider(requested: str | None, env: dict[str, str]) -> tuple[str, str]:
    """(provider dùng thật, ghi chú). Thiếu key → dry-run, ghi chú nêu tên biến env."""
    from kaizen.llm.providers import KEY_ENV, PROVIDER_NAMES
    name = (requested or env.get("LLM_PROVIDER") or "gemini").strip().lower()
    if name not in PROVIDER_NAMES:
        return "dry-run", f"provider {name!r} không hỗ trợ → dry-run"
    if name == "dry-run" or any(env.get(k) for k in KEY_ENV[name]):
        return name, ""
    return "dry-run", f"thiếu {KEY_ENV[name][0]} cho {name} → giải thích dry-run (văn bản giả, ghi rõ trong report)"


class _Progress:
    """Ghi tiến độ 6 bước vào status.json + gọi callback."""

    def __init__(self, pid: str, keys: list[str], on_step: Callable[[dict], None] | None):
        self.pid, self.on_step = pid, on_step
        self.steps = [{"key": k, "label": lab, "state": "pending" if k in keys else "skipped", "detail": ""}
                      for k, lab in STEPS]

    def set(self, key: str, state: str, detail: str = "", **kw) -> None:
        for s in self.steps:
            if s["key"] == key:
                s["state"] = state
                if detail:
                    s["detail"] = detail
        st = projects.write_status(self.pid, steps=self.steps, current=key, **kw)
        if self.on_step:
            self.on_step(st)


def _sh(args: list[str], env: dict, logf: Path, step: str) -> None:
    with open(logf, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(str(a) for a in args)}\n")
        fh.flush()
        r = subprocess.run([str(a) for a in args], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = "\n".join(logf.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise PipelineError(step, f"{Path(str(args[1])).name if len(args) > 1 else args[0]} thoát mã {r.returncode}", tail)


def run_project(pid: str, *, provider: str | None = None, only: str | None = None, force_keypoints: bool = False,
                log: Callable[[str], None] = print, on_step: Callable[[dict], None] | None = None) -> str:
    """Chạy luồng cho project (đồng bộ). only="render" → chỉ render lại video từ LATEST run. Trả run_id."""
    CALLS.append(pid)
    info = projects.load_project(pid)
    pdir = projects.project_dir(pid).resolve()
    (pdir / "logs").mkdir(parents=True, exist_ok=True)
    logf = pdir / "logs" / f"{time.strftime('%Y%m%d-%H%M%S')}.log"
    file_env = load_env_file()
    env = {**os.environ, **file_env, "KAIZEN_RUNS_DIR": str(pdir / "runs"), "PYTHONUNBUFFERED": "1"}
    keys = ["render"] if only == "render" else [k for k, _ in STEPS]
    prog = _Progress(pid, keys, on_step)
    projects.write_status(pid, state="running", pid=os.getpid(), started_at=projects._now(), message="",
                          log=str(logf), notes=[], run_id=projects.latest_run(pid) if only else None, finished_at=None)
    py, S = sys.executable, (lambda n: REPO / "src" / n)  # noqa: E731
    st, kind = info["stem"], info.get("kind")
    is_ha4m = kind in ("ha4m", "adopted") and info.get("source_uri", "").startswith("ha4m://")
    labels = bool(info.get("labels_available"))
    root = str(info.get("data_root") or pdir)
    sel = ["--subjects", st.split("_")[0], "--trials", st.split("_")[1]] if is_ha4m else []
    dr = ["--data-root", root]
    station = info.get("station") or DEFAULT_CONFIG
    kp_dir = Path(env.get("KAIZEN_KP_DIR") or REPO / "data" / "keypoints")   # .npz trial HA4M dùng chung giữa các run
    notes: list[str] = []
    cur = keys[0]
    try:
        if only == "render":
            rid = projects.latest_run(pid)
            if not rid:
                raise PipelineError("render", "dự án chưa có run nào để render")
        else:
            # 1 — Keypoints
            cur = "keypoints"
            prog.set(cur, "running", projects.POSE_MODEL_LABEL)
            if is_ha4m:
                npz = kp_dir / f"{st}_{projects.POSE_MODEL}.npz"
                if npz.exists() and not force_keypoints:
                    prog.set(cur, "running", f"{projects.POSE_MODEL_LABEL} — tái dùng {npz.name} (đã trích trước)")
                else:
                    _sh([py, S("01_extract.py"), *dr, *sel, "--model", projects.POSE_MODEL, "--out", kp_dir,
                         "--force"], env, logf, cur)
            else:
                kp = pdir / "keypoints" / f"{st}_{projects.POSE_MODEL}.npz"
                if not kp.exists() or force_keypoints:
                    mf = int(float(info["max_seconds"]) * float(info.get("fps") or 30)) if info.get("max_seconds") else 0
                    _sh([py, "-m", "kaizen.skeleton", "extract", info["source_uri"], "--model", projects.POSE_MODEL,
                         "--out", pdir / "keypoints", "--name", st, "--force",
                         *(["--max-frames", str(mf)] if mf else [])], env, logf, cur)
            prog.set(cur, "done")
            # 2 — Features/Events (run mới trong <project>/runs)
            cur = "features"
            prog.set(cur, "running")
            prov, pnote = resolve_provider(provider, env)
            if pnote:
                notes.append(pnote)
            ctx = run_io.new_run(station, source="oracle" if labels else "pred", pose_model=projects.POSE_MODEL,
                                 llm=prov, tag="ui", data_root=root, subjects=sel[1] if sel else "",
                                 trials=sel[3] if sel else "", cmd=["ui", "run_project", pid], root=pdir / "runs")
            rid = ctx.run_id
            projects.write_status(pid, run_id=rid, notes=notes)
            R = ["--run", rid, *dr, "--force"]
            kp_args = ["--kp-dir", kp_dir] if is_ha4m else ["--kp-dir", pdir / "keypoints", "--any-root"]
            _sh([py, S("02_features.py"), *R, *sel, "--model", projects.POSE_MODEL, *kp_args], env, logf, cur)
            _sh([py, S("03_events.py"), *R, *sel], env, logf, cur)
            prog.set(cur, "done")
            # 3 — Steps
            cur = "steps"
            prog.set(cur, "running", "nhãn chuẩn (oracle) + dự đoán (pred)" if labels else "dự đoán (pred) — không có nhãn")
            if labels and is_ha4m:
                _sh([py, S("04_steps.py"), *R, *sel, "--oracle"], env, logf, cur)
            _sh([py, S("04_steps.py"), *R, *sel], env, logf, cur)
            prog.set(cur, "done")
            # 4 — Metrics/Findings
            cur = "metrics"
            prog.set(cur, "running")
            _sh([py, S("05_metrics.py"), *R, *sel, "--source", "oracle" if labels else "pred"], env, logf, cur)
            _sh([py, S("06_findings.py"), "--run", rid, "--force"], env, logf, cur)
            prog.set(cur, "done")
            # 5 — Giải thích
            cur = "explain"
            prog.set(cur, "running", f"provider {prov}" + (f" — {pnote}" if pnote else ""))
            _sh([py, S("07_explain.py"), "--run", rid, "--force", "--provider", prov], env, logf, cur)
            prog.set(cur, "done")
        # 6 — Render video overlay (+ report.md)
        cur = "render"
        prog.set(cur, "running", "nhãn chuẩn (oracle) nếu có, không thì dự đoán")
        from kaizen.skeleton.render import render_overlay_video
        run_root = pdir / "runs" / rid
        cfg = load_config(run_root / "station.yaml" if (run_root / "station.yaml").exists() else station)
        with open(logf, "a", encoding="utf-8") as fh:
            meta = render_overlay_video(run_root, st, cfg, info["source_uri"], pdir / "media" / "demo.mp4",
                                        prefer="oracle", max_seconds=info.get("max_seconds"),
                                        log=lambda m: (fh.write(m + "\n"), fh.flush()),
                                        progress=lambda f: prog.set(cur, "running", f"{f * 100:.0f}%"))
        prog.set(cur, "running", f"nguồn nhãn: {meta['source_label']} — đang ghi report.md")
        _sh([py, S("08_report.py"), "--run", rid, *dr, *sel, "--force"], env, logf, cur)
        prog.set(cur, "done", f"nguồn nhãn: {meta['source_label']}")
        projects.write_status(pid, state="done", message="xong", finished_at=projects._now(), run_id=rid)
        log(f"[pipeline] {pid}: xong run {rid}")
        return rid
    except Exception as e:
        tail = getattr(e, "log_tail", "")
        prog.set(cur, "error", str(e))
        projects.write_status(pid, state="error", message=f"lỗi ở bước {dict(STEPS)[cur]}: {e}", log_tail=tail,
                              finished_at=projects._now())
        log(f"[pipeline] {pid}: lỗi ở bước {cur}: {e}")
        raise


def start_job(pid: str, *, provider: str | None = None, only: str | None = None) -> int:
    """UI: chạy nền `python -m kaizen.core.pipeline run <pid>`; JobBusy nếu project khác đang chạy. Trả pid tiến trình."""
    CALLS.append(pid)
    j = projects.current_job()
    if j:
        raise projects.JobBusy(f"đang chạy dự án {j['project_id']} — chờ xong rồi chạy tiếp")
    projects.load_project(pid)
    pdir = projects.project_dir(pid)
    (pdir / "logs").mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-m", "kaizen.core.pipeline", "run", pid]
    if provider:
        args += ["--provider", provider]
    if only:
        args += ["--only", only]
    env = {**os.environ, "KAIZEN_PROJECTS_DIR": str(projects.projects_root().resolve())}
    out = open(pdir / "logs" / "job.out", "a", encoding="utf-8")  # noqa: SIM115 — tiến trình con giữ file
    p = subprocess.Popen(args, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    projects.acquire_job(pid, p.pid)
    projects.write_status(pid, state="running", pid=p.pid, started_at=projects._now(), message="đang khởi động",
                          steps=[{"key": k, "label": lab, "state": "pending", "detail": ""} for k, lab in STEPS])
    return p.pid


def main() -> None:
    """CLI: run <project_id>."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("project_id")
    r.add_argument("--provider", default=None)
    r.add_argument("--only", choices=["render"], default=None)
    r.add_argument("--force-keypoints", action="store_true")
    a = ap.parse_args()
    me = os.getpid()
    try:
        projects.acquire_job(a.project_id, me)
    except projects.JobBusy as e:
        print(f"[pipeline] {e}", file=sys.stderr)
        sys.exit(2)
    try:
        run_project(a.project_id, provider=a.provider, only=a.only, force_keypoints=a.force_keypoints)
    except Exception:
        sys.exit(1)
    finally:
        projects.release_job(me)


if __name__ == "__main__":
    main()

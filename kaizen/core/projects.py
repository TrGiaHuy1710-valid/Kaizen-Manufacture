"""Dự án video: 1 video = 1 project, lưu trọn trong outputs/projects/<project_id>/ (env KAIZEN_PROJECTS_DIR).

    outputs/projects/U001_V001/
    ├── project.yaml        # id, name, stem, source_uri, kind, station, labels_available, created_at, max_seconds
    ├── status.json         # trạng thái job gần nhất: running | done | error + tiến độ 6 bước
    ├── runs/<run_id>/      # cấu trúc run như outputs/runs (run_io với KAIZEN_RUNS_DIR) — schema artifact không đổi
    ├── runs/LATEST
    ├── keypoints/          # .npz của video upload (trial HA4M dùng chung data/keypoints/)
    ├── logs/               # log từng lần chạy
    └── media/              # thumb.jpg, demo.mp4 (+ demo.json: nguồn nhãn của bản render), source.mp4 (upload)

API: create_project(source_uri, name) / create_project_from_upload(bytes, filename) / list_projects() /
project_status(id) (empty | running | done | error) / latest_run(id) / benchmark(id) / demo_ready(id).
Khoá job toàn app: outputs/projects/.job.json (1 pipeline chạy tại một thời điểm, mọi phiên UI thấy chung).
CLI:
    python -m kaizen.core.projects list
    python -m kaizen.core.projects adopt <run_id> [--runs-dir outputs/runs]     # bọc run cũ thành project
    python -m kaizen.core.projects prepare-demo --trials U001_V001,U001_V002    # kiểm + chạy bù, in READY/MISSING
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Callable

import yaml

from kaizen.core.config import DEFAULT_CONFIG, DEFAULT_DATA_ROOT

STATES = ("empty", "running", "done", "error")
STATE_VI = {"empty": "Chưa chạy", "running": "Đang chạy", "done": "Xong", "error": "Lỗi"}
BENCH_NOTE = "Baseline rule-based, chưa huấn luyện model — sai số phản ánh giới hạn hiện tại."
MODEL_LINE = "Model vision: base (pretrained, chưa train) · roadmap: fine-tune trên dữ liệu tích lũy"
POSE_MODEL = "mediapipe"
POSE_MODEL_LABEL = "MediaPipe Pose + Hands (pretrained, chưa fine-tune)"


class ProjectError(RuntimeError):
    """Lỗi thao tác project (không tồn tại, nguồn hỏng...)."""


class JobBusy(ProjectError):
    """Đang có pipeline khác chạy (khoá toàn app)."""


# ---------------------------------------------------------------- đường dẫn
def projects_root() -> Path:
    """env KAIZEN_PROJECTS_DIR hoặc outputs/projects."""
    return Path(os.environ.get("KAIZEN_PROJECTS_DIR") or "outputs/projects")


def project_dir(pid: str) -> Path:
    """outputs/projects/<pid>/."""
    return projects_root() / pid


def runs_root(pid: str) -> Path:
    """<project>/runs (truyền vào KAIZEN_RUNS_DIR khi chạy stage)."""
    return project_dir(pid) / "runs"


def media_path(pid: str, name: str) -> Path:
    """<project>/media/<name>."""
    return project_dir(pid) / "media" / name


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------- tạo / đọc project
def slug(s: str, n: int = 40) -> str:
    """'Video tự quay #1.mp4' → 'Video-tu-quay-1' (chỉ chữ/số/-; không '_' để stem tách subject_trial đúng)."""
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n] or "video"


def _unique(pid: str) -> str:
    out, k = pid, 2
    while project_dir(out).exists():
        out, k = f"{pid}-{k}", k + 1
    return out


def load_project(pid: str) -> dict:
    """Nội dung project.yaml (ProjectError nếu không có)."""
    p = project_dir(pid) / "project.yaml"
    if not p.exists():
        raise ProjectError(f"không có dự án {pid!r} trong {projects_root()}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def save_project(info: dict) -> None:
    """Ghi project.yaml."""
    d = project_dir(info["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "project.yaml").write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _thumb(pid: str, uri: str) -> None:
    import cv2
    from kaizen.skeleton.sources import open_source
    try:
        for _, img in open_source(uri).frames(max_frames=1):
            h, w = img.shape[:2]
            img = cv2.resize(img, (320, int(320 * h / w)), interpolation=cv2.INTER_AREA)
            p = media_path(pid, "thumb.jpg")
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            break
    except Exception:                       # thumbnail chỉ để hiển thị — lỗi đọc ảnh không chặn tạo project
        pass


def create_project(source_uri: str, name: str | None = None, *, station: str = DEFAULT_CONFIG,
                   max_seconds: float | None = None) -> dict:
    """Tạo project từ URI nguồn (ha4m://… | video://… | frames://…?fps=). Trả project.yaml (dict)."""
    from kaizen.skeleton.sources import open_source, parse_uri
    try:
        src = open_source(source_uri)
        meta = src.meta()
    except Exception as e:
        raise ProjectError(f"không mở được nguồn {source_uri}: {e}") from e
    kind = meta.get("kind") or parse_uri(source_uri)[0]
    if kind == "ha4m":
        stem = meta["stem"]
        pid = _unique(stem)
    else:
        base = slug(name or meta.get("stem") or "video")
        pid = _unique(f"VID_{base}")
        stem = pid
    info = {"id": pid, "name": name or (stem if kind == "ha4m" else base), "stem": stem, "kind": kind,
            "source_uri": source_uri, "station": station, "labels_available": bool(meta.get("labels_available")),
            "data_root": meta.get("data_root") if kind == "ha4m" else None,
            "fps": float(src.fps()), "n_frames": meta.get("n_frames"), "width": meta.get("width"),
            "height": meta.get("height"), "max_seconds": max_seconds, "created_at": _now()}
    save_project(info)
    for sub in ("runs", "media", "logs"):
        (project_dir(pid) / sub).mkdir(parents=True, exist_ok=True)
    _thumb(pid, source_uri)
    return info


def create_project_from_upload(data: bytes, filename: str, name: str | None = None,
                               max_seconds: float | None = None, station: str = DEFAULT_CONFIG) -> dict:
    """Lưu file upload (.mp4/.mov/...) vào <project>/media/source<ext> rồi tạo project video://."""
    ext = Path(filename).suffix.lower() or ".mp4"
    tmp = projects_root() / ".uploads" / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{slug(Path(filename).stem)}{ext}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    info = create_project(f"video://{tmp.resolve()}", name or Path(filename).stem, station=station,
                          max_seconds=max_seconds)
    dst = media_path(info["id"], f"source{ext}")
    shutil.move(str(tmp), dst)
    info["source_uri"] = f"video://{dst.resolve()}"
    save_project(info)
    return info


def latest_run(pid: str) -> str | None:
    """run_id mới nhất của project (<project>/runs/LATEST) hoặc None."""
    try:
        rid = (runs_root(pid) / "LATEST").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return rid if rid and (runs_root(pid) / rid).is_dir() else None


def run_path(pid: str, run_id: str | None = None) -> Path | None:
    """Thư mục run (mặc định LATEST) của project."""
    rid = run_id or latest_run(pid)
    return runs_root(pid) / rid if rid else None


def read_status(pid: str) -> dict:
    """status.json ({} nếu chưa chạy lần nào)."""
    return _read_json(project_dir(pid) / "status.json") or {}


def write_status(project_id: str, /, **kw) -> dict:
    """Gộp kw vào status.json (+ updated_at). kw có thể chứa pid = id tiến trình đang chạy."""
    s = read_status(project_id)
    s.update(kw)
    s["updated_at"] = _now()
    _write_json(project_dir(project_id) / "status.json", s)
    return s


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:    # con của tiến trình này đã thoát → dọn zombie
            return False
    except (ChildProcessError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_complete(pid: str, run_id: str | None = None) -> bool:
    """Run có đủ metrics.json + findings.json."""
    r = run_path(pid, run_id)
    return bool(r) and (r / "metrics" / "metrics.json").exists() and (r / "findings" / "findings.json").exists()


def project_status(pid: str) -> str:
    """empty | running | done | error (running mà tiến trình đã chết → error)."""
    s = read_status(pid)
    if s.get("state") == "running":
        if _alive(s.get("pid")):
            return "running"
        write_status(pid, state="error", message=s.get("message") or "tiến trình chạy bị dừng bất thường")
        return "error"
    if s.get("state") == "error":
        return "error"
    return "done" if run_complete(pid) else "empty"


def _metric(m: dict | None, key: str):
    v = (m or {}).get(key)
    return v.get("value") if isinstance(v, dict) else None


def run_summary(pid: str, info: dict | None = None) -> dict:
    """Số tóm tắt của LATEST run (cycle time, F1, nguồn bước, thời điểm) — chỉ đọc artifact."""
    info = info or load_project(pid)
    r = run_path(pid)
    out = {"run_id": r.name if r else None, "cycle_time_s": None, "f1_iou05": None, "source": None, "run_at": None}
    if not r:
        return out
    man = _read_json(r / "manifest.json") or {}
    met = _read_json(r / "metrics" / "metrics.json") or {}
    m = (met.get("trials") or {}).get(info["stem"])
    ev = _read_json(r / "steps" / f"{info['stem']}_eval.json") or {}
    out.update(cycle_time_s=_metric(m, "cycle_time_s"), f1_iou05=ev.get("f1_iou05"),
               source=(met.get("meta") or {}).get("source") or man.get("source"), run_at=man.get("created_at"))
    return out


def list_projects() -> list[dict]:
    """Mọi project (mới tạo trước): project.yaml + status + tóm tắt LATEST run + đường dẫn thumbnail."""
    root = projects_root()
    out = []
    for d in sorted(root.glob("*/project.yaml")) if root.exists() else []:
        try:
            info = yaml.safe_load(d.read_text(encoding="utf-8")) or {}
            pid = info["id"]
            thumb = media_path(pid, "thumb.jpg")
            out.append({**info, "status": project_status(pid), **run_summary(pid, info),
                        "thumb": str(thumb) if thumb.exists() else None})
        except Exception as e:             # 1 project hỏng không làm hỏng cả danh sách
            out.append({"id": d.parent.name, "name": d.parent.name, "status": "error", "error": str(e)})
    return sorted(out, key=lambda x: x.get("created_at") or "", reverse=True)


def thumb_data_uri(path: str | None) -> str | None:
    """thumb.jpg → data URI (bảng UI)."""
    if not path or not Path(path).exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ---------------------------------------------------------------- khoá job toàn app
def _job_path() -> Path:
    return projects_root() / ".job.json"


def current_job() -> dict | None:
    """Job đang chạy {project_id, pid, started_at} hoặc None (khoá mồ côi → tự dọn)."""
    j = _read_json(_job_path())
    if not j:
        return None
    if _alive(j.get("pid")):
        return j
    _job_path().unlink(missing_ok=True)
    return None


def acquire_job(pid: str, proc_pid: int) -> None:
    """Giữ khoá cho project pid (JobBusy nếu project khác đang chạy)."""
    j = current_job()
    if j and j.get("pid") != proc_pid:
        raise JobBusy(f"đang chạy dự án {j['project_id']} — chờ xong rồi chạy tiếp")
    _write_json(_job_path(), {"project_id": pid, "pid": proc_pid, "started_at": _now()})


def release_job(proc_pid: int) -> None:
    """Nhả khoá nếu đúng tiến trình đang giữ."""
    j = _read_json(_job_path())
    if j and j.get("pid") == proc_pid:
        _job_path().unlink(missing_ok=True)


# ---------------------------------------------------------------- benchmark + demo
def benchmark(pid: str) -> dict:
    """Số benchmark per-video của LATEST run (chỉ đọc artifact): cycle time, số bước, F1/MAE/sai số CT (khi có nhãn),
    detect_rate pose, pass-rate validator. Kèm chú thích baseline cố định."""
    info = load_project(pid)
    r = run_path(pid)
    out = {"model_line": MODEL_LINE, "note": BENCH_NOTE, "labels_available": bool(info.get("labels_available")),
           "rows": [], "run_id": r.name if r else None}
    if not r:
        return out
    st = info["stem"]
    met = _read_json(r / "metrics" / "metrics.json") or {}
    man = _read_json(r / "manifest.json") or {}
    ev = _read_json(r / "steps" / f"{st}_eval.json")
    ex = _read_json(r / "explained" / "explained.json") or {}
    m = (met.get("trials") or {}).get(st)
    rows = [("Cycle time", _metric(m, "cycle_time_s"), "s", "nguồn bước: " + str((met.get("meta") or {}).get("source")))]
    pose = (((man.get("trials") or {}).get(st) or {}).get("pose_meta") or {})
    rows.append(("Tỷ lệ phát hiện pose (detect_rate)", pose.get("detect_rate"), "", pose.get("model") or POSE_MODEL))
    if ev:
        rows += [("Số bước detect / nhãn", f"{ev.get('n_pred')} / {ev.get('n_oracle')}", "", "pred vs nhãn HA4M"),
                 ("F1@IoU0.5", ev.get("f1_iou05"), "", "theo bước"),
                 ("MAE thời lượng bước", ev.get("mae_step_dur_s"), "s", ""),
                 ("Sai số cycle time", ev.get("cycle_time_err_s"), "s", "pred − nhãn")]
    else:
        n_pred = None
        p = r / "steps" / f"{st}_pred.csv"
        if p.exists():
            import pandas as pd
            n_pred = int((pd.read_csv(p)["step_id"] > 0).sum())
        rows.append(("Số bước detect", n_pred, "", "không có nhãn để so"))
    if ex.get("pass_rate") is not None:
        rows.append(("Validator pass-rate (sau retry / lần 1)",
                     f"{ex['pass_rate'] * 100:.0f}% / {(ex.get('pass_rate_first_attempt') or 0) * 100:.0f}%", "",
                     f"{ex.get('provider')} {ex.get('llm_model') or ''}".strip()))
    out["rows"] = [{"label": a, "value": b, "unit": c, "note": d} for a, b, c, d in rows]
    return out


def demo_video_meta(pid: str) -> dict | None:
    """media/demo.json (nguồn nhãn của bản render) hoặc None nếu chưa render."""
    if not media_path(pid, "demo.mp4").exists():
        return None
    return _read_json(media_path(pid, "demo.json")) or {"source": None}


def demo_ready(pid: str) -> tuple[bool, list[str]]:
    """Dự án demo = LATEST run đủ artifact + có nhãn + demo.mp4 bản oracle + benchmark. Trả (ok, danh sách thiếu)."""
    try:
        info = load_project(pid)
    except ProjectError as e:
        return False, [str(e)]
    miss = []
    r = run_path(pid)
    st = info.get("stem")
    if not info.get("labels_available"):
        miss.append("nhãn chuẩn (labels_available=false)")
    if not r:
        return False, miss + ["LATEST run"]
    for rel_ in ("metrics/metrics.json", "findings/findings.json", "explained/explained.json"):
        if not (r / rel_).exists():
            miss.append(rel_)
    if info.get("labels_available") and not (r / "steps" / f"{st}_oracle.csv").exists():
        miss.append(f"steps/{st}_oracle.csv")
    if info.get("labels_available") and not (r / "steps" / f"{st}_eval.json").exists():
        miss.append("benchmark (steps eval)")
    vm = demo_video_meta(pid)
    if not vm or vm.get("source") != "oracle":
        miss.append("media/demo.mp4 (bản oracle)")
    return not miss, miss


# ---------------------------------------------------------------- adopt run cũ
def unadopted_runs(runs_dir: str | Path = "outputs/runs") -> list[str]:
    """Run trong outputs/runs/ chưa thuộc project nào."""
    rd = Path(runs_dir)
    if not rd.exists():
        return []
    owned = {p.name for p in projects_root().glob("*/runs/*") if p.is_dir()} if projects_root().exists() else set()
    return sorted(d.name for d in rd.iterdir() if d.is_dir() and (d / "manifest.json").exists() and d.name not in owned)


def adopt(run_id: str, runs_dir: str | Path = "outputs/runs") -> list[str]:
    """Bọc run cũ thành project (mỗi trial 1 project; copy run vào <project>/runs/, demo_<trial>.mp4 → media/demo.mp4)."""
    src = Path(runs_dir) / run_id
    man = _read_json(src / "manifest.json")
    if not man:
        raise ProjectError(f"không thấy run {run_id} trong {runs_dir}")
    met = _read_json(src / "metrics" / "metrics.json") or {}
    stems = sorted(set(met.get("trials") or {}) | set(man.get("trials") or {}))
    root = str(man.get("data_root") or DEFAULT_DATA_ROOT).split(",")[0]
    made = []
    for st in stems:
        existing = [p.parent.name for p in projects_root().glob("*/project.yaml")
                    if (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("stem") == st
                    and (p.parent / "runs" / run_id).exists()]
        if existing:
            continue
        m = re.fullmatch(r"(U\d+)_(V\d+)", st)
        uri = f"ha4m://{root}?subject={m.group(1)}&trial={m.group(2)}" if m else None
        try:
            info = create_project(uri, st) if uri else None
        except ProjectError:
            info = None
        if info is None:                    # nguồn gốc không còn → project chỉ đọc artifact
            pid = _unique(st)
            info = {"id": pid, "name": st, "stem": st, "kind": "adopted", "source_uri": uri, "station": DEFAULT_CONFIG,
                    "labels_available": bool(((man.get("trials") or {}).get(st) or {}).get("labels_available")),
                    "data_root": root, "created_at": _now()}
            save_project(info)
        info["adopted_from"] = run_id
        save_project(info)
        pid = info["id"]
        shutil.copytree(src, runs_root(pid) / run_id, dirs_exist_ok=True)
        (runs_root(pid) / "LATEST").write_text(run_id + "\n", encoding="utf-8")
        demo = src / "report" / f"demo_{st}.mp4"
        if demo.exists():
            media_path(pid, "demo.mp4").parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(demo, media_path(pid, "demo.mp4"))
            _write_json(media_path(pid, "demo.json"), {"source": man.get("source"), "run_id": run_id, "trial": st,
                                                       "adopted": True})
        write_status(pid, state="done" if run_complete(pid) else "empty", message=f"adopt từ {run_id}")
        made.append(pid)
    return made


def prepare_demo(trials: list[str], provider: str | None = None, data_root: str = DEFAULT_DATA_ROOT,
                 log: Callable[[str], None] = print) -> list[dict]:
    """Bảo đảm mỗi trial HA4M là 1 dự án demo đủ artifact (thiếu → chạy bù phần thiếu). Trả bảng READY/MISSING."""
    from kaizen.core import pipeline
    rows = []
    for st in trials:
        m = re.fullmatch(r"(U\d+)_(V\d+)", st.strip())
        if not m:
            rows.append({"trial": st, "project": None, "state": "MISSING", "missing": ["tên trial phải dạng U001_V001"]})
            continue
        pid = next((p["id"] for p in list_projects() if p.get("stem") == st and p.get("kind") in ("ha4m", "adopted")),
                   None)
        if pid is None:
            pid = create_project(f"ha4m://{data_root}?subject={m.group(1)}&trial={m.group(2)}", st)["id"]
            log(f"[demo] tạo dự án {pid}")
        ok, miss = demo_ready(pid)
        if not ok:
            only_video = all(x.startswith("media/demo.mp4") for x in miss)
            log(f"[demo] {pid}: thiếu {miss} → " + ("render bù video" if only_video else "chạy bù pipeline"))
            try:
                pipeline.run_project(pid, provider=provider, only="render" if only_video else None, log=log)
            except Exception as e:
                log(f"[demo] {pid}: lỗi khi chạy bù: {e}")
            ok, miss = demo_ready(pid)
        rows.append({"trial": st, "project": pid, "state": "READY" if ok else "MISSING", "missing": miss})
    return rows


def main() -> None:
    """CLI: list | adopt | prepare-demo."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a_ = sub.add_parser("adopt")
    a_.add_argument("run_id")
    a_.add_argument("--runs-dir", default="outputs/runs")
    d_ = sub.add_parser("prepare-demo")
    d_.add_argument("--trials", required=True, help="vd U001_V001,U001_V002")
    d_.add_argument("--provider", default=None, help="mặc định env LLM_PROVIDER (thiếu key → dry-run)")
    d_.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    a = ap.parse_args()
    if a.cmd == "list":
        for p in list_projects():
            print(f"{p['id']:<28} {STATE_VI.get(p['status'], p['status']):<10} {p.get('kind') or '?':<8} "
                  f"CT={p.get('cycle_time_s')} F1={p.get('f1_iou05')} run={p.get('run_id')}")
    elif a.cmd == "adopt":
        made = adopt(a.run_id, a.runs_dir)
        print(f"[projects] adopt {a.run_id} → {made or 'không có project mới (đã adopt trước đó)'}")
    else:
        rows = prepare_demo([t for t in a.trials.split(",") if t.strip()], a.provider, a.data_root)
        print(f"\n{'Trial':<12} {'Dự án':<16} {'Trạng thái':<9} Thiếu")
        for r_ in rows:
            print(f"{r_['trial']:<12} {str(r_['project']):<16} {r_['state']:<9} {', '.join(r_['missing']) or '—'}")
        sys.exit(0 if all(r_["state"] == "READY" for r_ in rows) else 1)


if __name__ == "__main__":
    main()

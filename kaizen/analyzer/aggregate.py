"""3M tổng hợp trên NHIỀU video (dự án) — tách khỏi dự án đơn lẻ: người đánh giá chọn video rồi mới tính.

    aggregate_3m(project_ids) -> Aggregate3M          # thuần tính toán, không LLM
    save_aggregate(agg) / load_aggregate(agg_id) / list_aggregates() / set_review(agg_id, note)

Đọc metrics.json của LATEST run mỗi dự án (dự án chưa xong → `skipped` kèm lý do). Dùng lại code sẵn có:
  - Mura: kaizen.analyzer.metrics.compute_mura (mean/std/CV cycle time + từng bước, trong/giữa subject, toàn bộ).
  - Findings tổng hợp: detect_findings với rules=[R1, R6] (R1 median+2·MAD cần ≥ 3 video; R6 CV cần ≥ 2 video
    cùng subject) — rule không đủ dữ liệu → `rules[].ran = False` + lý do (hệ thống biết giới hạn của nó).
  - Muda so sánh: bảng cycle time / %VA / pick / quãng đường / idle theo video; video nhanh nhất = tham chiếu.
  - Muri: gộp bảng góc cổ tay các video (ước lượng 2D, sàng lọc).
Mỗi số giữ unit + definition. Lưu outputs/aggregates/<agg_id>/{aggregate.json, findings.json} (env KAIZEN_AGGREGATES_DIR).
< 2 video hợp lệ → lỗi mềm (Aggregate3M.error = "cần ≥2 …"), không lưu.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kaizen.analyzer.findings import RULES, detect_findings
from kaizen.analyzer.metrics import compute_mura
from kaizen.core import projects

AGG_RULES = ("R1", "R6")
MUDA_COLS = [("cycle_time_s", "Cycle time", "s"), ("va_pct", "%VA", "%"), ("picks_total", "Số pick", "count/cycle"),
             ("distance_left", "Quãng đường trái", "px"), ("distance_right", "Quãng đường phải", "px"),
             ("idle_left_s", "Idle trái", "s"), ("idle_right_s", "Idle phải", "s")]
MURA_DEF = "mean/std (s) và CV = std/mean giữa các video (ddof=1); cần ≥ 2 video"
COVERAGE_LOW = 50.0


@dataclass
class Aggregate3M:
    """Kết quả 3M tổng hợp (nội dung aggregate.json; findings riêng ở findings.json)."""
    agg_id: str = ""
    created_at: str = ""
    projects: list[dict] = field(default_factory=list)      # {project_id, name, stem, run_id, source}
    skipped: list[dict] = field(default_factory=list)       # {project_id, reason}
    mura: dict = field(default_factory=dict)
    muda: dict = field(default_factory=dict)
    muri: dict = field(default_factory=dict)
    yamazumi: dict = field(default_factory=dict)            # {project_id: [{step_id, name, dur_s, va}]}
    rules: list[dict] = field(default_factory=list)         # {rule, description, ran, reason}
    findings: dict = field(default_factory=dict)            # nội dung findings.json
    review: dict = field(default_factory=lambda: {"note": "", "updated_at": None})
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True nếu tính được (≥ 2 video hợp lệ)."""
        return self.error is None

    def to_dict(self) -> dict:
        """aggregate.json (không gồm findings — file riêng)."""
        d = asdict(self)
        d.pop("findings")
        return d


def aggregates_root() -> Path:
    """env KAIZEN_AGGREGATES_DIR hoặc outputs/aggregates."""
    return Path(os.environ.get("KAIZEN_AGGREGATES_DIR") or "outputs/aggregates")


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _collect(project_ids: list[str]) -> tuple[dict[str, dict], list[dict], list[dict]]:
    """{project_id: metrics trial} + mô tả video + danh sách bỏ qua (chưa xong / thiếu trial)."""
    trials, used, skipped = {}, [], []
    for pid in project_ids:
        try:
            info = projects.load_project(pid)
        except projects.ProjectError as e:
            skipped.append({"project_id": pid, "reason": str(e)})
            continue
        status = projects.project_status(pid)
        if status != "done":
            skipped.append({"project_id": pid, "reason": f"chưa chạy xong (trạng thái: {projects.STATE_VI.get(status, status)})"})
            continue
        run = projects.run_path(pid)
        met = _read(run / "metrics" / "metrics.json") or {}
        m = (met.get("trials") or {}).get(info["stem"])
        if not m:
            skipped.append({"project_id": pid, "reason": "chưa phát hiện được bước hợp lệ nào trong video "
                            f"(run {run.name}) — không có số để so sánh"})
            continue
        trials[pid] = m
        used.append({"project_id": pid, "name": info.get("name") or pid, "stem": info["stem"], "run_id": run.name,
                     "source": (met.get("meta") or {}).get("source"), "labels_available": info.get("labels_available")})
    return trials, used, skipped


def _mv(m: dict, key: str):
    v = m.get(key)
    return v.get("value") if isinstance(v, dict) else None


def _muda(trials: dict[str, dict]) -> dict:
    rows = []
    for pid, m in trials.items():
        rows.append({"project_id": pid, **{k: _mv(m, k) for k, _, _ in MUDA_COLS}})
    ref = min(rows, key=lambda x: x["cycle_time_s"] if x["cycle_time_s"] is not None else float("inf"))["project_id"]
    ref_ct = trials[ref]["cycle_time_s"]["value"]
    for x in rows:
        x["is_reference"] = x["project_id"] == ref
        x["delta_vs_ref_s"] = round(x["cycle_time_s"] - ref_ct, 2) if x["cycle_time_s"] is not None else None
    any_m = next(iter(trials.values()))
    units = {k: (any_m.get(k) or {}).get("unit", u) for k, _, u in MUDA_COLS}
    defs = {k: (any_m.get(k) or {}).get("definition", "") for k, _, _ in MUDA_COLS}
    units["delta_vs_ref_s"], defs["delta_vs_ref_s"] = "s", "cycle time − cycle time video tham chiếu (nhanh nhất)"
    return {"rows": rows, "reference": ref, "labels": {k: lab for k, lab, _ in MUDA_COLS}, "units": units,
            "definitions": defs}


def _muri(trials: dict[str, dict]) -> dict:
    rows, note = [], ""
    for pid, m in trials.items():
        mw = m.get("muri_wrist") or {}
        note = note or mw.get("note") or ""
        for hand in ("left", "right"):
            h = mw.get(hand) or {}
            cov = _mv(h, "coverage_pct") if isinstance(h.get("coverage_pct"), dict) else h.get("coverage_pct")
            rows.append({"project_id": pid, "hand": hand, "available": bool(h.get("available")), "coverage_pct": cov,
                         "median_angle_deg": _mv(h, "median_angle_deg"), "dev_gt_thr_pct": _mv(h, "dev_gt_thr_pct"),
                         "dev_p90_deg": _mv(h, "dev_p90_deg"),
                         "low_coverage": cov is None or cov < COVERAGE_LOW})
    return {"rows": rows, "note": note or "ước lượng 2D, chỉ mang tính sàng lọc",
            "coverage_low_pct": COVERAGE_LOW, "unit": {"coverage_pct": "%", "median_angle_deg": "°",
                                                      "dev_gt_thr_pct": "%", "dev_p90_deg": "°"}}


def _mura(trials: dict[str, dict]) -> dict:
    mu = compute_mura(trials)
    names = {}
    for m in trials.values():
        for row in m["steps"]["rows"]:
            names.setdefault(int(row["step_id"]), row.get("name", ""))
    steps_cv = [{"step_id": int(sid), "name": names.get(int(sid), ""), **v}
                for sid, v in sorted(mu["all_trials"]["steps"].items(), key=lambda kv: int(kv[0]))]
    cvs = [s["cv"] for s in steps_cv if s.get("cv") is not None]
    top = max(cvs) if cvs else None
    for s in steps_cv:
        s["is_max"] = top is not None and s.get("cv") == top
    return {"definition": MURA_DEF, "unit": "s (CV không đơn vị)",
            "cycle_time": {**mu["all_trials"]["cycle_time"], "unit": "s", "definition": MURA_DEF},
            "steps_cv": steps_cv, "top_cv_steps": mu["top_cv_steps"], "within_subject": mu["within_subject"],
            "between_subjects": mu["between_subjects"]}


def aggregate_3m(project_ids: list[str], *, save: bool = True) -> Aggregate3M:
    """Tính 3M trên các dự án đã chọn (LATEST run mỗi dự án). < 2 video hợp lệ → lỗi mềm."""
    ids = list(dict.fromkeys(project_ids))
    trials, used, skipped = _collect(ids)
    now = dt.datetime.now()
    agg = Aggregate3M(created_at=now.isoformat(timespec="seconds"), projects=used, skipped=skipped)
    if len(trials) < 2:
        agg.error = f"cần ≥2 video đã chạy xong để tính 3M tổng hợp (hiện có {len(trials)})"
        return agg
    agg.mura = _mura(trials)
    agg.muda = _muda(trials)
    agg.muri = _muri(trials)
    agg.yamazumi = {pid: [{"step_id": int(r_["step_id"]), "name": r_.get("name", ""), "dur_s": r_["dur_s"],
                           "va": bool(r_.get("va"))} for r_ in sorted(m["steps"]["rows"], key=lambda x: x["step_id"])]
                    for pid, m in trials.items()}
    sources = {u["source"] for u in used}
    source = "oracle" if sources == {"oracle"} else "pred"
    rules = [r_ for r_ in RULES if r_.id in AGG_RULES]
    metrics_all = {"meta": {"source": source}, "trials": trials, "mura": compute_mura(trials)}
    res = detect_findings(metrics_all, {}, rules=rules)
    fired = {f.rule for f in res.findings}
    for r_ in rules:
        note = next((n for n in res.notes if n.startswith(f"{r_.id} bỏ qua")), None)
        agg.rules.append({"rule": r_.id, "description": r_.description, "ran": note is None,
                          "fired": r_.id in fired, "reason": note.split(":", 1)[1].strip() if note else ""})
    agg.findings = res.to_json_dict(source if len(sources) == 1 else "mixed")
    if len(sources) > 1:
        agg.findings["notes"] = agg.findings.get("notes", []) + [
            "nguồn bước trộn lẫn oracle + pred giữa các video — so sánh có thể lệch do sai số phân đoạn"]
    agg.agg_id = f"{now:%Y%m%d-%H%M%S}_{len(trials)}v"
    if save:
        save_aggregate(agg)
    return agg


def save_aggregate(agg: Aggregate3M) -> Path:
    """Ghi aggregates/<agg_id>/{aggregate.json, findings.json} (agg_id trùng → -2, -3)."""
    root = aggregates_root()
    base, k = agg.agg_id, 2
    while (root / agg.agg_id).exists():
        agg.agg_id, k = f"{base}-{k}", k + 1
    d = root / agg.agg_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregate.json").write_text(json.dumps(agg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "findings.json").write_text(json.dumps(agg.findings, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def load_aggregate(agg_id: str) -> Aggregate3M:
    """Mở lại aggregate đã lưu (FileNotFoundError nếu không có)."""
    d = aggregates_root() / agg_id
    a = _read(d / "aggregate.json")
    if a is None:
        raise FileNotFoundError(f"không có aggregate {agg_id} trong {aggregates_root()}")
    a = {k: v for k, v in a.items() if k in Aggregate3M.__dataclass_fields__}
    return Aggregate3M(**a, findings=_read(d / "findings.json") or {})


def list_aggregates() -> list[dict]:
    """Aggregate đã lưu (mới nhất trước): {agg_id, created_at, n_videos, projects, has_review}."""
    root = aggregates_root()
    out = []
    for p in sorted(root.glob("*/aggregate.json"), reverse=True) if root.exists() else []:
        a = _read(p) or {}
        out.append({"agg_id": a.get("agg_id") or p.parent.name, "created_at": a.get("created_at"),
                    "n_videos": len(a.get("projects") or []),
                    "projects": [x["project_id"] for x in a.get("projects") or []],
                    "has_review": bool((a.get("review") or {}).get("note"))})
    return out


def set_review(agg_id: str, note: str) -> Aggregate3M:
    """Ghi ghi chú đánh giá của người xem vào aggregate.json."""
    agg = load_aggregate(agg_id)
    agg.review = {"note": note, "updated_at": dt.datetime.now().isoformat(timespec="seconds")}
    d = aggregates_root() / agg_id
    (d / "aggregate.json").write_text(json.dumps(agg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return agg

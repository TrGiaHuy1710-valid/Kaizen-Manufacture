"""[08] Tổng hợp → <run>/report/report.md + ảnh PNG trong <run>/report/report_assets/.

Mục: 1 Tóm tắt & đề xuất (≤ 6 dòng) · 2 Tổng quan trial · 3 Bảng bước + TG chuẩn/Lệch %/Nguồn chuẩn ·
4 Timeline + Yamazumi · 5 3M tổng hợp (hoãn khi < 2 trial) · 6 Muri · 7 Findings (Quan sát/Giả thuyết/Đề xuất +
ECRS + Đo lại + Nguồn) · 8 Tài liệu tham chiếu (mọi [WI-EGT-001 · tr.1] ↔ chunk_id) · 9 Giả định & giới hạn.
TG chuẩn + ngưỡng + citation: findings.json["standards"] (06); findings.json cũ chưa có → tra RAG store tại chỗ.

    python src/08_report.py [--force] [--rag-store data/rag_store] [--no-rag]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.analyzer.report import (cycle_dist, evidence_frame, gantt, md_table, trace_check,  # noqa: E402,F401
                                    write_report, yamazumi)
from kaizen.core.cli import base_parser, setup, stage_dir, stage_file  # noqa: E402
from kaizen.core.config import load_config  # noqa: E402
from kaizen.core.io import can_write, read_json  # noqa: E402

TAG = "08"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--metrics", default=None, help="mặc định <run>/metrics/metrics.json")
    ap.add_argument("--findings", default=None, help="mặc định <run>/findings/findings.json")
    ap.add_argument("--explained", default=None, help="mặc định <run>/explained/explained.json")
    ap.add_argument("--steps-dir", default=None)
    ap.add_argument("--events-dir", default=None)
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--inspect", default="outputs/inspect_report.md")
    ap.add_argument("--out", default=None, help="mặc định <run>/report/report.md")
    ap.add_argument("--rag-store", default=None, help="chỉ dùng khi findings.json chưa có 'standards' (run cũ)")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--no-rag", action="store_true", help="run cũ: không tra tài liệu (std chỉ từ station.yaml)")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    steps_dir = stage_dir(ctx, a.steps_dir, "steps", "outputs/steps")
    events_dir = stage_dir(ctx, a.events_dir, "events", "outputs/events")
    features_dir = stage_dir(ctx, a.features_dir, "features", "outputs/features")
    f_metrics = stage_file(ctx, a.metrics, "metrics", "metrics.json", "outputs/metrics/metrics.json")
    f_findings = stage_file(ctx, a.findings, "findings", "findings.json", "outputs/findings/findings.json")
    f_explained = stage_file(ctx, a.explained, "explained", "explained.json", "outputs/findings/explained.json")
    out = stage_file(ctx, a.out, "report", "report.md", "outputs/report.md")
    man = ctx.manifest if ctx else {}
    if not can_write(out, a.force, TAG):
        return
    cfg = load_config(a.config)
    fdoc = read_json(f_findings) if f_findings.exists() else None
    standards = None
    if fdoc is not None and "standards" not in fdoc:      # run cũ (trước R8) → tra chuẩn tại chỗ
        from kaizen.rag import report_queries
        standards = report_queries.resolve_standards(cfg, None if a.no_rag else report_queries.open_store(a.rag_store, a.embedder))
    assets, n_trials = write_report(out, cfg, read_json(f_metrics), fdoc,
                                    read_json(f_explained) if f_explained.exists() else None,
                                    steps_dir, events_dir, features_dir, a.data_root, a.inspect, a.subjects, a.trials,
                                    ctx.run_id if ctx else None, man, standards=standards)
    missing, unused = trace_check(out.read_text(encoding="utf-8"))
    print(f"[{TAG}] đã ghi {out} + {len(list(assets.glob('*.png')))} ảnh trong {assets}")
    print(f"[{TAG}] truy vết trích dẫn: " + ("khớp 2 chiều ✓" if not (missing or unused)
                                          else f"thiếu dòng tham chiếu {sorted(missing)}, dòng thừa {sorted(unused)}"))
    if ctx:
        ctx.record_stage("08_report", report=str(out), n_trials=n_trials)


if __name__ == "__main__":
    main()

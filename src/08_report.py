"""[08] Tổng hợp → outputs/report.md + ảnh PNG trong outputs/report_assets/.

Gồm: bảng tổng theo trial, sai số vs oracle, Gantt timeline (bước oracle/predicted + event 2 tay),
Yamazumi (màu VA/NVA, nhãn số bước), phân bố cycle time + bảng CV (Mura), Muri, từng finding kèm
giải thích LLM + trạng thái validator + frame bằng chứng (vẽ zone + cổ tay), mục "Giả định & giới hạn".

    python src/08_report.py [--force]
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, load_config, parse_stem, read_json, select_stems  # noqa: E402
import ha4m_io  # noqa: E402

TAG = "08"
# Bảng màu tham chiếu (dataviz reference palette, light mode) — thứ tự slot cố định.
C_VA, C_NVA, C_MOVE, C_WAIT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_GRAY, C_UNCL = "#c3c2b7", "#8d8c86"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
EVENT_GROUP = {"pick": ("Lấy ở khay", C_NVA), "reach": ("Lấy ở khay", C_NVA), "search": ("Lấy ở khay", C_NVA),
               "assemble": ("Lắp / thao tác", C_VA), "place": ("Lắp / thao tác", C_VA),
               "transfer": ("Di chuyển ngoài zone", C_MOVE),
               "idle": ("Chờ (idle/pause)", C_WAIT), "pause": ("Chờ (idle/pause)", C_WAIT),
               "lost": ("Mất dấu", C_GRAY)}


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def legend_handles(items):
    from matplotlib.patches import Patch
    return [Patch(facecolor=c, edgecolor="none", label=l) for l, c in items]


def gantt(st: str, steps: dict[str, pd.DataFrame], ev: pd.DataFrame | None, va: dict[int, bool], out: Path) -> None:
    plt = plt_setup()
    rows = [(f"Bước ({k})", df) for k, df in steps.items()]
    hands = [("Tay trái", "left"), ("Tay phải", "right")] if ev is not None else []
    n = len(rows) + len(hands)
    fig, ax = plt.subplots(figsize=(14, 0.75 * n + 1.3))
    ylabels = []
    for i, (lab, df) in enumerate(rows):
        y = n - 1 - i
        for rw in df.itertuples():
            c = C_UNCL if rw.step_id == 0 else (C_VA if va.get(int(rw.step_id), False) else C_NVA)
            ax.broken_barh([(rw.start_s, rw.dur_s)], (y - 0.35, 0.7), facecolors=c, edgecolor=SURFACE, linewidth=2)
            if rw.dur_s > 1.2:
                txt = "?" if rw.step_id == 0 else (f"{rw.step_id}" + (f".{rw.repetition}" if rw.repetition else ""))
                ax.text(rw.start_s + rw.dur_s / 2, y, txt, ha="center", va="center", color="white", fontsize=8,
                        fontweight="bold")
        ylabels.append((y, lab))
    for j, (lab, hand) in enumerate(hands):
        y = n - 1 - len(rows) - j
        he = ev[ev["hand"] == hand]
        for et, g in he.groupby("event_type"):
            c = EVENT_GROUP.get(et, ("?", C_GRAY))[1]
            ax.broken_barh(list(zip(g["start_s"], g["dur_s"])), (y - 0.35, 0.7), facecolors=c, edgecolor="none")
        ylabels.append((y, lab))
    ax.set_yticks([y for y, _ in ylabels], [l for _, l in ylabels])
    ax.set_xlabel("thời gian (s)")
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    items = [("Bước VA", C_VA), ("Bước NVA", C_NVA), ("Unclassified", C_UNCL)]
    if hands:
        items += [("Tay: lấy ở khay", C_NVA), ("Tay: lắp/thao tác", C_VA), ("Tay: di chuyển", C_MOVE),
                  ("Tay: chờ", C_WAIT), ("Tay: mất dấu", C_GRAY)]
    ax.legend(handles=legend_handles(items), loc="upper center", bbox_to_anchor=(0.5, -0.18 if n > 3 else -0.3),
              ncol=8, frameon=False, fontsize=8)
    ax.set_title(f"{st} — timeline bước và event cổ tay", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def yamazumi(trials: dict, va: dict[int, bool], target: float | None, out: Path) -> None:
    plt = plt_setup()
    names = list(trials)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(names) + 3), 6))
    for x, st in enumerate(names):
        bottom = 0.0
        for row in sorted(trials[st]["steps"]["rows"], key=lambda r: r["step_id"]):
            d = row["dur_s"]
            c = C_VA if va.get(row["step_id"], False) else C_NVA
            ax.bar(x, d, bottom=bottom, width=0.6, color=c, edgecolor=SURFACE, linewidth=2)
            if d >= 2.5:
                ax.text(x, bottom + d / 2, f"{row['step_id']}", ha="center", va="center", color="white", fontsize=8,
                        fontweight="bold")
            bottom += d
        ax.text(x, bottom + 0.8, f"{trials[st]['cycle_time_s']['value']:.1f}s", ha="center", va="bottom",
                color=INK, fontsize=9)
    if target:
        ax.axhline(target, color=INK, ls="--", lw=1)
        ax.text(len(names) - 0.5, target, f" mục tiêu {target:g}s", va="bottom", ha="right", color=INK, fontsize=8)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("thời lượng bước cộng dồn (s)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles([("Bước VA", C_VA), ("Bước NVA", C_NVA)]), loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    ax.set_title("Yamazumi — thời lượng từng bước (số = id bước)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def cycle_dist(trials: dict, out: Path) -> None:
    plt = plt_setup()
    v = np.array([m["cycle_time_s"]["value"] for m in trials.values()], float)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if len(v) >= 3:
        ax.hist(v, bins=min(12, max(3, len(v) // 2)), color=C_VA, edgecolor=SURFACE, linewidth=2)
        ax.set_ylabel("số trial")
    else:
        ax.scatter(v, np.zeros_like(v), s=60, color=C_VA, edgecolor=SURFACE, linewidth=2, zorder=3)
        for x, st in zip(v, trials):
            ax.annotate(f"{st}\n{x:.1f}s", (x, 0), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(v.min() - 10, v.max() + 10)
        ax.text(0.01, 0.92, f"n = {len(v)} trial — chưa đủ để vẽ histogram (cần ≥ 3)", transform=ax.transAxes,
                fontsize=8, color=INK2)
    ax.set_xlabel("cycle time (s)")
    ax.set_title("Phân bố cycle time", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def evidence_frame(data_root: str, st: str, t: float, cfg: dict, feat: pd.DataFrame | None, out: Path,
                   label: str) -> bool:
    """Đọc lại ảnh tại t, vẽ zone + vết cổ tay ±1s + vị trí hiện tại."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from pick_zones import draw_zones
    sub, tri = parse_stem(st)
    fidx = int(round(t * cfg["fps"]))
    try:
        img = ha4m_io.read_frame(data_root, sub, tri, fidx)
    except FileNotFoundError:
        img = None
    if img is None:
        return False
    draw_zones(img, cfg.get("zones", {}), thick=2)
    if feat is not None:
        w = feat[(feat["t"] >= t - 1) & (feat["t"] <= t + 1)]
        for p, c in (("lw", (91, 73, 209)), ("rw", (214, 120, 42))):
            pts = w[[f"{p}_x", f"{p}_y"]].dropna().to_numpy(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, c, 3)
            now = feat.iloc[(feat["t"] - t).abs().argsort()[:1]]
            x, y = now[f"{p}_x"].iloc[0], now[f"{p}_y"].iloc[0]
            if not np.isnan(x):
                cv2.circle(img, (int(x), int(y)), 12, c, -1)
                cv2.circle(img, (int(x), int(y)), 12, (255, 255, 255), 2)
                cv2.putText(img, "L" if p == "lw" else "R", (int(x) + 14, int(y) - 10), 0, 1.0, c, 3)
    cv2.rectangle(img, (0, 0), (img.shape[1], 60), (20, 20, 20), -1)
    cv2.putText(img, f"{label}  {st}  t={t:.2f}s  frame {fidx}", (15, 42), 0, 1.2, (255, 255, 255), 2)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), img)
    return True


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> list[str]:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |")
    return lines


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--metrics", default="outputs/metrics/metrics.json")
    ap.add_argument("--findings", default="outputs/findings/findings.json")
    ap.add_argument("--explained", default="outputs/findings/explained.json")
    ap.add_argument("--steps-dir", default="outputs/steps")
    ap.add_argument("--events-dir", default="outputs/events")
    ap.add_argument("--features-dir", default="outputs/features")
    ap.add_argument("--inspect", default="outputs/inspect_report.md")
    ap.add_argument("--out", default="outputs/report.md")
    a = ap.parse_args()
    out = Path(a.out)
    if not can_write(out, a.force, TAG):
        return
    assets = out.parent / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(out.parent).as_posix()  # noqa: E731

    cfg = load_config(a.config)
    metrics = read_json(Path(a.metrics))
    trials = {st: m for st, m in metrics["trials"].items() if st in select_stems(list(metrics["trials"]), a.subjects, a.trials)}
    fdoc = read_json(Path(a.findings)) if Path(a.findings).exists() else {"findings": [], "notes": []}
    edoc = read_json(Path(a.explained)) if Path(a.explained).exists() else {"explanations": [], "mode": "none"}
    exps = {e["finding_id"]: e for e in edoc.get("explanations", [])}
    source = metrics["meta"]["source"]
    va = {int(s["id"]): bool(s.get("va", False)) for s in cfg.get("steps", [])}
    pxu = "cm" if cfg.get("px_per_cm") else "px"

    md = ["# Báo cáo Kaizen Motion Analyzer — HA4M", "",
          f"- Sinh lúc: {dt.datetime.now():%Y-%m-%d %H:%M} · nguồn bước: **{source}** "
          f"({'nhãn HA4M — ground truth' if source == 'oracle' else 'dự đoán từ keypoints'}) · "
          f"{len(trials)} trial · subject: {', '.join(metrics['meta']['subjects'])}",
          f"- Mọi con số do code tính (02→06). LLM chỉ giải thích (07), có validator kiểm tra số & trích dẫn.", ""]

    # 1. Bảng tổng
    md += ["## 1. Tổng quan theo trial", ""]
    rows = []
    for st, m in trials.items():
        ev = m.get("eval_vs_oracle", {})
        rows.append(dict(trial=st, ct=m["cycle_time_s"]["value"], va=m["va_pct"]["value"],
                         il=m.get("idle_left_s", {}).get("value"), ir=m.get("idle_right_s", {}).get("value"),
                         ib=m.get("idle_both_s", {}).get("value"), pk=m.get("picks_total", {}).get("value"),
                         dl=m.get("distance_left", {}).get("value"), dr=m.get("distance_right", {}).get("value"),
                         err=ev.get("cycle_time_err_s"), f1=ev.get("f1_iou05")))
    md += md_table(rows, [("trial", "Trial"), ("ct", "Cycle time (s)"), ("va", "%VA"), ("il", "Idle trái (s)"),
                          ("ir", "Idle phải (s)"), ("ib", "Idle cả 2 tay (s)"), ("pk", "Số pick"),
                          ("dl", f"Quãng đường trái ({pxu})"), ("dr", f"Quãng đường phải ({pxu})"),
                          ("err", "Sai số CT vs oracle (s)"), ("f1", "F1@IoU0.5")])
    md += ["", "_%VA = Σ thời lượng bước va=true / cycle time (phân loại VA theo `station.yaml`, cần kỹ sư IE duyệt)._", ""]

    # 2. Eval
    evals = {st: read_json(Path(a.steps_dir) / f"{st}_eval.json") for st in trials
             if (Path(a.steps_dir) / f"{st}_eval.json").exists()}
    if evals:
        md += ["## 2. Độ chính xác phân đoạn bước (predicted vs nhãn HA4M)", ""]
        md += md_table([dict(trial=st, **{k: v for k, v in e.items() if not isinstance(v, (list, dict))})
                        for st, e in evals.items()],
                       [("trial", "Trial"), ("n_pred", "Bước dự đoán"), ("n_oracle", "Bước nhãn"),
                        ("n_unclassified", "Unclassified"), ("mae_step_dur_s", "MAE thời lượng (s)"),
                        ("mae_start_s", "MAE start (s)"), ("mae_end_s", "MAE end (s)"),
                        ("cycle_time_err_s", "Sai số CT (s)"), ("f1_iou05", "F1@IoU0.5")])
        for st, e in evals.items():
            md += ["", f"<details><summary>{st}: sai số từng bước</summary>", ""]
            md += md_table(e["per_step"], [("step_id", "Bước"), ("repetition", "Lần"), ("start_err_s", "Lệch start (s)"),
                                           ("end_err_s", "Lệch end (s)"), ("dur_err_s", "Lệch thời lượng (s)"),
                                           ("iou", "IoU")])
            md += ["", f"Bước không có zone (không làm mốc được): {e.get('no_zone') or 'không'}; "
                       f"không tìm thấy: {e.get('not_found') or 'không'}", "", "</details>", ""]

    # 3. Timeline
    md += ["## 3. Timeline (Gantt)", ""]
    for st in trials:
        steps = {}
        for k in ("oracle", "pred"):
            p = Path(a.steps_dir) / f"{st}_{k}.csv"
            if p.exists():
                steps[k] = pd.read_csv(p)
        evp = Path(a.events_dir) / f"{st}.csv"
        ev = pd.read_csv(evp) if evp.exists() else None
        png = assets / f"gantt_{st}.png"
        gantt(st, steps, ev, va, png)
        md += [f"![Gantt {st}]({rel(png)})", ""]
        sp = Path(a.features_dir) / f"{st}_speed.png"
        if sp.exists():
            link = Path(os.path.relpath(sp.resolve(), out.parent.resolve())).as_posix()
            md += [f"<details><summary>Tốc độ cổ tay {st}</summary>", "", f"![speed {st}]({link})", "", "</details>", ""]

    # 4. Yamazumi
    yz = assets / "yamazumi.png"
    yamazumi(trials, va, cfg.get("cycle_time_target_s"), yz)
    md += ["## 4. Yamazumi", "", f"![Yamazumi]({rel(yz)})", ""]
    first = next(iter(trials.values()))
    md += [f"Chi tiết bước — {next(iter(trials))}:", ""]
    md += md_table(first["steps"]["rows"], [("step_id", "Bước"), ("name", "Tên"), ("va", "VA"), ("n_rep", "Lần"),
                                            ("dur_s", "Thời lượng (s)"), ("pct_cycle", "% chu kỳ"),
                                            ("left_idle_pct", "Idle trái %"), ("right_idle_pct", "Idle phải %"),
                                            ("left_active_pct", "Active trái %"), ("right_active_pct", "Active phải %")])
    md.append("")

    # 5. Mura
    cd = assets / "cycle_time_dist.png"
    cycle_dist(trials, cd)
    mu = metrics["mura"]
    md += ["## 5. Mura — biến động", "", f"![Cycle time]({rel(cd)})", ""]
    ct = mu["all_trials"]["cycle_time"]
    md += [f"- Cycle time toàn bộ: n={ct['n']}, mean={ct['mean']}s, std={ct['std']}, CV={ct['cv']}"
           + (f" ({ct.get('note')})" if ct.get("note") else "")]
    for sub, v in mu["within_subject"].items():
        c = v["cycle_time"]
        md.append(f"- Trong subject {sub}: n={c['n']}, mean={c['mean']}s, CV={c['cv']}" + (f" ({c.get('note')})" if c.get("note") else ""))
    b = mu["between_subjects"]["cycle_time"]
    md.append(f"- Giữa các subject: n={b['n']}, CV={b['cv']}" + (f" ({b.get('note')})" if b.get("note") else ""))
    if mu.get("top_cv_steps"):
        md += ["", "Top bước CV cao nhất:", ""]
        md += md_table(mu["top_cv_steps"], [("step_id", "Bước"), ("cv", "CV")])
    md.append("")

    # 6. Muri
    md += ["## 6. Muri — tư thế cổ tay (ước lượng 2D, chỉ mang tính sàng lọc)", ""]
    mrows = []
    for st, m in trials.items():
        for hand in ("left", "right"):
            h = m.get("muri_wrist", {}).get(hand, {})
            if h.get("available"):
                mrows.append(dict(trial=st, hand=hand, cov=h["coverage_pct"]["value"], med=h["median_angle_deg"]["value"],
                                  pct=h["dev_gt_thr_pct"]["value"], p90=h["dev_p90_deg"]["value"]))
            else:
                mrows.append(dict(trial=st, hand=hand, cov=h.get("coverage_pct")))
    md += md_table(mrows, [("trial", "Trial"), ("hand", "Tay"), ("cov", "% frame đo được"), ("med", "Góc trung vị (°)"),
                           ("pct", f"% thời gian lệch > {cfg.get('muri_dev_deg', 30)}°"), ("p90", "P90 độ lệch (°)")])
    md.append("")

    # 7. Findings
    fs = fdoc.get("findings", [])
    md += ["## 7. Findings (rule-based) + giải thích LLM", ""]
    if edoc.get("pass_rate") is not None:
        md += [f"**Validator LLM:** {edoc['n_passed']}/{len(edoc['explanations'])} explanation đạt "
               f"(pass rate {edoc['pass_rate'] * 100:.0f}%) — chế độ `{edoc.get('mode')}`"
               + (f", model `{edoc.get('llm_model')}`" if edoc.get("llm_model") else "") + ".", ""]
    for n in fdoc.get("notes", []):
        md.append(f"- _{n}_")
    if fdoc.get("notes"):
        md.append("")
    if not fs:
        md += ["Không có finding nào vượt ngưỡng.", ""]
    feats = {}
    for f in fs:
        e = f["evidence"]
        st = e["trials"][0]
        md += [f"### {f['id']} · {f['type']} · rule {f['rule']} · severity **{f['severity']}** · confidence {f['confidence']}", "",
               f"- Bằng chứng: **{e['metric']} = {e['value']} {e['unit']}** · trial {', '.join(e['trials'])}",
               f"- Ảnh hưởng ước tính: {f['impact_s_per_cycle']} s/chu kỳ · điểm xếp hạng {f['rank_score']}"]
        ctx = {k: v for k, v in e.items() if k not in ("metric", "value", "unit", "trials", "timestamps")}
        if ctx:
            md.append(f"- Ngữ cảnh: `{ctx}`")
        if e.get("timestamps"):
            md.append(f"- Mốc thời gian (s): {e['timestamps'][:5]}")
        x = exps.get(f["id"])
        if x:
            v = x["validation"]
            badge = "✅ passed" if v["status"] == "passed" else (
                f"❌ failed (citation sai: {v.get('bad_citations')}, số không có trong finding: {v.get('unsupported_numbers')}"
                + (f", {v['reason']}" if v.get("reason") else "") + ")")
            md += ["", f"> **Quan sát:** {x.get('observation', '')}  ",
                   f"> **Giả thuyết:** {x.get('hypothesis', '')}  ",
                   f"> **Đề xuất:** {x.get('recommendation', '')}  ",
                   f"> **Trích dẫn:** {', '.join(x.get('citations', [])) or '—'} · **Validator:** {badge}"]
        if st not in feats:
            fp = Path(a.features_dir) / f"{st}.parquet"
            feats[st] = pd.read_parquet(fp) if fp.exists() else None
        if e.get("timestamps"):
            t0, t1 = e["timestamps"][0]
            t = (t0 + t1) / 2
        else:
            w = trials.get(st, {}).get("window_s", [0, 0])
            t = (w[0] + w[1]) / 2
        png = assets / f"evidence_{f['id']}.png"
        if evidence_frame(a.data_root, st, t, cfg, feats[st], png, f["id"]):
            md += ["", f"![{f['id']}]({rel(png)})"]
        md.append("")

    # 8. Giả định & giới hạn
    md += ["## 8. Giả định & giới hạn (tự sinh)", ""]
    lim = []
    insp = Path(a.inspect).read_text(encoding="utf-8") if Path(a.inspect).exists() else ""
    fps_note = "xác nhận bằng DeviceTimeStamp trong tên file và paper HA4M" if "DeviceTimeStamp" in insp else "GIẢ ĐỊNH (chưa kiểm chứng)"
    lim.append(f"fps = {cfg['fps']} — {fps_note}; t = frame_idx / fps.")
    lim.append("Quãng đường tính theo **pixel** (px_per_cm = null) — không so sánh được giữa camera/khoảng cách khác nhau."
               if not cfg.get("px_per_cm") else f"Quy đổi px→cm với px_per_cm = {cfg['px_per_cm']} (phẳng, bỏ qua phối cảnh).")
    lim.append("Bước lấy từ **nhãn HA4M (oracle)** — số liệu Muda/Mura phản ánh nhãn, không phản ánh sai số CV."
               if source == "oracle" else
               "Bước **dự đoán** bằng rule zone/pick (baseline v1) — xem mục 2 cho sai số; bước không có khay đặc trưng bị gộp vào 'unclassified'.")
    lim.append(f"Ngưỡng moving/still = {cfg['moving_speed_px_s']} px/s, min event {cfg['min_event_s']}s, idle khi đứng yên ngoài zone > "
               f"{cfg.get('idle_still_s', 1.5)}s — chỉnh tay theo đồ thị tốc độ, chưa hiệu chuẩn hệ thống.")
    lim.append("Zone là vùng **cổ tay** khi với vào khay, phác tay trên 1 video; vùng lắp (assembly_area) rộng nên tay đứng yên "
               "trước ngực được tính là 'lắp', không phải idle — idle có thể bị đánh giá thấp.")
    lim.append("Muri: góc cổ tay **2D** từ camera đặt chéo, chỉ mang tính sàng lọc; tay bị che khuất (thường tay trái) có độ phủ thấp.")
    lim.append("Pose: MediaPipe (2D), 1 người/khung; tay người gán nhãn ở mép trái được loại bằng ROI. Cổ tay bị che → nội suy ≤ 15 frame, dài hơn bỏ qua.")
    for st in trials:
        kc = out.parent / f"kinect_compare_{st}.json"
        if kc.exists():
            k = read_json(kc)
            lim.append(f"Đối chứng Kinect ({st}): lệch cổ tay so với skeleton Azure Kinect — phải: trung vị {k['right']['median_px']}px, "
                       f"P90 {k['right']['p90_px']}px, {k['right']['pct_within_50px']}% frame < 50px; trái: trung vị "
                       f"{k['left']['median_px']}px, P90 {k['left']['p90_px']}px, {k['left']['pct_within_50px']}% < 50px "
                       "→ số liệu tay trái kém tin cậy hơn (hay bị thân che).")
    n_sub = len(metrics["meta"]["subjects"])
    if len(trials) < 3:
        lim.append(f"Chỉ có {len(trials)} trial / {n_sub} subject: R1 (bước chậm, median+2·MAD) và R6 (CV cycle time) không chạy được; "
                   "Mura chưa có ý nghĩa thống kê — cần thêm video HA4M.")
    lim.append("%VA phụ thuộc phân loại va trong station.yaml (giả định, cần kỹ sư IE duyệt). Idle không đồng nghĩa lãng phí (KB-16).")
    if edoc.get("mode") == "dry-run":
        lim.append("Giải thích LLM đang ở chế độ **dry-run** (văn bản giả) — chạy `07_explain.py` với GEMINI_API_KEY để có giải thích thật.")
    md += [f"- {x}" for x in lim]
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[{TAG}] đã ghi {out} + {len(list(assets.glob('*.png')))} ảnh trong {assets}")


if __name__ == "__main__":
    main()

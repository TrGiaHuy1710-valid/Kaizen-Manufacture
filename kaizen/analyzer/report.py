"""Báo cáo Markdown + hình (stage 08) — chỉ trình bày số đã có trong artifact (không tính chỉ số mới).

Thứ tự mục (report truy vết được — mọi chuẩn/khuyến nghị chỉ rõ tài liệu + trang):
  1. Tóm tắt & đề xuất (≤ 6 dòng: top ≤ 3 finding + tổng tiềm năng s/chu kỳ)
  2. Tổng quan trial (+ độ chính xác phân đoạn nếu có eval)
  3. Bảng bước + TG chuẩn (s) · Lệch % · Nguồn chuẩn (lệch vượt ngưỡng in đậm)
  4. Timeline (Gantt) + Yamazumi
  5. 3M tổng hợp (Mura giữa trial/subject, phân bố cycle time, R1/R6) — HOÃN khi run < 2 trial (1 dòng)
  6. Muri per-video (sàng lọc 2D)
  7. Findings: bằng chứng · Quan sát/Giả thuyết/Đề xuất + ECRS + Đo lại · Nguồn · ảnh bằng chứng
  8. Tài liệu tham chiếu: mọi nhãn [WI-EGT-001 · tr.1] trong report ↔ 1 dòng (chunk_id, nơi dùng)
  9. Giả định & giới hạn
Chuẩn/ngưỡng + citation đọc từ findings.json["standards"] (06), citation của giải thích từ explained.json
["citations_detail"] (07) — module này KHÔNG truy vấn RAG (analyzer không phụ thuộc rag/llm).
"""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pandas as pd

from kaizen.core.io import parse_stem, read_json, select_stems
from kaizen.skeleton import ha4m_io
from kaizen.skeleton.overlay import draw_zones

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


def plt_setup() -> ModuleType:
    """matplotlib Agg + style bảng màu tham chiếu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def legend_handles(items: list[tuple[str, str]]) -> list:
    """[(nhãn, màu)] → Patch cho legend."""
    from matplotlib.patches import Patch
    return [Patch(facecolor=c, edgecolor="none", label=l) for l, c in items]


def gantt(st: str, steps: dict[str, pd.DataFrame], ev: pd.DataFrame | None, va: dict[int, bool], out: Path) -> None:
    """Gantt bước (oracle/pred) + event 2 tay → PNG."""
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
    """Bar chồng thời lượng bước theo trial (+ đường mục tiêu) → PNG."""
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
        unc = (trials[st].get("unclassified_s") or {}).get("value") or 0
        if unc > 0:
            ax.bar(x, unc, bottom=bottom, width=0.6, color=C_UNCL, edgecolor=SURFACE, linewidth=2)
            if unc >= 2.5:
                ax.text(x, bottom + unc / 2, "?", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
            bottom += unc
        ax.text(x, bottom + 0.8, f"{trials[st]['cycle_time_s']['value']:.1f}s", ha="center", va="bottom",
                color=INK, fontsize=9)
    if target:
        ax.axhline(target, color=INK, ls="--", lw=1)
        ax.text(len(names) - 0.5, target, f" mục tiêu {target:g}s", va="bottom", ha="right", color=INK, fontsize=8)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("thời lượng bước cộng dồn (s)")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    items = [("Bước VA", C_VA), ("Bước NVA", C_NVA)]
    if any(((m.get("unclassified_s") or {}).get("value") or 0) > 0 for m in trials.values()):
        items.append(("Unclassified (?)", C_UNCL))
    ax.legend(handles=legend_handles(items), loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
    ax.set_title("Yamazumi — thời lượng bước (số = id, ? = chưa gán)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def cycle_dist(trials: dict, out: Path) -> None:
    """Phân bố cycle time (hist khi ≥ 3 trial, scatter nếu ít hơn) → PNG."""
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
    """Bảng Markdown từ list dict theo cột [(key, tiêu đề)]."""
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |")
    return lines


CITE_LABEL_RE = re.compile(r"(?<!!)\[((?:[^\[\]]+ · tr\.\d+)|(?:KB|SOP)-\d+|station\.yaml)\](?!\()")
REF_HEADING = "## 8. Tài liệu tham chiếu"
AGG_NOTE_PREFIX = ("R1 bỏ qua", "R6 bỏ qua")
HAND_VI = {"left": "trái", "right": "phải"}
BADGE = {"passed": "✅ passed", "failed_after_retry": "❌ failed sau retry", "parse_failed": "❌ không parse được response",
         "failed": "❌ failed"}


class References:
    """Gom citation dùng trong report → bảng tham chiếu: đúng 1 dòng / nhãn ngắn (1 nhãn ↔ 1 dòng). Nhiều chunk cùng
    trang cùng nhãn → 1 dòng liệt kê mọi chunk_id, cột 'Dùng ở' ghi chunk nào dùng ở đâu."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def use(self, cit: dict | None, where: str) -> str:
        """Ghi nhận citation dùng ở `where`; trả nhãn ngắn để chèn vào văn bản ('' nếu không có / mặc định)."""
        if not cit or cit.get("kind") == "default":
            return ""
        label = cit.get("short") or f"[{cit.get('code') or cit.get('id')}]"
        row = self.rows.setdefault(label, {"label": label, "title": cit.get("doc_title") or "", "page": cit.get("page"),
                                           "chunks": {}})
        cid = cit.get("id") if cit.get("kind") == "chunk" else ""
        uses = row["chunks"].setdefault(cid, [])
        if where not in uses:
            uses.append(where)
        return label

    def rows_list(self) -> list[dict]:
        """Dòng bảng tham chiếu dạng dict (UI): label, title, page, chunk_ids, uses."""
        return [{"label": r_["label"], "title": r_["title"], "page": r_["page"],
                 "chunk_ids": [c for c in r_["chunks"] if c],
                 "uses": sorted({u for us in r_["chunks"].values() for u in us}, key=str)} for r_ in self.rows.values()]

    def table(self) -> list[str]:
        """Bảng Markdown 'Mã ngắn | Tài liệu | Trang | chunk_id | Dùng ở'."""
        if not self.rows:
            return ["Report không dùng trích dẫn tài liệu nào."]
        lines = ["| Mã ngắn | Tài liệu | Trang | chunk_id | Dùng ở |", "|---|---|---|---|---|"]
        for r_ in self.rows.values():
            ch = r_["chunks"]
            ids = "<br>".join(f"`{c}`" for c in ch if c) or "—"
            if len(ch) == 1:
                used = ", ".join(next(iter(ch.values())))
            else:
                used = "; ".join(f"{(c.rsplit(':', 1)[-1] if c else '—')}: {', '.join(u)}" for c, u in ch.items())
            lines.append(f"| {r_['label']} | {r_['title'] or '—'} | {r_['page'] or '—'} | {ids} | {used} |")
        return lines


def trace_check(md: str) -> tuple[set[str], set[str]]:
    """Kiểm truy vết 2 chiều: (nhãn trong thân report thiếu dòng tham chiếu, dòng tham chiếu không được dùng)."""
    body, _, refs = md.partition(REF_HEADING)
    refs = refs.split("\n## ", 1)[0]
    used = set(CITE_LABEL_RE.findall(body))
    rows = {m for ln in refs.splitlines() if ln.startswith("| [") for m in CITE_LABEL_RE.findall(ln.split("|")[1])}
    return used - rows, rows - used


def _words_cut(text: str, n: int) -> str:
    """Câu đầu, cắt ≤ n từ (thêm … nếu cắt)."""
    first = re.split(r"(?<=[.!?])\s+", str(text).strip())[0].rstrip(".")
    w = first.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


def problem_phrase(f: dict) -> str:
    """Vấn đề 5–8 từ cho dòng tóm tắt (số lấy nguyên từ evidence)."""
    e, t = f["evidence"], f["type"]
    hand = HAND_VI.get(e.get("hand") or e.get("idle_hand") or "", e.get("hand") or "")
    if t == "std_time_slow":
        return f"Bước {e['step_id']} chậm hơn chuẩn {e['deviation_pct']:g}%"
    if t == "std_time_fast":
        return f"Bước {e['step_id']} nhanh hơn chuẩn {abs(e['deviation_pct']):g}%"
    return {"slow_step": f"Bước {e.get('step_id')} chậm bất thường so với trial khác",
            "extra_pick": f"Lấy dư {e.get('extra')} lần ở {e.get('zone')}",
            "long_idle": f"Tay {hand} chờ {e.get('value')}s mỗi chu kỳ",
            "hand_imbalance": f"Tay {hand} rảnh khi tay kia thao tác",
            "high_variation": f"Cycle time biến động mạnh, CV {e.get('value')}",
            "wrist_posture": f"Cổ tay {hand} lệch góc {e.get('value')}% thời gian",
            "sequence_deviation": "Chuỗi bước lệch so với SOP"}.get(t, t)


def standards_view(standards: dict | None, cfg: dict) -> tuple[dict[int, dict], dict, dict]:
    """(std theo bước, ngưỡng lệch TG chuẩn, ngưỡng góc cổ tay) từ findings.json["standards"]; thiếu → station.yaml
    std_time_s + ngưỡng mặc định cấu hình."""
    stdd = standards or {}
    std = {int(k): v for k, v in (stdd.get("std_times") or {}).items()}
    if not any((v or {}).get("std_s") for v in std.values()):
        std = {int(s["id"]): {"std_s": float(s["std_time_s"]), "source": "station.yaml",
                              "citation": {"id": "station.yaml", "kind": "manual", "code": "station.yaml",
                                           "short": "[station.yaml]", "doc_title": "station.yaml (ghi tay)"}}
               for s in cfg.get("steps", []) if s.get("std_time_s") is not None}
    ths = stdd.get("thresholds") or {}
    thr_std = ths.get("std_dev_pct") or {"value": float(cfg.get("std_dev_threshold_pct", 30.0)), "source": "default"}
    thr_wrist = ths.get("wrist_angle_deg") or {"value": float(cfg.get("muri_dev_deg", 30)), "source": "default"}
    return std, thr_std, thr_wrist


def std_rows(m: dict, std: dict[int, dict], thr: float) -> list[dict]:
    """Bảng bước + TG chuẩn: {..dòng bước, std_s, dev_pct (lệch %), over (|lệch| > ngưỡng), citation}."""
    rows = []
    for row in sorted(m["steps"]["rows"], key=lambda x: x["step_id"]):
        s = std.get(int(row["step_id"])) or {}
        std_s = s.get("std_s")
        dev = round((row["dur_s"] - std_s) / std_s * 100, 1) if std_s else None
        rows.append({**row, "std_s": std_s, "dev_pct": dev, "over": dev is not None and abs(dev) > thr,
                     "citation": s.get("citation") if std_s else None})
    return rows


def step_std_rows(m: dict, std: dict[int, dict], thr: float, refs: References) -> list[dict]:
    """Dòng bảng bước + TG chuẩn / lệch % / nguồn cho Markdown (lệch vượt ngưỡng in đậm)."""
    out = []
    for row in std_rows(m, std, thr):
        dev = row["dev_pct"]
        dev_txt = "—" if dev is None else (f"**{dev:+.1f}**" if row["over"] else f"{dev:+.1f}")
        out.append({**row, "dev": dev_txt,
                    "src": (refs.use(row["citation"], "bảng bước") or "—") if row["std_s"] else "—"})
    return out


def finding_sources(f: dict, exp: dict | None, std: dict[int, dict], thr_std: dict, thr_wrist: dict) -> list[dict]:
    """Citation của 1 finding: nguồn giải thích LLM (07, citations_detail) + nguồn TG chuẩn/ngưỡng của rule (R8, R7)."""
    x = exp or {}
    cits = list(x.get("citations_detail") or [])
    if not cits:                                                       # explained.json cũ: chỉ có id KB-xx
        cits = [{"id": cid, "kind": "kb", "code": cid, "short": f"[{cid}]", "doc_title": "Kaizen KB / SOP (config/)"}
                for cid in (x.get("final_check") or {}).get("citations_normalized") or []
                if re.fullmatch(r"(KB|SOP)-\d+", cid)]
    e = f["evidence"]
    if f["rule"] == "R8":
        cits += [(std.get(int(e["step_id"])) or {}).get("citation"), thr_std.get("citation")]
    if f["rule"] == "R7":
        cits.append(thr_wrist.get("citation"))
    out, seen = [], set()
    for c in cits:
        if c and c.get("kind") != "default" and (c.get("id") or c.get("short")) not in seen:
            seen.add(c.get("id") or c.get("short"))
            out.append(c)
    return out


def write_report(out: Path, cfg: dict, metrics: dict, fdoc: dict | None, edoc: dict | None, steps_dir: Path,
                 events_dir: Path, features_dir: Path, data_root: str, inspect: str | Path = "outputs/inspect_report.md",
                 subjects: str = "", trials_sel: str = "", run_id: str | None = None,
                 manifest: dict | None = None, standards: dict | None = None) -> tuple[Path, int]:
    """Sinh report.md + report_assets/*.png từ artifact của 1 run; trả (thư mục assets, số trial trong report).

    standards: {std_times, thresholds} (mặc định findings.json["standards"]) — nguồn TG chuẩn + ngưỡng kèm citation.
    """
    man = manifest or {}
    assets = out.parent / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(out.parent).as_posix()  # noqa: E731

    trials = {st: m for st, m in metrics["trials"].items() if st in select_stems(list(metrics["trials"]), subjects, trials_sel)}
    fdoc = fdoc or {"findings": [], "notes": []}
    edoc = edoc or {"explanations": [], "mode": "none"}
    exps = {e["finding_id"]: e for e in edoc.get("explanations", [])}
    source = metrics["meta"]["source"]
    va = {int(s["id"]): bool(s.get("va", False)) for s in cfg.get("steps", [])}
    pxu = "cm" if cfg.get("px_per_cm") else "px"
    stdd = standards if standards is not None else (fdoc.get("standards") or {})
    std, thr_std, thr_wrist = standards_view(stdd, cfg)
    refs = References()
    fs = fdoc.get("findings", [])
    n_tr = len(trials)

    def thr_label(t: dict, where: str) -> str:
        lab = refs.use(t.get("citation"), where)
        return lab or "_(ngưỡng mặc định hệ thống)_"

    def finding_cites(f: dict) -> list[str]:
        """Nhãn nguồn của 1 finding (ghi nhận vào bảng tham chiếu)."""
        out_ = []
        for c in finding_sources(f, exps.get(f["id"]), std, thr_std, thr_wrist):
            lab = refs.use(c, f["id"])
            if lab and lab not in out_:
                out_.append(lab)
        return out_

    md = ["# Báo cáo Kaizen Motion Analyzer — HA4M", "",
          (f"- Run: `{run_id}` · pose `{man.get('pose_model')}` · LLM `{(man.get('llm') or {}).get('provider')}`"
           f" · git `{man.get('git_hash') or '—'}` · manifest: `../manifest.json`" if run_id else "- Run: (đường dẫn phẳng cũ)"),
          f"- Sinh lúc: {dt.datetime.now():%Y-%m-%d %H:%M} · nguồn bước: **{source}** "
          f"({'nhãn HA4M — ground truth' if source == 'oracle' else 'dự đoán từ keypoints'}) · "
          f"{n_tr} trial · subject: {', '.join(metrics['meta']['subjects'])}",
          "- Mọi con số do code tính (02→06). LLM chỉ diễn giải (07), validator kiểm số, trích dẫn & định dạng.", ""]

    # 1. Tóm tắt & đề xuất
    md += ["## 1. Tóm tắt & đề xuất", ""]
    top = sorted(fs, key=lambda f: f["severity"] == "info")[:3]
    for f in top:
        x = exps.get(f["id"]) or {}
        fin = x.get("final") or {}
        if x.get("validation") == "passed" and fin.get("recommendation"):
            ecrs = (fin.get("ecrs") or "").strip()
            prop = (f"{ecrs}: " if ecrs else "") + _words_cut(fin["recommendation"], 12)
        else:
            prop = "chưa có đề xuất hợp lệ — xem mục 7"
        cites = " ".join(finding_cites(f)[:2])
        md.append(f"- {f['id']} · {problem_phrase(f)} · {f['impact_s_per_cycle']:g}s/chu kỳ · {prop}" + (f" · {cites}" if cites else ""))
    if not fs:
        md.append("- Không có finding nào vượt ngưỡng.")
    per_trial: dict[str, float] = {}
    for f in fs:
        for st in f["evidence"].get("trials", [])[:1]:
            per_trial[st] = per_trial.get(st, 0.0) + max(f.get("impact_s_per_cycle") or 0.0, 0.0)
    pot = sum(per_trial.values()) / max(len(trials), 1)
    md += [f"- **Tổng tiềm năng: {pot:.1f} s/chu kỳ**" + (f" (trung bình {n_tr} trial)" if n_tr > 1 else "")
           + f" — cộng thô impact {len(fs)} finding, chưa khử trùng lặp.", ""]

    # 2. Tổng quan trial
    md += ["## 2. Tổng quan theo trial", ""]
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
    evals = {st: read_json(steps_dir / f"{st}_eval.json") for st in trials
             if (steps_dir / f"{st}_eval.json").exists()}
    if evals:
        md += ["### Độ chính xác phân đoạn bước (predicted vs nhãn HA4M)", ""]
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

    # 3. Bảng bước + TG chuẩn
    thr_v = float(thr_std["value"])
    md += ["## 3. Bảng bước & thời gian chuẩn", "",
           f"Lệch % = (thời lượng − TG chuẩn) / TG chuẩn. Ngưỡng lệch ±{thr_v:g}% "
           f"{thr_label(thr_std, 'ngưỡng lệch TG chuẩn (R8)')} — vượt ngưỡng in **đậm** (R8). Ô không có chuẩn: —.", ""]
    cols = [("step_id", "Bước"), ("name", "Tên"), ("va", "VA"), ("n_rep", "Lần"), ("dur_s", "Thời lượng (s)"),
            ("std_s", "TG chuẩn (s)"), ("dev", "Lệch %"), ("src", "Nguồn chuẩn"), ("pct_cycle", "% chu kỳ"),
            ("left_idle_pct", "Idle trái %"), ("right_idle_pct", "Idle phải %")]
    for i, (st, m) in enumerate(trials.items()):
        tbl = md_table(step_std_rows(m, std, thr_v, refs), cols)
        if i == 0:
            md += [f"**{st}**", ""] + tbl + [""]
        else:
            md += [f"<details><summary>{st}</summary>", ""] + tbl + ["", "</details>", ""]

    # 4. Timeline + Yamazumi
    md += ["## 4. Timeline & Yamazumi", ""]
    for st in trials:
        steps = {}
        for k in ("oracle", "pred"):
            p = steps_dir / f"{st}_{k}.csv"
            if p.exists():
                steps[k] = pd.read_csv(p)
        evp = events_dir / f"{st}.csv"
        ev = pd.read_csv(evp) if evp.exists() else None
        png = assets / f"gantt_{st}.png"
        gantt(st, steps, ev, va, png)
        md += [f"![Gantt {st}]({rel(png)})", ""]
        sp = features_dir / f"{st}_speed.png"
        if sp.exists():
            link = Path(os.path.relpath(sp.resolve(), out.parent.resolve())).as_posix()
            md += [f"<details><summary>Tốc độ cổ tay {st}</summary>", "", f"![speed {st}]({link})", "", "</details>", ""]
    yz = assets / "yamazumi.png"
    yamazumi(trials, va, cfg.get("cycle_time_target_s"), yz)
    md += [f"![Yamazumi]({rel(yz)})", ""]

    # 5. 3M tổng hợp — hoãn khi < 2 trial
    md += ["## 5. 3M tổng hợp (Mura giữa trial/subject)", ""]
    if n_tr < 2:
        md += [f"Đánh giá 3M tổng hợp: chờ đủ ≥2 video (hiện có {n_tr}).", ""]
    else:
        cd = assets / "cycle_time_dist.png"
        cycle_dist(trials, cd)
        mu = metrics["mura"]
        md += [f"![Cycle time]({rel(cd)})", ""]
        ct = mu["all_trials"]["cycle_time"]
        na = lambda v: "—" if v is None else v  # noqa: E731
        md += [f"- Cycle time toàn bộ: n={ct['n']}, mean={na(ct['mean'])}s, std={na(ct['std'])}, CV={na(ct['cv'])}"
               + (f" ({ct.get('note')})" if ct.get("note") else "")]
        for sub, v in mu["within_subject"].items():
            c = v["cycle_time"]
            md.append(f"- Trong subject {sub}: n={c['n']}, mean={na(c['mean'])}s, CV={na(c['cv'])}"
                      + (f" ({c.get('note')})" if c.get("note") else ""))
        b = mu["between_subjects"]["cycle_time"]
        md.append(f"- Giữa các subject: n={b['n']}, CV={na(b['cv'])}" + (f" ({b.get('note')})" if b.get("note") else ""))
        agg = [f["id"] + f" ({f['rule']})" for f in fs if f["rule"] in ("R1", "R6")]
        md.append(f"- Findings tổng hợp (R1 bước chậm bất thường, R6 biến động): {', '.join(agg) or 'không có'}")
        if mu.get("top_cv_steps"):
            md += ["", "Top bước CV cao nhất:", ""]
            md += md_table(mu["top_cv_steps"], [("step_id", "Bước"), ("cv", "CV")])
        md.append("")

    # 6. Muri per-video
    deg = thr_wrist["value"]
    md += ["## 6. Muri — tư thế cổ tay (ước lượng 2D, chỉ mang tính sàng lọc)", "",
           f"Ngưỡng góc lệch {deg:g}° {thr_label(thr_wrist, 'ngưỡng góc cổ tay (R7)')}"
           + (f" — {thr_wrist['note']}" if thr_wrist.get("note") and thr_wrist.get("source") == "document" else "")
           + f"; hệ thống đo với muri_dev_deg = {cfg.get('muri_dev_deg', 30)}°.", ""]
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
    md += ["## 7. Findings (rule-based) + diễn giải LLM", ""]
    if edoc.get("pass_rate") is not None:
        line = (f"**Validator LLM:** {edoc['n_passed']}/{edoc['n_findings']} explanation đạt "
                f"(pass rate {edoc['pass_rate'] * 100:.0f}%")
        if edoc.get("pass_rate_first_attempt") is not None:
            line += f"; đạt ngay lần 1: {edoc['pass_rate_first_attempt'] * 100:.0f}%, retry {edoc.get('n_retried', 0)}"
        line += f") — provider `{edoc.get('provider', edoc.get('mode'))}`"
        line += (f", model `{edoc.get('llm_model')}`" if edoc.get("llm_model") else "")
        line += (f", prompt `{edoc['prompt']}`" if edoc.get("prompt") else "") + "."
        md += [line, ""]
    if edoc.get("citation_note"):
        md += [f"> ⚠️ {edoc['citation_note']}", ""]
    notes = [n for n in fdoc.get("notes", []) if not (n_tr < 2 and n.startswith(AGG_NOTE_PREFIX))]
    md += [f"- _{n}_" for n in notes] + ([""] if notes else [])
    if not fs:
        md += ["Không có finding nào vượt ngưỡng.", ""]
    feats = {}
    for f in fs:
        e = f["evidence"]
        st = e["trials"][0]
        ts = e.get("timestamps") or []
        md += [f"### {f['id']} · {f['type']} · {f['rule']} · {f['severity']}", "",
               f"**Bằng chứng:** {e['metric']} = **{e['value']} {e['unit']}** · trial {', '.join(e['trials'])}"
               f" · ảnh hưởng {f['impact_s_per_cycle']} s/chu kỳ · confidence {f['confidence']}"
               + (f" · mốc {ts[0][0]}–{ts[0][1]}s" if ts else ""), ""]
        x = exps.get(f["id"])
        if x:
            fin, chk, status = x.get("final") or {}, x.get("final_check") or {}, x.get("validation")
            ecrs = fin.get("ecrs") or "—"
            md += [f"> **Quan sát:** {fin.get('observation') or '—'}  ",
                   f"> **Giả thuyết:** {fin.get('hypothesis') or '—'}  ",
                   f"> **Đề xuất:** {fin.get('recommendation') or '—'} · `ECRS: {ecrs}`  ",
                   f"> **Đo lại:** {fin.get('remeasure') or '—'}", ""]
            badge = BADGE.get(status, str(status))
            if status == "passed" and len(x.get("attempts", [])) > 1:
                badge += " (sau retry)"
            if status != "passed" and chk.get("message"):
                badge += f" — {chk['message']}"
        else:
            md += ["> _Chưa có diễn giải LLM (chạy 07)._", ""]
            badge = "—"
        md += [f"Nguồn: {' '.join(finding_cites(f)) or '—'} · Validator: {badge}", ""]
        if st not in feats:
            fp = features_dir / f"{st}.parquet"
            feats[st] = pd.read_parquet(fp) if fp.exists() else None
        if ts:
            t = (ts[0][0] + ts[0][1]) / 2
        else:
            w = trials.get(st, {}).get("window_s", [0, 0])
            t = (w[0] + w[1]) / 2
        png = assets / f"evidence_{f['id']}.png"
        if evidence_frame(data_root, st, t, cfg, feats[st], png, f["id"]):
            md += [f"![{f['id']}]({rel(png)})", ""]

    # 8. Tài liệu tham chiếu
    md += [REF_HEADING, "", "Mọi nhãn `[...]` trong report ↔ 1 dòng dưới đây (chunk_id đầy đủ để tra lại trong RAG store).", ""]
    md += refs.table() + [""]

    # 9. Giả định & giới hạn
    md += ["## 9. Giả định & giới hạn (tự sinh)", ""]
    lim = []
    doc_src = any((v or {}).get("source") == "document" for v in std.values()) or \
        any((t or {}).get("source") == "document" for t in (thr_std, thr_wrist))
    if doc_src:
        lim.append("Thời gian chuẩn & ngưỡng lấy từ tài liệu đã ingest, xem bảng tham chiếu (mục 8).")
    if not stdd.get("store"):
        lim.append("RAG store rỗng/không dùng: thời gian chuẩn chỉ từ station.yaml (nếu có), ngưỡng = mặc định hệ thống, "
                   "trích dẫn giải thích là KB-xx tĩnh — không truy vết tới trang tài liệu.")
    insp = Path(inspect).read_text(encoding="utf-8") if Path(inspect).exists() else ""
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
        kc = Path("outputs") / f"kinect_compare_{st}.json"
        if kc.exists():
            k = read_json(kc)
            lim.append(f"Đối chứng Kinect ({st}): lệch cổ tay so với skeleton Azure Kinect — phải: trung vị {k['right']['median_px']}px, "
                       f"P90 {k['right']['p90_px']}px, {k['right']['pct_within_50px']}% frame < 50px; trái: trung vị "
                       f"{k['left']['median_px']}px, P90 {k['left']['p90_px']}px, {k['left']['pct_within_50px']}% < 50px "
                       "→ số liệu tay trái kém tin cậy hơn (hay bị thân che).")
    no_lab = sorted(st for st, v in man.get("trials", {}).items() if v.get("labels_available") is False)
    if no_lab:
        lim.append(f"Không có nhãn (Labels*.txt) cho: {', '.join(no_lab)} — không có oracle/eval cho các trial này"
                   + ("; chúng KHÔNG có mặt trong report (nguồn bước = oracle)." if source == "oracle"
                      else "; số liệu của chúng chỉ từ bước dự đoán, không kiểm chứng được."))
    n_sub = len(metrics["meta"]["subjects"])
    if n_tr < 3:
        lim.append(f"Chỉ có {n_tr} trial / {n_sub} subject: R1 (bước chậm, median+2·MAD) cần ≥ 3 trial, R6 (CV cycle time) "
                   "cần ≥ 2 trial cùng subject — " + ("mục 5 hoãn" if n_tr < 2 else "Mura chưa có ý nghĩa thống kê")
                   + "; cần thêm video HA4M.")
    lim.append("%VA phụ thuộc phân loại va trong station.yaml (giả định, cần kỹ sư IE duyệt). Idle không đồng nghĩa lãng phí (KB-16).")
    if edoc.get("mode") == "dry-run":
        lim.append("Giải thích LLM đang ở chế độ **dry-run** (văn bản giả) — chạy `07_explain.py --provider gemini|anthropic|openai` "
                   "(cần GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY) để có giải thích thật.")
    md += [f"- {x}" for x in lim]
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return assets, len(trials)

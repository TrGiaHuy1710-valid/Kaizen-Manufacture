"""Hình plotly cho UI. Chỉ vẽ số đọc từ artifact; màu bước lấy từ overlay.step_color (khớp video + Yamazumi)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_access import overlay

INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
TRIAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EVENT_GROUP = {"pick": ("Lấy ở khay", "#eb6834"), "reach": ("Lấy ở khay", "#eb6834"), "search": ("Lấy ở khay", "#eb6834"),
               "assemble": ("Lắp/thao tác", "#2a78d6"), "place": ("Lắp/thao tác", "#2a78d6"),
               "transfer": ("Di chuyển", "#1baf7a"), "idle": ("Chờ", "#eda100"), "pause": ("Chờ", "#eda100"),
               "lost": ("Mất dấu", "#c3c2b7")}


LEGEND = {
    "bottom": dict(orientation="h", yanchor="top", y=-0.22, x=0),
    "top": dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    "right": dict(orientation="v", yanchor="top", y=1, x=1.02, traceorder="reversed"),
}


def _layout(fig: go.Figure, height: int, legend: str = "bottom", **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor=SURFACE,
                      legend=LEGEND[legend], font=dict(size=13), uniformtext=dict(minsize=10, mode="hide"), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def step_label(sid: int, rep: int = 0) -> str:
    return "?" if sid == 0 else f"{sid}" + (f".{rep}" if rep else "")


def gantt(steps: dict[str, pd.DataFrame], events: pd.DataFrame | None = None, names: dict[int, str] | None = None) -> go.Figure:
    """Mỗi nguồn bước (oracle/pred) 1 hàng; mỗi bước 1 trace (legend bật/tắt được); unclassified xám.
    events (tuỳ chọn) → thêm 2 hàng event cổ tay."""
    fig = go.Figure()
    rows = [f"Bước — {k}" for k in steps]
    seen = set()
    for (k, df), row in zip(steps.items(), rows):
        for sid, g in df.groupby("step_id"):
            sid = int(sid)
            nm = "Unclassified" if sid == 0 else f"B{sid}"   # tên đầy đủ ở hover + Yamazumi
            fig.add_trace(go.Bar(
                y=[row] * len(g), x=g["dur_s"], base=g["start_s"], orientation="h", name=nm,
                legendgroup=f"s{sid}", showlegend=sid not in seen,
                marker=dict(color=overlay.step_color(sid), line=dict(color="white", width=1.5)),
                text=[step_label(sid, int(r)) for r in g.get("repetition", [0] * len(g))],
                textposition="inside", insidetextanchor="middle", textfont=dict(color="white"),
                customdata=list(zip(g["name"], g["start_s"], g["end_s"], g["dur_s"])),
                hovertemplate="<b>%{customdata[0]}</b><br>start %{customdata[1]:.2f} s · end %{customdata[2]:.2f} s"
                              "<br>dur %{customdata[3]:.2f} s<extra>" + k + "</extra>"))
            seen.add(sid)
    if events is not None:
        for hand, lab in (("left", "Tay trái"), ("right", "Tay phải")):
            he = events[events["hand"] == hand]
            for et, g in he.groupby("event_type"):
                grp, col = EVENT_GROUP.get(et, (et, "#c3c2b7"))
                fig.add_trace(go.Bar(
                    y=[lab] * len(g), x=g["dur_s"], base=g["start_s"], orientation="h", name=f"Tay: {grp}",
                    legendgroup=f"e{grp}", showlegend=f"e{grp}" not in seen, marker=dict(color=col),
                    customdata=list(zip([et] * len(g), g["zone"], g["start_s"], g["dur_s"])),
                    hovertemplate="%{customdata[0]} · %{customdata[1]}<br>start %{customdata[2]:.2f} s · "
                                  "dur %{customdata[3]:.2f} s<extra>" + lab + "</extra>"))
                seen.add(f"e{grp}")
        rows += ["Tay trái", "Tay phải"]
    fig.update_layout(barmode="overlay", bargap=0.25)
    fig.update_yaxes(categoryorder="array", categoryarray=rows[::-1])
    fig.update_xaxes(title="thời gian (s)")
    fig.update_traces(textangle=0, selector=dict(type="bar"))
    _layout(fig, 220 + 70 * len(rows), legend="top")
    fig.update_layout(margin=dict(t=90))
    return fig


def yamazumi(trials: dict, target: float | None = None) -> go.Figure:
    """Bar chồng dur bước theo trial (từ metrics.json); unclassified xám; đường mục tiêu nếu có."""
    names = list(trials)
    sids = sorted({r["step_id"] for m in trials.values() for r in m["steps"]["rows"]})
    fig = go.Figure()
    for sid in sids:
        ys, nm = [], ""
        for st in names:
            row = next((r for r in trials[st]["steps"]["rows"] if r["step_id"] == sid), None)
            ys.append(row["dur_s"] if row else None)
            nm = nm or (row["name"] if row else "")
        fig.add_trace(go.Bar(x=names, y=ys, name=f"B{sid} · {nm}", marker=dict(color=overlay.step_color(sid),
                             line=dict(color="white", width=1.5)), text=[str(sid)] * len(names),
                             textposition="inside", textfont=dict(color="white"),
                             hovertemplate=f"B{sid} · {nm}<br>%{{x}}: %{{y:.2f}} s<extra></extra>"))
    unc = [((m.get("unclassified_s") or {}).get("value") or None) for m in trials.values()]
    if any(unc):
        fig.add_trace(go.Bar(x=names, y=unc, name="Unclassified", marker=dict(color=overlay.GRAY),
                             text=["?"] * len(names), textposition="inside",
                             hovertemplate="Unclassified<br>%{x}: %{y:.2f} s<extra></extra>"))
    ct = [m["cycle_time_s"]["value"] for m in trials.values()]
    fig.add_trace(go.Scatter(x=names, y=ct, mode="text", text=[f"{v:.1f} s" for v in ct], textposition="top center",
                             showlegend=False, hoverinfo="skip"))
    if target:
        fig.add_hline(y=target, line_dash="dash", line_color=INK2, annotation_text=f"mục tiêu {target:g} s")
    fig.update_layout(barmode="stack", bargap=0.45 if len(names) > 1 else 0.7)
    fig.update_traces(textangle=0, selector=dict(type="bar"))
    fig.update_yaxes(title="thời lượng bước cộng dồn (s)")
    return _layout(fig, 600, legend="right")


def step_compare(trials: dict, highlight: int | None = None) -> go.Figure:
    """Dur từng bước giữa các trial (bar nhóm); viền đậm bước có CV cao nhất (lấy từ metrics.mura)."""
    names = list(trials)
    sids = sorted({r["step_id"] for m in trials.values() for r in m["steps"]["rows"]})
    fig = go.Figure()
    for i, st in enumerate(names):
        rows = {r["step_id"]: r["dur_s"] for r in trials[st]["steps"]["rows"]}
        fig.add_trace(go.Bar(x=[f"B{s}" for s in sids], y=[rows.get(s) for s in sids], name=st,
                             marker_color=TRIAL_COLORS[i % len(TRIAL_COLORS)],
                             hovertemplate=st + " · %{x}: %{y:.2f} s<extra></extra>"))
    if highlight:
        fig.add_vrect(x0=sids.index(highlight) - 0.5, x1=sids.index(highlight) + 0.5, fillcolor="#e34948",
                      opacity=0.08, line_width=0, annotation_text="CV cao nhất", annotation_position="top")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="dur (s)")
    return _layout(fig, 380)


def cycle_box(values: dict[str, float], mean: float | None = None) -> go.Figure:
    """n ≥ 5: box plot; n < 5: chấm từng trial + đường trung bình (mean do 05_metrics tính)."""
    if len(values) >= 5:
        fig = go.Figure(go.Box(y=list(values.values()), text=list(values), boxpoints="all", jitter=0.3, pointpos=0,
                               marker_color="#2a78d6", name="cycle time",
                               hovertemplate="%{text}: %{y:.2f} s<extra></extra>"))
    else:
        fig = go.Figure(go.Scatter(x=list(values), y=list(values.values()), mode="markers+text",
                                   marker=dict(size=14, color="#2a78d6", line=dict(color="white", width=2)),
                                   text=[f"{v:.1f} s" for v in values.values()], textposition="middle right",
                                   hovertemplate="%{x}: %{y:.2f} s<extra></extra>"))
        if mean is not None:
            fig.add_hline(y=mean, line_dash="dot", line_color=INK2, annotation_text=f"TB {mean:g} s")
    fig.update_yaxes(title="cycle time (s)")
    return _layout(fig, 340, showlegend=False)


def compare_bars(summary: list[dict], cols: list[tuple[str, str]]) -> go.Figure:
    """Small multiples: mỗi chỉ số 1 ô riêng (không dùng 2 trục y)."""
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=[h for _, h in cols])
    runs = [s["run"] for s in summary]
    short = [r.split("_", 1)[1] if "_" in r else r for r in runs]
    for j, (k, _) in enumerate(cols, 1):
        fig.add_trace(go.Bar(x=short, y=[s.get(k) for s in summary], marker_color=TRIAL_COLORS[:len(runs)],
                             customdata=runs, hovertemplate="%{customdata}<br>%{y}<extra></extra>", showlegend=False),
                      row=1, col=j)
    fig.update_xaxes(tickangle=-30)
    return _layout(fig, 380)

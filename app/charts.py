"""Biểu đồ Plotly cho UI — chỉ VẼ số đã có (aggregate.json), không tính chỉ số."""
from __future__ import annotations

import plotly.graph_objects as go

INK2, GRID, REF = "#52514e", "#e4e3df", "#1baf7a"


def _layout(fig: go.Figure, h: int = 380, **kw) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13), **kw)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def cycle_bar(rows: list[dict], names: dict[str, str]) -> go.Figure:
    """Cycle time theo video; video tham chiếu (nhanh nhất) tô màu khác + chú thích."""
    xs = [names.get(r["project_id"], r["project_id"]) for r in rows]
    ys = [r["cycle_time_s"] for r in rows]
    cols = [REF if r["is_reference"] else "#2a78d6" for r in rows]
    txt = [f"{y:.1f}s" + (" · tham chiếu" if r["is_reference"] else f" (+{r['delta_vs_ref_s']:.1f}s)")
           for y, r in zip(ys, rows)]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=cols, text=txt, textposition="outside", cliponaxis=False))
    return _layout(fig, 340, yaxis_title="cycle time (s)", showlegend=False)


def yamazumi(stacks: dict[str, list[dict]], names: dict[str, str], color_of) -> go.Figure:
    """Yamazumi nhiều cột cạnh nhau: mỗi video 1 cột chồng thời lượng bước (màu theo bước)."""
    fig = go.Figure()
    sids = sorted({s["step_id"] for rows in stacks.values() for s in rows})
    xs = [names.get(pid, pid) for pid in stacks]
    for sid in sids:
        ys, hover = [], []
        for pid in stacks:
            row = next((s for s in stacks[pid] if s["step_id"] == sid), None)
            ys.append(row["dur_s"] if row else 0)
            hover.append(f"Bước {sid} · {row['name']}: {row['dur_s']:.1f}s" if row else "")
        name = next((s["name"] for rows in stacks.values() for s in rows if s["step_id"] == sid), "")
        fig.add_bar(x=xs, y=ys, name=f"{sid} · {name}"[:40], marker_color=color_of(sid),
                    text=[str(sid) if y >= 2.5 else "" for y in ys], textposition="inside",
                    hovertext=hover, hoverinfo="text")
    return _layout(fig, 460, barmode="stack", yaxis_title="thời lượng bước cộng dồn (s)",
                   legend=dict(orientation="v", font=dict(size=11)))

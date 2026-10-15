"""Trình phát 2 video overlay cạnh nhau, đồng bộ (HTML/JS chạy trong st.components.v1.html).

- Phát/Dừng chung, tốc độ chung, thanh tua chung (theo thời gian video A; B = A + độ lệch).
- "Nhảy tới bước": mỗi video tới ĐẦU bước đó theo mốc của chính trial ấy (steps của run) → độ lệch
  B−A được đặt lại để hai bên cùng bắt đầu bước; khoá đồng bộ giữ độ lệch khi phát (sửa trôi > 0.25 s).
- Dưới mỗi video: bước đang chạy (màu bước cố định như Gantt/Yamazumi/video) + thời gian trong bước.
Chỉ hiển thị mốc thời gian đọc từ artifact; phép trừ t − start chỉ để hiện đồng hồ như banner video.
"""
from __future__ import annotations

import html
import json

import pandas as pd

from data_access import overlay


def steps_payload(steps: pd.DataFrame | None) -> list[dict]:
    """Danh sách bước cho JS: key (vd '3.1'), nhãn, start/end (s), màu."""
    if steps is None or steps.empty:
        return []
    out = []
    for r in steps.sort_values("start_s").itertuples():
        sid, rep = int(r.step_id), int(getattr(r, "repetition", 0) or 0)
        key = "?" if sid == 0 else f"{sid}" + (f".{rep}" if rep else "")
        lab = "Unclassified" if sid == 0 else f"B{key} · {r.name}"
        out.append({"key": key, "label": lab, "start": float(r.start_s), "end": float(r.end_s),
                    "color": overlay.step_color(sid)})
    return out


def build_html(a: dict, b: dict, height_px: int = 430) -> str:
    """a/b: {title, subtitle, b64 (mp4 base64), steps: steps_payload(...)}."""
    def side(x: dict) -> dict:
        return {"title": x["title"], "subtitle": x.get("subtitle", ""), "steps": x.get("steps", [])}

    keys = []
    for s in a.get("steps", []) + b.get("steps", []):
        if s["key"] != "?" and s["key"] not in [k for k, _ in keys]:
            keys.append((s["key"], s["label"]))
    keys.sort(key=lambda kv: [float(p) for p in kv[0].split(".")])
    opts = "".join(f'<option value="{html.escape(k)}">{html.escape(lab)}</option>' for k, lab in keys)
    meta = json.dumps({"a": side(a), "b": side(b)}, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
:root {{ --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --bg:#fcfcfb; --acc:#2a78d6; }}
body {{ margin:0; font-family: "Source Sans Pro", system-ui, sans-serif; color:var(--ink); background:var(--bg); }}
.bar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:6px 2px 10px; font-size:14px; }}
.bar button {{ font-size:14px; padding:6px 14px; border-radius:8px; border:1px solid var(--line); background:#fff; cursor:pointer; }}
.bar button.primary {{ background:var(--acc); color:#fff; border-color:var(--acc); }}
.bar select, .bar input[type=number] {{ font-size:14px; padding:5px 6px; border-radius:6px; border:1px solid var(--line); }}
.bar label {{ color:var(--ink2); }}
#seek {{ width:100%; margin:4px 0 10px; accent-color:var(--acc); }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.panel h4 {{ margin:0 0 2px; font-size:15px; }} .panel .sub {{ color:var(--ink2); font-size:12px; margin-bottom:4px; }}
video {{ width:100%; max-height:{height_px}px; background:#111; border-radius:6px; display:block; }}
.step {{ display:flex; gap:8px; align-items:center; margin-top:6px; font-size:14px; min-height:22px; }}
.chip {{ width:12px; height:12px; border-radius:3px; flex:none; }}
.t {{ color:var(--ink2); font-size:12px; margin-left:auto; white-space:nowrap; }}
</style></head><body>
<div class="bar">
  <button id="play" class="primary">▶ Phát cả hai</button>
  <button id="back">⏪ 5 s</button><button id="fwd">5 s ⏩</button>
  <label>Tốc độ <select id="rate"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label>
  <label>Nhảy tới bước <select id="step"><option value="">— chọn —</option>{opts}</select></label>
  <label>Lệch B − A (s) <input id="off" type="number" step="0.1" value="0" style="width:70px"></label>
  <label><input id="lock" type="checkbox" checked> Khoá đồng bộ</label>
  <span id="msg" style="color:#a12b2b"></span>
</div>
<input id="seek" type="range" min="0" max="1" step="0.05" value="0">
<div class="grid">
  <div class="panel"><h4 id="ta"></h4><div class="sub" id="sa"></div><video id="va" muted playsinline preload="auto"></video>
    <div class="step"><span class="chip" id="ca"></span><span id="pa">—</span><span class="t" id="tta"></span></div></div>
  <div class="panel"><h4 id="tb"></h4><div class="sub" id="sb"></div><video id="vb" muted playsinline preload="auto"></video>
    <div class="step"><span class="chip" id="cb"></span><span id="pb">—</span><span class="t" id="ttb"></span></div></div>
</div>
<script>
const META = {meta};
const B64 = {{a: "{a['b64']}", b: "{b['b64']}"}};
function blobUrl(b64) {{
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([u], {{type: "video/mp4"}}));
}}
const $ = id => document.getElementById(id);
const va = $("va"), vb = $("vb"), seek = $("seek"), off = $("off"), lock = $("lock");
va.src = blobUrl(B64.a); vb.src = blobUrl(B64.b);
$("ta").textContent = META.a.title; $("sa").textContent = META.a.subtitle;
$("tb").textContent = META.b.title; $("sb").textContent = META.b.subtitle;
const clamp = (v, t) => Math.max(0, Math.min(v, (isFinite(t.duration) ? t.duration : 1e9) - 0.05));
function curStep(steps, t) {{ return steps.find(s => s.start <= t && t < s.end); }}
function show(v, steps, chip, lab, tt) {{
  const t = v.currentTime, s = curStep(steps, t);
  chip.style.background = s ? s.color : "#c3c2b7";
  lab.textContent = s ? s.label : (steps.length && t < steps[0].start ? "Chưa vào chu kỳ" : (steps.length ? "Ngoài chu kỳ" : "Không có dữ liệu bước"));
  tt.textContent = (s ? `bước ${{(t - s.start).toFixed(1)}} / ${{(s.end - s.start).toFixed(1)}} s · ` : "") + `t = ${{t.toFixed(1)}} s`;
}}
function syncB() {{ if (lock.checked) vb.currentTime = clamp(va.currentTime + parseFloat(off.value || 0), vb); }}
$("play").onclick = () => {{
  if (va.paused) {{ syncB(); va.play(); vb.play(); $("play").textContent = "⏸ Dừng"; }}
  else {{ va.pause(); vb.pause(); $("play").textContent = "▶ Phát cả hai"; }}
}};
$("back").onclick = () => {{ va.currentTime = clamp(va.currentTime - 5, va); syncB(); }};
$("fwd").onclick = () => {{ va.currentTime = clamp(va.currentTime + 5, va); syncB(); }};
$("rate").onchange = e => {{ va.playbackRate = vb.playbackRate = parseFloat(e.target.value); }};
off.onchange = syncB;
seek.oninput = () => {{ va.currentTime = clamp(parseFloat(seek.value), va); syncB(); }};
$("step").onchange = e => {{
  const k = e.target.value; if (!k) return;
  const sa = META.a.steps.find(s => s.key === k), sb = META.b.steps.find(s => s.key === k);
  if (sa) va.currentTime = clamp(sa.start, va);
  if (sb) vb.currentTime = clamp(sb.start, vb);
  if (sa && sb) {{ off.value = (sb.start - sa.start).toFixed(1); $("msg").textContent = ""; }}
  else {{
    lock.checked = false;
    $("msg").textContent = `${{sa ? "B" : "A"}} không có bước ${{k}} trong nguồn bước này (chưa gán được) — đã tắt khoá đồng bộ`;
  }}
}};
va.onloadedmetadata = () => {{ seek.max = va.duration.toFixed(2); }};
va.onended = () => {{ vb.pause(); $("play").textContent = "▶ Phát cả hai"; }};
setInterval(() => {{
  show(va, META.a.steps, $("ca"), $("pa"), $("tta")); show(vb, META.b.steps, $("cb"), $("pb"), $("ttb"));
  if (document.activeElement !== seek) seek.value = va.currentTime;
  if (lock.checked) $("msg").textContent = "";
  if (lock.checked && !va.paused && !vb.paused) {{
    const want = va.currentTime + parseFloat(off.value || 0);
    if (Math.abs(vb.currentTime - want) > 0.25) vb.currentTime = clamp(want, vb);
  }}
}}, 200);
</script></body></html>"""

"""Kaizen Motion Analyzer — UI demo theo "dự án video" (Streamlit). Chạy:  streamlit run app/ui.py
                                                            (demo:  KAIZEN_DEMO=1 streamlit run app/ui.py)

4 trang, mỗi trang MỘT hành động chính:
  📁 Dự án          thêm video (HA4M / upload / thư mục ảnh), danh sách, chọn ≥2 → tính 3M
  🎬 Chi tiết dự án  chạy full luồng (6 bước) · video overlay · Kaizen Brain · benchmark
  📚 Tài liệu (RAG)  ingest tài liệu + thử truy vấn
  📊 So sánh 3M      Mura / Muda / Muri / R1-R6 trên nhiều video + ghi chú người đánh giá
UI chỉ gọi app/data_access.py — không mở file, không tính chỉ số. Demo mode: tiến độ MÔ PHỎNG trên dự án đã tính
sẵn (badge cố định báo rõ), video theo nhãn chuẩn; video mới vẫn chạy THẬT (không bao giờ bịa số).
"""
from __future__ import annotations

import os
import random
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)                      # đường dẫn tương đối (outputs/, config/, data/) tính từ repo root
sys.path.insert(0, str(REPO / "app"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Kaizen Motion Analyzer", page_icon="🏭", layout="wide")

import charts  # noqa: E402
import data_access as da  # noqa: E402

P_HOME, P_PROJ, P_DOCS, P_3M = "📁 Dự án", "🎬 Chi tiết dự án", "📚 Tài liệu (RAG)", "📊 So sánh 3M"
PAGES = [P_HOME, P_PROJ, P_DOCS, P_3M]
DEMO_BADGE = "⚡ Demo — kết quả tính sẵn, overlay theo nhãn chuẩn"
STATE_CHIP = {"empty": ("⚪ Chưa chạy", "#e4e3df", "#3b3a37"), "running": ("🔵 Đang chạy", "#d6e6fb", "#1d4f91"),
              "done": ("🟢 Xong", "#d9f2e6", "#11643f"), "error": ("🔴 Lỗi", "#fbdada", "#8c1d1d")}
STEP_ICON = {"done": "✅", "running": "⏳", "pending": "⬜", "error": "❌", "skipped": "▫️"}
SIM_DUR = {"keypoints": None, "features": 1.5, "steps": 1.0, "metrics": 1.5, "explain": 2.5, "render": 2.0}
BASE_MODEL_NOTE = "Model hiện tại: base (MediaPipe pretrained, chưa fine-tune) — kết quả phân đoạn là baseline."

st.markdown("""<style>
[data-testid="stMetricValue"] {font-size: 1.8rem;}
.kz-demo-badge {position: fixed; top: 3.6rem; right: 1.2rem; z-index: 999990; background: #fff4d6; color: #6b4a00;
  border: 1px solid #e9c46a; border-radius: 8px; padding: 4px 12px; font-size: .85rem; font-weight: 600;
  box-shadow: 0 1px 4px rgba(0,0,0,.12);}
.kz-chip {display: inline-block; border-radius: 12px; padding: 2px 10px; font-size: .85rem; font-weight: 600;}
.kz-muted {opacity: .5;}
</style>""", unsafe_allow_html=True)


# ============================ tiện ích ============================
def go(page: str, **state) -> None:
    """Chuyển trang ở lần chạy kế tiếp (an toàn cả trong callback lẫn thân script)."""
    st.session_state["_nav_next"] = page
    st.session_state.update(state)


def fmt(v, unit: str = "", nd: int = 1) -> str:
    """Số → chuỗi gọn ('—' nếu thiếu); số nguyên giữ nguyên."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, (int, float)):
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
        return f"{s}{unit}"
    return f"{v}{unit}"


def chip(state: str) -> str:
    """HTML chip trạng thái dự án."""
    lab, bg, fg = STATE_CHIP.get(state, STATE_CHIP["empty"])
    return f'<span class="kz-chip" style="background:{bg};color:{fg}">{lab}</span>'


def safe(title: str):
    """Lỗi trong 1 trang → thông báo ngắn + chi tiết kỹ thuật trong expander (không trắng trang)."""
    def deco(fn):
        def wrap(*a, **k):
            try:
                fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ in ("RerunException", "StopException"):
                    raise
                st.error(f"Không hiển thị được {title}: {e}")
                with st.expander("Chi tiết kỹ thuật"):
                    st.code(traceback.format_exc())
        return wrap
    return deco


def demo_on() -> bool:
    """Demo mode đang bật (toggle sidebar)."""
    return bool(st.session_state.get("demo"))


# ============================ sidebar ============================
def sidebar() -> str:
    if not st.session_state.get("_qp_done"):            # deep link: ?page=home|project|docs|3m&id=<dự án>&agg=<id>
        st.session_state["_qp_done"] = True
        qp = st.query_params
        pages = {"home": P_HOME, "project": P_PROJ, "docs": P_DOCS, "3m": P_3M}
        if qp.get("page") in pages:
            st.session_state["_nav_next"] = pages[qp["page"]]
        if qp.get("id"):
            st.session_state["project"] = qp["id"]
        if qp.get("agg"):
            st.session_state["agg_id"] = qp["agg"]
    if "_nav_next" in st.session_state:
        st.session_state["nav"] = st.session_state.pop("_nav_next")
    st.session_state.setdefault("nav", P_HOME)
    st.session_state.setdefault("demo", da.demo_default())
    st.session_state.setdefault("revealed", [])
    st.sidebar.title("🏭 Kaizen Motion Analyzer")
    page = st.sidebar.radio("Trang", PAGES, key="nav", label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.toggle("Demo mode", key="demo",
                      help="Dự án đã tính sẵn: tiến độ mô phỏng + video theo nhãn chuẩn. Video mới vẫn chạy thật.")
    job = da.current_job()
    if job:
        st.sidebar.info(f"⚙️ Đang chạy pipeline: **{job['project_id']}**")
    st.sidebar.caption("Mọi con số do code tính; LLM chỉ diễn giải, có validator kiểm số & trích dẫn.")
    return page


# ============================ Trang 1 — Dự án ============================
def _selected_ids(projs: list[dict]) -> list[str]:
    stt = st.session_state.get("proj_table")
    rows = []
    try:
        rows = list(stt["selection"]["rows"]) if stt else []
    except (KeyError, TypeError):
        rows = []
    return [projs[i]["id"] for i in rows if i < len(projs)]


def _do_3m(ids: list[str]) -> None:
    try:
        agg = da.compute_3m(ids)
    except Exception as e:  # noqa: BLE001
        st.session_state["agg_error"] = f"Không tính được 3M: {e}"
        return
    if agg.ok:
        st.session_state.pop("agg_error", None)
        go(P_3M, agg_id=agg.agg_id)
    else:
        st.session_state["agg_error"] = agg.error + (
            "" if not agg.skipped else " · bỏ qua: " + "; ".join(f"{s['project_id']} ({s['reason']})" for s in agg.skipped))


def add_video_form() -> None:
    with st.container(border=True):
        st.markdown("**Thêm video mới** — 1 video = 1 dự án")
        kind = st.radio("Nguồn", ["Trial HA4M", "Upload video (.mp4)", "Thư mục ảnh"], horizontal=True, key="add_kind")
        if kind == "Upload video (.mp4)" and demo_on():
            st.info(BASE_MODEL_NOTE + " Video mới luôn được xử lý thật, không dùng kết quả tính sẵn.")
        with st.form("add_form", clear_on_submit=True, border=False):
            trial = up = folder = None
            fps, max_s = 30.0, 0.0
            if kind == "Trial HA4M":
                trials = da.ha4m_trials()
                if not trials:
                    st.warning("Không thấy trial HA4M nào trong data/ha4m.")
                trial = st.selectbox("Trial", trials) if trials else None
            elif kind == "Upload video (.mp4)":
                up = st.file_uploader("File video", type=["mp4", "mov", "avi", "mkv", "m4v", "webm"])
                max_s = st.number_input("Chỉ xử lý N giây đầu (0 = cả video)", 0.0, 3600.0, 90.0, 10.0,
                                        help="Video dài: nên giới hạn 60–90 s đầu để chạy nhanh trên CPU.")
            else:
                folder = st.text_input("Đường dẫn thư mục ảnh", placeholder="vd data/my_frames")
                fps = st.number_input("fps của chuỗi ảnh", 1.0, 240.0, 30.0)
            name = st.text_input("Tên dự án", placeholder="để trống = tự điền theo file / trial")
            ok = st.form_submit_button("Tạo dự án")
        if ok:
            try:
                if kind == "Trial HA4M":
                    if not trial:
                        st.warning("Chưa chọn trial.")
                        return
                    info = da.create_ha4m(trial, name)
                elif kind == "Upload video (.mp4)":
                    if up is None:
                        st.warning("Chưa chọn file video.")
                        return
                    info = da.create_upload(up.getvalue(), up.name, name, max_s or None)
                else:
                    if not folder:
                        st.warning("Chưa nhập thư mục ảnh.")
                        return
                    info = da.create_frames(folder, fps, name)
            except Exception as e:  # noqa: BLE001
                st.error(f"Không tạo được dự án: {e}")
                return
            st.session_state["show_add"] = False
            go(P_PROJ, project=info["id"])
            st.rerun()


@safe("danh sách dự án")
def page_home() -> None:
    st.title("📁 Dự án video")
    projs = da.list_projects()
    sel = _selected_ids(projs)
    multi = len(sel) >= 2
    c1, c2, _ = st.columns([1.2, 2.2, 3])
    c1.button("➕ Thêm video", type="secondary" if multi else "primary", width="stretch",
              on_click=lambda: st.session_state.update(show_add=not st.session_state.get("show_add")))
    if multi:
        c2.button(f"📊 Tính 3M các video đã chọn ({len(sel)})", type="primary", width="stretch",
                  on_click=_do_3m, args=(sel,))
    elif len(sel) == 1:
        c2.button("🎬 Mở dự án đã chọn", width="stretch", on_click=go, args=(P_PROJ,),
                  kwargs={"project": sel[0]})
    if st.session_state.get("agg_error"):
        st.warning(st.session_state["agg_error"])
    if st.session_state.get("show_add"):
        add_video_form()
    old = da.unadopted_runs()
    if not projs:
        st.info("Chưa có dự án — bấm **➕ Thêm video** để bắt đầu.")
        if old:
            adopt_box(old, expanded=True)
        return
    st.caption("Chọn 1 dòng để mở dự án, chọn ≥ 2 dòng để tính 3M tổng hợp.")
    rows = []
    for p in projs:
        rows.append({"Ảnh": p.get("thumb_uri"), "Tên": p.get("name") or p["id"], "Mã": p["id"],
                     "Trạng thái": STATE_CHIP.get(p.get("status"), STATE_CHIP["empty"])[0],
                     "Nguồn bước": da.source_badge(p.get("source")) if p.get("source") else "—",
                     "Cycle time (s)": p.get("cycle_time_s"),
                     "F1@IoU0.5": p.get("f1_iou05") if p.get("labels_available") else None,
                     "Ngày chạy": (p.get("run_at") or "").replace("T", " ")[:16] or "—"})
    df = pd.DataFrame(rows)

    def color(v):
        for lab, bg, fg in STATE_CHIP.values():
            if v == lab:
                return f"background-color:{bg};color:{fg};font-weight:600"
        return ""
    sty = df.style.applymap(color, subset=["Trạng thái"]).format({"Cycle time (s)": lambda v: fmt(v),
                                                               "F1@IoU0.5": lambda v: fmt(v, nd=3)}, na_rep="—")
    st.dataframe(sty, key="proj_table", on_select="rerun", selection_mode="multi-row", hide_index=True,
                 width="stretch", row_height=64,
                 column_config={"Ảnh": st.column_config.ImageColumn("Ảnh", width="small")})
    if old:
        adopt_box(old)


def adopt_box(old: list[str], expanded: bool = False) -> None:
    with st.expander(f"📦 {len(old)} run cũ chưa thuộc dự án nào", expanded=expanded):
        rid = st.selectbox("Run cũ (outputs/runs/)", old, key="adopt_sel")
        if st.button("Bọc run này thành dự án"):
            try:
                made = da.adopt(rid)
                st.success(f"Đã tạo dự án: {', '.join(made)}" if made else "Run này đã được bọc trước đó.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Không adopt được: {e}")


# ============================ Trang 2 — Chi tiết dự án ============================
def simulate(pid: str) -> None:
    """Demo mode: tiến độ MÔ PHỎNG 6 bước (không gọi pipeline) rồi mở kết quả tính sẵn."""
    k = da.demo_speed()
    with st.status("Đang xử lý video…", expanded=True) as box:
        for key, label in da.STEPS:
            ph = st.empty()
            box.update(label=f"{label}…")
            if key == "keypoints":
                ph.markdown(f"⏳ **{label}** — {da.POSE_MODEL_LABEL}")
                bar = st.progress(0, text="0%")
                d = random.uniform(4.0, 6.0) * k
                for i in range(1, 21):
                    time.sleep(d / 20)
                    bar.progress(i * 5, text=f"{i * 5}%")
            else:
                ph.markdown(f"⏳ **{label}**")
                time.sleep(SIM_DUR[key] * k)
            ph.markdown(f"✅ **{label}**" + (f" — {da.POSE_MODEL_LABEL}" if key == "keypoints" else ""))
        box.update(label="Xong — hiển thị kết quả tính sẵn", state="complete", expanded=False)


def _status_box(s: dict) -> None:
    steps = s.get("steps") or [{"key": k, "label": lab, "state": "pending", "detail": ""} for k, lab in da.STEPS]
    run = s.get("state_now") == "running"
    lab = "Đang chạy full luồng…" if run else ("Lỗi khi chạy" if s.get("state_now") == "error" else "Lần chạy gần nhất")
    with st.status(lab, state="running" if run else ("error" if s.get("state_now") == "error" else "complete"),
                   expanded=run):
        for x in steps:
            det = x.get("detail") or (da.POSE_MODEL_LABEL if x["key"] == "keypoints" else "")
            st.markdown(f"{STEP_ICON.get(x['state'], '⬜')} **{x['label']}**" + (f" — {det}" if det else ""))
        for n in s.get("notes") or []:
            st.caption(f"ℹ️ {n}")


@st.fragment(run_every="2s")
def live_progress(pid: str) -> None:
    s = da.project_status(pid)
    _status_box(s)
    if s.get("state_now") != "running":
        st.rerun(scope="app")


def block_run(p: dict) -> None:
    pid, demo = p["id"], demo_on()
    s = da.project_status(pid)
    state = s.get("state_now", "empty")
    job = da.current_job()
    sim = st.session_state.get("sim_running")
    st.markdown(chip(state) + (f" &nbsp; run `{p.get('run_id')}`" if p.get("run_id") else ""), unsafe_allow_html=True)
    ready, missing = da.demo_ready(pid) if demo else (False, [])
    if state == "running":
        live_progress(pid)
        return
    busy = bool(job) or bool(sim)
    if busy and (not job or job.get("project_id") != pid):
        st.info(f"⏸ Đang chạy dự án **{job['project_id'] if job else '(mô phỏng)'}** — nút chạy tạm khoá, chờ xong.")
    if demo and ready:
        if st.button("▶ Chạy full luồng", type="primary", disabled=busy, key="run_btn"):
            st.session_state["sim_running"] = True
            try:
                simulate(pid)
            finally:
                st.session_state["sim_running"] = False
            if pid not in st.session_state["revealed"]:
                st.session_state["revealed"].append(pid)
            st.rerun()
        return
    if demo and p.get("labels_available") and not ready:
        st.warning("Dự án này chưa tính sẵn — chạy thật?")
        with st.expander("Còn thiếu"):
            st.write(", ".join(missing))
        label = "▶ Chạy thật"
    else:
        label = "▶ Chạy full luồng"
        if demo:
            st.caption(BASE_MODEL_NOTE)
    if st.button(label, type="primary", disabled=busy, key="run_btn"):
        ok, msg = da.start_job(pid)
        (st.toast if ok else st.warning)(msg)
        time.sleep(0.3)
        st.rerun()
    if state == "error":
        st.error(s.get("message") or "Lần chạy trước bị lỗi.")
        with st.expander("Log lỗi"):
            st.code(da.log_tail(pid) or "(không có log)")
    elif state != "empty" and s.get("steps"):
        _status_box(s)


def block_video(p: dict) -> None:
    st.subheader("🎞 Video có nhãn vẽ lại")
    pid = p["id"]
    path, meta = da.demo_video(pid)
    if path is None:
        st.info("Chưa render video overlay.")
        if st.button("🎞 Render video", disabled=bool(da.current_job()), key="render_btn"):
            ok, msg = da.start_job(pid, only="render")
            (st.toast if ok else st.warning)(msg)
            st.rerun()
        return
    st.video(str(path))
    src = (meta or {}).get("source")
    st.caption(f"Nguồn nhãn trên video: **{da.source_label(src)}** · {da.source_badge(src) if src else ''}")
    if demo_on() and p.get("labels_available") and src != "oracle":
        st.warning("Demo mode cần bản overlay theo nhãn chuẩn — bản hiện tại dùng nhãn dự đoán.")
        if st.button("🎞 Render lại theo nhãn chuẩn", disabled=bool(da.current_job()), key="rerender_btn"):
            ok, msg = da.start_job(pid, only="render")
            (st.toast if ok else st.warning)(msg)
            st.rerun()


def finding_card(c: dict, explained: bool = True) -> None:
    with st.container(border=True):
        sev = {"high": "🔴", "med": "🟠", "low": "🟡", "info": "🔵"}.get(c["severity"], "⚪")
        st.markdown(f"{sev} **{c['id']} · {c['problem']}** · {c['rule']} · {c['severity']}")
        st.caption(f"Bằng chứng: {c.get('metric')} = {fmt(c.get('value'), ' ' + (c.get('unit') or ''))}"
                   f" · ảnh hưởng {fmt(c.get('impact'), ' s/chu kỳ')}"
                   + (f" · video {', '.join(c['trials'])}" if c.get("trials") else ""))
        if not explained:
            return
        if not c.get("observation"):
            st.caption("_Chưa có diễn giải LLM._")
        else:
            st.markdown(f"**Quan sát:** {c['observation']}  \n**Giả thuyết:** {c.get('hypothesis') or '—'}  \n"
                        f"**Đề xuất:** {c.get('recommendation') or '—'} · `ECRS: {c.get('ecrs') or '—'}`  \n"
                        f"**Đo lại:** {c.get('remeasure') or '—'}")
        badge = {"passed": "✅ validator passed", "failed_after_retry": "❌ validator fail sau retry",
                 "parse_failed": "❌ không parse được"}.get(c.get("validation") or "", "")
        st.caption(f"Nguồn: {' '.join(c.get('sources') or []) or '—'}" + (f" · {badge}" if badge else ""))


def block_brain(v: dict) -> None:
    st.subheader("🧠 Kaizen Brain")
    st.caption(f"Nguồn bước: {da.source_badge(v.get('source'))} · run `{v['run_id']}`")
    if v["steps"]:
        thr = v["threshold"]
        st.markdown(f"**Bảng bước** — ngưỡng lệch ±{fmt(thr['value'])}% {thr['label']} (vượt ngưỡng tô đỏ)")
        df = pd.DataFrame(v["steps"])
        over = df.pop("Vượt ngưỡng")
        sty = df.style.apply(lambda r: ["color:#b42318;font-weight:700" if over[r.name] and c == "Lệch %" else ""
                                        for c in df.columns], axis=1).format(
            {"Thời lượng (s)": lambda x: fmt(x, nd=2), "TG chuẩn (s)": lambda x: fmt(x),
             "Lệch %": lambda x: "—" if x is None or pd.isna(x) else f"{x:+.1f}"})
        st.dataframe(sty, hide_index=True, width="stretch")
    else:
        st.info("Run chưa có bảng bước cho video này.")
    ex = v.get("explain") or {}
    if ex.get("citation_note"):
        st.caption(f"⚠️ {ex['citation_note']}")
    if ex.get("mode") == "dry-run":
        st.caption("⚠️ Giải thích đang ở chế độ dry-run (văn bản giả — thiếu API key).")
    st.markdown(f"**Findings** ({len(v['findings'])})")
    if not v["findings"]:
        st.caption("Không có finding nào vượt ngưỡng.")
    for c in v["findings"]:
        finding_card(c)
    if v.get("notes"):
        with st.expander("Ghi chú rule"):
            for n in v["notes"]:
                st.caption(n)
    with st.expander("📄 Tài liệu đã dùng"):
        if v["references"]:
            st.dataframe(pd.DataFrame([{"Mã ngắn": r["label"], "Tài liệu": r["title"], "Trang": r["page"] or "—",
                                        "chunk_id": ", ".join(r["chunk_ids"]) or "—", "Dùng ở": ", ".join(r["uses"])}
                                       for r in v["references"]]), hide_index=True, width="stretch")
        else:
            st.caption("Chưa dùng trích dẫn tài liệu nào.")


def block_benchmark(pid: str) -> None:
    b = da.benchmark(pid)
    with st.expander("📏 Benchmark", expanded=False):
        st.markdown(b["model_line"])
        if not b.get("labels_available"):
            st.info("Không có nhãn chuẩn — chỉ hiển thị chỉ số vận hành.")
        if b["rows"]:
            st.dataframe(pd.DataFrame([{"Chỉ số": r["label"], "Giá trị": fmt(r["value"], " " + r["unit"] if r["unit"] else "", 3),
                                        "Ghi chú": r["note"] or ""} for r in b["rows"]]),
                         hide_index=True, width="stretch")
        else:
            st.caption("Chưa có run nào.")
        st.caption(b["note"])


@safe("chi tiết dự án")
def page_project() -> None:
    projs = da.list_projects()
    if not projs:
        st.title("🎬 Chi tiết dự án")
        st.info("Chưa có dự án nào — vào trang 📁 Dự án để thêm video.")
        return
    ids = [p["id"] for p in projs]
    names = {p["id"]: p.get("name") or p["id"] for p in projs}
    if st.session_state.get("project") not in ids:
        st.session_state["project"] = ids[0]
    pid = st.session_state["project"]
    p = da.get_project(pid)
    if p is None:
        st.warning(f"Không đọc được dự án {pid}.")
        return
    top = st.columns([4, 2])
    top[0].title(f"🎬 Dự án: {p.get('name') or pid}")
    top[1].selectbox("Đổi dự án", ids, index=ids.index(pid), format_func=lambda x: names[x], key="proj_pick",
                     on_change=lambda: st.session_state.update(project=st.session_state["proj_pick"]))
    kind = {"ha4m": "trial HA4M", "video": "video upload", "frames": "thư mục ảnh", "adopted": "run cũ"}.get(
        p.get("kind"), p.get("kind"))
    st.caption(f"{kind} · `{p['id']}` · {'có nhãn chuẩn' if p.get('labels_available') else 'không có nhãn chuẩn'}"
               + (f" · chỉ xử lý {fmt(p['max_seconds'])} s đầu" if p.get("max_seconds") else ""))
    block_run(p)
    if p.get("status") == "running":
        return
    if demo_on() and da.demo_ready(pid)[0] and pid not in st.session_state.get("revealed", []):
        st.caption("Bấm **▶ Chạy full luồng** để xem kết quả.")
        return
    st.divider()
    block_video(p)
    st.divider()
    v = da.project_view(pid)
    if v is None:
        st.info("Chưa có kết quả — bấm **▶ Chạy full luồng**.")
        return
    block_brain(v)
    block_benchmark(pid)
    rp = da.report_path(pid)
    if rp:
        st.download_button("⬇ Tải report.md", rp.read_bytes(), file_name=f"report_{pid}.md", mime="text/markdown")


# ============================ Trang 3 — Tài liệu ============================
@safe("trang tài liệu")
def page_docs() -> None:
    st.title("📚 Tài liệu (RAG)")
    st.caption(f"Store: `{da.rag_store_path()}` — work instruction / guideline dùng để trích dẫn TG chuẩn, ngưỡng, khuyến nghị.")
    up = st.file_uploader("Tải tài liệu (PDF / DOCX / MD / TXT)", type=["pdf", "docx", "md", "txt"])
    if st.button("📥 Ingest tài liệu", type="primary", disabled=up is None):
        with st.spinner("Đang đọc, chia chunk và embed…"):
            try:
                st.session_state["last_ingest"] = da.rag_ingest(up.getvalue(), up.name)
            except Exception as e:  # noqa: BLE001
                st.session_state["last_ingest"] = {"error": str(e)}
    r = st.session_state.get("last_ingest")
    if r:
        if r.get("error"):
            st.error(f"Không ingest được: {r['error']}")
        else:
            st.success(f"{'Đã có sẵn' if r.get('already_present') else 'Đã ingest'} **{r['title']}** (`{r['doc_id']}`): "
                       f"{r['pages']} trang, {r['chunks']} chunk"
                       + (f" · trang cần OCR: {r['needs_ocr_pages']}" if r.get("needs_ocr_pages") else ""))
    docs = da.rag_documents()
    st.subheader("Tài liệu đã ingest")
    if not docs:
        st.info("Store rỗng — report sẽ trích dẫn KB-xx tĩnh cho tới khi có tài liệu.")
    else:
        st.dataframe(pd.DataFrame([{"Tên": d.get("title"), "File": d.get("filename"), "Trang": d.get("pages"),
                                    "Số chunk": d.get("chunks"), "Ngày": (d.get("ingested_at") or "").replace("T", " "),
                                    "doc_id": d.get("doc_id")} for d in docs]), hide_index=True, width="stretch")
    st.subheader("Thử truy vấn")
    q = st.text_input("Câu hỏi", placeholder="vd: tư thế cổ tay khi ấn vòng bi")
    if st.button("🔎 Tìm", disabled=not q or not docs):
        try:
            st.session_state["last_query"] = (q, da.rag_query(q, 4))
        except Exception as e:  # noqa: BLE001
            st.error(f"Truy vấn lỗi: {e}")
    lq = st.session_state.get("last_query")
    if lq:
        st.caption(f"Top-4 cho “{lq[0]}”")
        for i, h in enumerate(lq[1], 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {h['label']}** · {h['title']} · trang {h['page']} · score {h['score']:.3f}")
                st.caption(f"{h['section']}" if h.get("section") else "")
                st.markdown(f"> {h['excerpt']}")
                with st.expander(f"Toàn văn `{h['chunk_id']}`"):
                    st.text(h["text"])


# ============================ Trang 4 — So sánh 3M ============================
@safe("so sánh 3M")
def page_3m() -> None:
    st.title("📊 So sánh 3M nhiều video")
    aggs = da.list_aggregates()
    ids = [a["agg_id"] for a in aggs]
    cur = st.session_state.get("agg_id")
    if aggs:
        lab = {a["agg_id"]: f"{a['agg_id']} · {a['n_videos']} video · {', '.join(a['projects'])}"
               + (" · 📝" if a["has_review"] else "") for a in aggs}
        idx = ids.index(cur) if cur in ids else 0
        st.selectbox("Phép tính đã lưu", ids, index=idx, format_func=lambda x: lab[x], key="agg_pick",
                     on_change=lambda: st.session_state.update(agg_id=st.session_state["agg_pick"]))
        cur = cur if cur in ids else ids[0]
    if not cur:
        st.info("Chưa có phép tính nào — vào 📁 Dự án, chọn ≥ 2 video đã chạy xong rồi bấm **📊 Tính 3M**.")
        return
    agg = da.load_aggregate(cur)
    if agg is None:
        st.warning(f"Không mở được aggregate {cur}.")
        return
    names = {x["project_id"]: x["name"] for x in agg.projects}
    st.markdown("**Video trong phép tính**")
    st.dataframe(pd.DataFrame([{"Video": x["name"], "Mã": x["project_id"], "Nguồn bước": da.source_badge(x["source"]),
                                "Run": x["run_id"]} for x in agg.projects]), hide_index=True, width="stretch")
    for s in agg.skipped:
        st.caption(f"⏭ Bỏ qua {s['project_id']}: {s['reason']}")

    st.subheader("Mura — biến động giữa các video")
    ct = agg.mura["cycle_time"]
    c = st.columns(4)
    c[0].metric("CV cycle time", fmt(ct.get("cv"), nd=3))
    c[1].metric("Cycle time trung bình", fmt(ct.get("mean"), " s"))
    c[2].metric("Độ lệch chuẩn", fmt(ct.get("std"), " s"))
    c[3].metric("Số video", ct.get("n"))
    st.caption(agg.mura["definition"])
    st.plotly_chart(charts.cycle_bar(agg.muda["rows"], names), width="stretch")
    sdf = pd.DataFrame([{"Bước": s["step_id"], "Tên": s["name"], "n": s["n"], "Mean (s)": s.get("mean"),
                         "Std (s)": s.get("std"), "CV": s.get("cv"), "_max": s.get("is_max")} for s in agg.mura["steps_cv"]])
    if not sdf.empty:
        mx = sdf.pop("_max")
        st.markdown("**CV theo bước** (cao nhất tô đỏ)")
        st.dataframe(sdf.style.apply(lambda r: ["background-color:#fbdada;color:#8c1d1d;font-weight:700" if mx[r.name]
                                                else "" for _ in r], axis=1).format(
            {"Mean (s)": lambda x: fmt(x, nd=2), "Std (s)": lambda x: fmt(x, nd=2), "CV": lambda x: fmt(x, nd=3)}),
            hide_index=True, width="stretch")

    st.subheader("Muda — so sánh thao tác")
    st.plotly_chart(charts.yamazumi(agg.yamazumi, names, da.color_of_step), width="stretch")
    mu = agg.muda
    cols = list(mu["labels"])
    st.dataframe(pd.DataFrame([{"Video": names.get(r["project_id"], r["project_id"]) + (" ⭐ tham chiếu" if r["is_reference"] else ""),
                                **{f"{mu['labels'][k]} ({mu['units'].get(k, '')})": fmt(r[k], nd=1) for k in cols},
                                "Δ cycle vs tham chiếu (s)": fmt(r["delta_vs_ref_s"], nd=1)} for r in mu["rows"]]),
                 hide_index=True, width="stretch")
    st.caption("Tham chiếu = video có cycle time nhỏ nhất.")

    st.subheader("Muri — tư thế cổ tay (gộp)")
    mrows = agg.muri["rows"]
    mdf = pd.DataFrame([{"Video": names.get(r["project_id"], r["project_id"]), "Tay": "trái" if r["hand"] == "left" else "phải",
                         "% frame đo được": r["coverage_pct"], "Góc trung vị (°)": r["median_angle_deg"],
                         "% thời gian lệch": r["dev_gt_thr_pct"], "P90 lệch (°)": r["dev_p90_deg"]} for r in mrows])
    low = [r["low_coverage"] for r in mrows]
    if not mdf.empty:
        st.dataframe(mdf.style.apply(lambda r: ["opacity:.45;color:#8d8c86" if low[r.name] else "" for _ in r], axis=1)
                     .format(lambda x: fmt(x)), hide_index=True, width="stretch")
    st.caption(f"{agg.muri['note']} · dòng mờ: độ phủ < {fmt(agg.muri['coverage_low_pct'])}% (kém tin cậy).")

    st.subheader("Findings tổng hợp (R1 / R6)")
    for r in agg.rules:
        if not r["ran"]:
            st.markdown(f'<div class="kz-muted">⏸ <b>{r["rule"]}</b> — bỏ qua: {r["reason"]}<br>'
                        f'<small>{r["description"]}</small></div>', unsafe_allow_html=True)
        elif not r["fired"]:
            st.markdown(f"✓ **{r['rule']}** đã chạy — không video nào vượt ngưỡng.")
    for c in da.aggregate_cards(agg):
        finding_card(c, explained=False)
    for n in (agg.findings or {}).get("notes", []):
        if not n.startswith(("R1 bỏ qua", "R6 bỏ qua")):
            st.caption(f"ℹ️ {n}")

    st.subheader("📝 Đánh giá của người xem")
    note = st.text_area("Ghi chú", value=(agg.review or {}).get("note") or "", key=f"review_{cur}",
                        placeholder="Nhận định, quyết định cải tiến, điểm cần kiểm chứng…")
    if st.button("💾 Lưu ghi chú đánh giá", type="primary"):
        da.save_review(cur, note)
        st.success("Đã lưu vào aggregate.json.")
    if (agg.review or {}).get("updated_at"):
        st.caption(f"Lưu lần cuối: {agg.review['updated_at'].replace('T', ' ')}")


# ============================ main ============================
page = sidebar()
if demo_on():
    st.markdown(f'<div class="kz-demo-badge">{DEMO_BADGE}</div>', unsafe_allow_html=True)
{P_HOME: page_home, P_PROJ: page_project, P_DOCS: page_docs, P_3M: page_3m}[page]()

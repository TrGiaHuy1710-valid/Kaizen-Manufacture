"""Kaizen Motion Analyzer — dashboard demo (Streamlit). Chạy:  streamlit run app/ui.py

UI chỉ ĐỌC artifact của run (outputs/runs/<run_id>/): không tính lại chỉ số nào. Số do pipeline tính,
LLM giải thích, UI hiển thị. Artifact thiếu → cảnh báo nhẹ, không crash.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)                      # run_io/compare_runs dùng đường dẫn tương đối outputs/runs
sys.path.insert(0, str(REPO / "app"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Kaizen Motion Analyzer", page_icon="🏭", layout="wide")

import charts  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
import video_compare as vc  # noqa: E402
import data_access as da  # noqa: E402
from ui_utils import fmt, unit, val  # noqa: E402

st.markdown("""<style>
[data-testid="stMetricValue"] {font-size: 2.1rem;}
[data-testid="stMetricLabel"] p {font-size: 1rem;}
.muted {color: #8d8c86;}
</style>""", unsafe_allow_html=True)

NA = "chưa có trong run này"
SOURCE_BADGE = {"oracle": ("oracle (nhãn HA4M)", "green", "🟢"), "pred": ("pred (CV dự đoán)", "orange", "🟠")}
MURI_COVERAGE_MIN = 50.0


def safe(title: str):
    """Decorator: lỗi trong 1 tab → hiện lỗi gọn, các tab khác vẫn chạy."""
    def deco(fn):
        def wrap(*a, **k):
            try:
                fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                st.error(f"Lỗi khi hiển thị {title}: {e}")
                with st.expander("Chi tiết lỗi"):
                    st.exception(e)
        return wrap
    return deco


# ============================ Sidebar ============================
def sidebar() -> tuple[str | None, str | None]:
    st.sidebar.title("🏭 Kaizen Motion Analyzer")
    runs = da.list_runs()
    if not runs:
        st.sidebar.warning("Chưa có run nào trong outputs/runs/. Dùng tab **Chạy pipeline** hoặc `bash run_all.sh`.")
        return None, None
    ids = [m["run_id"] for m in runs]
    labels = {m["run_id"]: da.run_label(m) for m in runs}
    if "pending_run_sel" in st.session_state:          # run mới từ tab Chạy pipeline
        st.session_state["run_sel"] = st.session_state.pop("pending_run_sel")
    qp = st.query_params                                # deep link: ?run=<run_id>&trial=<trial>
    if not st.session_state.get("_qp_done"):
        st.session_state["_qp_done"] = True
        if qp.get("run") in ids:
            st.session_state["run_sel"] = qp.get("run")
        if qp.get("trial"):
            st.session_state["trial_sel"] = qp.get("trial")
    if st.session_state.get("run_sel") not in ids:
        st.session_state["run_sel"] = da.latest_run() if da.latest_run() in ids else ids[0]
    run_id = st.sidebar.selectbox("Run", ids, key="run_sel", format_func=lambda r: labels[r],
                                  help="Mới nhất ở trên. Thời điểm · nguồn bước · pose · LLM · tag")
    st.sidebar.caption(f"`{run_id}`" + ("  ·  LATEST" if run_id == da.latest_run() else "")
                       + f"  ·  [link](?run={run_id})")

    src = da.run_source(run_id)
    lab, color, icon = SOURCE_BADGE.get(src, (f"{src or 'không rõ'}", "gray", "⚪"))
    st.sidebar.badge(f"Nguồn bước: {lab}", icon=icon, color=color)
    if src == "pred":
        st.sidebar.caption("Số liệu bước từ CV dự đoán — xem sai số vs nhãn ở tab Tổng quan.")
    elif src == "oracle":
        st.sidebar.caption("Số liệu bước từ nhãn HA4M (ground truth) — không phản ánh sai số CV.")

    trials = da.run_trials(run_id)
    man = da.load_manifest(run_id) or {}
    nolab = {t for t, v in man.get("trials", {}).items() if v.get("labels_available") is False}
    trial = None
    if trials:
        if st.session_state.get("trial_sel") not in trials:
            st.session_state["trial_sel"] = trials[0]
        trial = st.sidebar.selectbox("Trial", trials, key="trial_sel",
                                     format_func=lambda t: f"{t}  (subject {t.split('_')[0]})"
                                     + ("  ⚠ không nhãn" if t in nolab else ""))
    else:
        st.sidebar.info("Run chưa có trial nào (chưa chạy 02).")

    rp = da.report_path(run_id)
    if rp.exists():
        st.sidebar.download_button("📄 Mở report.md", rp.read_bytes(), file_name=f"report_{run_id}.md",
                                   mime="text/markdown", width="stretch")
        st.sidebar.caption(f"`{rp.relative_to(REPO)}`")
    else:
        st.sidebar.caption("Run chưa có report.md (chưa chạy 08).")
    return run_id, trial


# ============================ Tab 1 — Tổng quan ============================
@safe("Tổng quan")
def tab_overview(run_id: str, trial: str | None):
    met = da.load_metrics(run_id)
    man = da.load_manifest(run_id) or {}
    if not met:
        st.warning("Run chưa có metrics.json (chưa chạy 05).")
    m = (met or {}).get("trials", {}).get(trial) if trial else None
    if met and trial and m is None:
        st.warning(f"{trial} không có trong metrics của run này (vd trial thiếu nhãn trong run oracle).")
    if m:
        st.subheader(f"{trial}")
        c = st.columns(6)
        c[0].metric("Cycle time", fmt(val(m, "cycle_time_s"), " s"), help=m["cycle_time_s"]["definition"])
        c[1].metric("%VA", fmt(val(m, "va_pct"), " %"), help=m["va_pct"]["definition"])
        il, ir = val(m, "idle_left_s"), val(m, "idle_right_s")
        c[2].metric("Idle (trái + phải)", fmt(il + ir, " s") if il is not None and ir is not None else NA,
                    help=f"trái {fmt(il, ' s')} · phải {fmt(ir, ' s')} · cả 2 tay cùng idle {fmt(val(m, 'idle_both_s'), ' s')}")
        c[3].metric("Số pick", fmt(val(m, "picks_total")) if val(m, "picks_total") is not None else NA,
                    help=(m.get("picks_total") or {}).get("definition"))
        for j, hand, lab in ((4, "left", "Quãng đường cổ tay trái"), (5, "right", "Quãng đường cổ tay phải")):
            v = val(m, f"distance_{hand}")
            c[j].metric(lab, fmt(v, f" {unit(m, f'distance_{hand}', 'px')}", 0) if v is not None else NA)
        ev = m.get("eval_vs_oracle")
        if ev:
            c = st.columns(6)
            err = ev.get("cycle_time_err_s")
            c[0].metric("Sai số CT vs oracle", fmt(err, " s"),
                        delta=f"{err:+.2f} s" if err is not None else None, delta_color="inverse",
                        help="cycle time dự đoán − cycle time theo nhãn")
            c[1].metric("F1@IoU0.5", fmt(ev.get("f1_iou05")), delta=f"MAE bước {fmt(ev.get('mae_step_dur_s'), ' s')}",
                        delta_color="off")
            c[2].metric("Bước dự đoán / nhãn", f"{ev.get('n_pred')} / {ev.get('n_oracle')}")
            c[3].metric("Unclassified", fmt(ev.get("n_unclassified")))
            if ev.get("no_zone"):
                c[4].metric("Bước không có zone", ", ".join(ev["no_zone"]))
        elif da.run_source(run_id) == "pred":
            st.info("Không có eval vs oracle cho trial này (thiếu nhãn) — số liệu chỉ từ CV dự đoán.")

    if met:
        st.markdown("#### Mọi trial trong run")
        rows = []
        for st_, mm in met["trials"].items():
            e = mm.get("eval_vs_oracle", {})
            lab_ok = man.get("trials", {}).get(st_, {}).get("labels_available")
            rows.append({"Trial": st_, "Cycle time (s)": val(mm, "cycle_time_s"), "%VA": val(mm, "va_pct"),
                         "Idle trái (s)": val(mm, "idle_left_s"), "Idle phải (s)": val(mm, "idle_right_s"),
                         "Số pick": val(mm, "picks_total"),
                         f"QĐ trái ({unit(mm, 'distance_left', 'px')})": val(mm, "distance_left"),
                         f"QĐ phải ({unit(mm, 'distance_right', 'px')})": val(mm, "distance_right"),
                         "Sai số CT (s)": e.get("cycle_time_err_s"), "F1@IoU0.5": e.get("f1_iou05"),
                         "Có nhãn": {True: "✓", False: "✗"}.get(lab_ok, "?")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        nolab = [t for t, v in man.get("trials", {}).items() if v.get("labels_available") is False]
        if nolab:
            st.caption(f"⚠ Không có nhãn: {', '.join(nolab)} — không có oracle/eval.")

    with st.expander("Thông tin run (manifest)"):
        if not man:
            st.info("Run không có manifest.json.")
            return
        llm = man.get("llm") or {}
        info = {"run_id": man.get("run_id"), "tạo lúc": man.get("created_at"), "git": man.get("git_hash"),
                "nguồn bước": man.get("source"), "pose model": man.get("pose_model"),
                "LLM": f"{llm.get('provider')} · {llm.get('model') or '—'}", "data_root": man.get("data_root"),
                "chọn subject/trial": man.get("selection"), "tag": man.get("tag") or "—"}
        st.table(pd.DataFrame({"": list(info), "giá trị": [str(v) for v in info.values()]}).set_index(""))
        cmd = man.get("cmd")
        st.markdown("**Lệnh tái lập:**")
        st.code(" ".join(cmd) if isinstance(cmd, list) else str(cmd), language="bash")
        stages = man.get("stages", {})
        if stages:
            st.markdown("**Các bước đã chạy:**")
            st.dataframe(pd.DataFrame([{"stage": k, "lúc": v.get("at"), "argv": " ".join(v.get("argv", [])[1:])}
                                       for k, v in stages.items()]), hide_index=True, width="stretch")
        diff = da.station_diff(run_id)
        if diff:
            st.warning(f"station.yaml của run khác config/station.yaml hiện tại ở: {', '.join(diff)} — "
                       "run dùng bản chụp riêng (outputs/runs/<run>/station.yaml).")
        else:
            st.caption("station.yaml của run trùng config/station.yaml hiện tại.")
        st.json(man, expanded=False)


# ============================ Tab 2 — Video & Timeline ============================
def render_video(run_id: str, trial: str):
    cmd = [sys.executable, str(REPO / "src" / "tools" / "render_demo.py"), "--run", run_id, "--trial", trial, "--force"]
    with st.status("Đang render video demo…", expanded=True) as s:
        log = st.empty()
        lines = []
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
        for ln in p.stdout:
            lines.append(ln.rstrip())
            log.code("\n".join(lines[-12:]), language=None)
        p.wait()
        s.update(label="Render xong" if p.returncode == 0 else f"Render lỗi (exit {p.returncode})",
                 state="complete" if p.returncode == 0 else "error")
    return p.returncode == 0


@st.fragment
def frame_viewer(run_id: str, trial: str, t_max: float, starts: list[tuple[str, float]]):
    key = f"t_{run_id}_{trial}"
    if starts:
        def jump():
            sel = st.session_state.get(f"j_{key}")
            if sel:
                st.session_state[key] = round(dict(starts)[sel], 1)
        st.selectbox("Nhảy tới đầu bước", [s for s, _ in starts], key=f"j_{key}", on_change=jump, index=None,
                     placeholder="chọn bước…")
    t_max = float(round(t_max, 1))
    if key not in st.session_state or not 0 <= st.session_state[key] <= t_max:
        st.session_state[key] = 0.0
    t = st.slider("Thời điểm (s)", 0.0, t_max, key=key, step=0.1)
    img = da.load_frame(run_id, trial, t)
    if img is None:
        st.warning("Không đọc được ảnh gốc (kiểm tra data_root trong manifest).")
    else:
        st.image(img, caption=f"{trial} · t = {t:.1f} s (frame tĩnh có overlay)", width=960)


@safe("Video & Timeline")
def tab_video(run_id: str, trial: str | None):
    if not trial:
        st.info("Chọn trial ở sidebar.")
        return
    steps = {k: df for k in ("oracle", "pred") if (df := da.load_steps(run_id, trial, k)) is not None}
    ev = da.load_events(run_id, trial)

    st.markdown("#### 🎬 Video overlay")
    vp = da.demo_video(run_id, trial)
    if vp.exists():
        st.video(str(vp))
        st.caption(f"`{vp.relative_to(REPO)}` · overlay: zone (màu theo bước), vệt cổ tay ~1 s, bước hiện tại, trạng thái tay")
        if st.button("Render lại video", key=f"rr_{run_id}_{trial}"):
            if render_video(run_id, trial):
                st.rerun()
    else:
        st.info("Run này chưa có video demo cho trial.")
        if st.button("Render video demo (~1-2 phút)", type="primary", key=f"r_{run_id}_{trial}"):
            if render_video(run_id, trial):
                st.rerun()

    st.markdown("#### Timeline bước")
    if not steps:
        st.warning("Run chưa có steps cho trial này (chưa chạy 04).")
    else:
        if len(steps) == 2:
            st.caption("2 hàng: bước theo nhãn (oracle) và bước CV dự đoán (pred) — so lệch trực tiếp.")
        show_ev = st.toggle("Hiện event cổ tay", value=False, key="gantt_ev") and ev is not None
        st.plotly_chart(charts.gantt(steps, ev if show_ev else None), width="stretch", key=f"g_{run_id}_{trial}")

    st.markdown("#### Xem frame tại thời điểm")
    feat = da.load_features(run_id, trial)
    t_max = float(feat["t"].max()) if feat is not None else max((float(d["end_s"].max()) for d in steps.values()), default=0.0)
    src = da.run_source(run_id)
    base = steps.get(src) if src in steps else next(iter(steps.values()), None)
    starts = []
    if base is not None:
        for r in base.itertuples():
            lab = "Unclassified" if r.step_id == 0 else f"B{r.step_id}{f'.{r.repetition}' if r.repetition else ''} · {r.name}"
            starts.append((f"{lab} @ {r.start_s:.1f}s", float(r.start_s)))
    if t_max > 0:
        frame_viewer(run_id, trial, t_max, starts)
    else:
        st.warning(NA)


# ============================ Tab — So sánh video ============================
@safe("So sánh video")
def tab_video_compare(run_id: str, trial: str | None):
    runs = da.list_runs()
    ids = [m["run_id"] for m in runs]
    labels = {m["run_id"]: da.run_label(m) for m in runs}
    st.caption("Phát 2 video overlay cạnh nhau, đồng bộ. **Nhảy tới bước** đưa mỗi video tới đầu bước đó theo mốc "
               "của chính trial ấy → so cách thao tác cùng một bước giữa 2 lần lắp (vd V001 vs V002) hoặc giữa 2 run "
               "(oracle vs pred).")
    sides = []
    for name, col in zip(("A", "B"), st.columns(2)):
        with col:
            st.markdown(f"**Video {name}**")
            r = st.selectbox(f"Run {name}", ids, index=ids.index(run_id) if run_id in ids else 0,
                             format_func=lambda x: labels[x], key=f"vc_run_{name}")
            trs = da.run_trials(r)
            if not trs:
                st.warning("Run chưa có trial nào.")
                sides.append(None)
                continue
            if name == "A":
                idx = trs.index(trial) if trial in trs else 0
            else:
                a_tr = sides[0][1] if sides and sides[0] else None
                others = [x for x in trs if x != a_tr]
                idx = trs.index(others[0]) if others else 0
            t = st.selectbox(f"Trial {name}", trs, index=idx, key=f"vc_trial_{name}_{r}")
            vp = da.demo_video(r, t)
            if vp.exists():
                st.caption(f"✓ `{vp.relative_to(REPO)}` · {vp.stat().st_size / 1e6:.1f} MB")
            else:
                st.warning("Chưa có video overlay cho lựa chọn này.")
                if st.button(f"Render video {name} (~1-2 phút)", key=f"vc_render_{name}_{r}_{t}"):
                    if render_video(r, t):
                        st.rerun()
            sides.append((r, t, vp))
    if not all(sd and sd[2].exists() for sd in sides):
        st.info("Cần có đủ 2 video để so sánh — render video còn thiếu ở trên.")
        return
    if sides[0][:2] == sides[1][:2]:
        st.info("Hai bên đang là cùng một video — chọn trial hoặc run khác cho B.")
    both_oracle = all(da.load_steps(r, t, "oracle") is not None for r, t, _ in sides)
    align = st.radio("Căn bước theo", ["oracle", "run"], horizontal=True, index=0 if both_oracle else 1,
                     format_func=lambda x: "nhãn HA4M (oracle) — chuẩn nhất để căn" if x == "oracle"
                     else "bước của run (oracle hoặc pred tuỳ run)", key="vc_align")
    if align == "oracle" and not both_oracle:
        st.caption("⚠ Một bên không có nhãn → dùng bước của run cho bên đó.")
    payload = []
    for name, (r, t, vp) in zip(("A", "B"), sides):
        steps, src = da.steps_for_run(r, t, "oracle" if align == "oracle" else None)
        payload.append({"title": f"{name} · {t}", "subtitle": f"{labels[r]} · bước: {src or 'không có'}",
                        "b64": da.video_b64(vp), "steps": vc.steps_payload(steps)})
    components.html(vc.build_html(*payload), height=690)
    st.caption("Mốc bước đọc từ steps của từng run (nguồn ghi ở phụ đề). Khoá đồng bộ giữ B = A + độ lệch khi phát.")


# ============================ Tab 3 — Muda ============================
@safe("Muda")
def tab_muda(run_id: str, trial: str | None):
    met = da.load_metrics(run_id)
    if not met or not met.get("trials"):
        st.warning("Run chưa có metrics.json (chưa chạy 05).")
        return
    trials = met["trials"]
    cfg = da.station(run_id)
    st.markdown("#### Yamazumi")
    st.caption("Click legend để ẩn/hiện bước. Số trên thanh = id bước; ? = chưa gán bước (unclassified).")
    st.plotly_chart(charts.yamazumi(trials, cfg.get("cycle_time_target_s")), width="stretch", key=f"y_{run_id}")

    m = trials.get(trial) if trial else None
    st.markdown(f"#### Chi tiết bước — {trial or ''}")
    if m is None:
        st.info("Trial không có trong metrics của run này.")
    else:
        df = pd.DataFrame(m["steps"]["rows"]).rename(columns={
            "step_id": "Bước", "name": "Tên", "va": "VA", "n_rep": "Lần", "dur_s": "Dur (s)", "pct_cycle": "% chu kỳ",
            "start_s": "Start (s)", "end_s": "End (s)", "left_idle_pct": "Idle trái %", "right_idle_pct": "Idle phải %",
            "left_active_pct": "Active trái %", "right_active_pct": "Active phải %"})
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(" · ".join(f"**{k}**: {v}" for k, v in m["steps"]["definition"].items()))

    if len(trials) >= 2:
        st.markdown("#### So sánh dur từng bước giữa trial")
        top = (met.get("mura", {}).get("top_cv_steps") or [{}])[0].get("step_id")
        st.plotly_chart(charts.step_compare(trials, top), width="stretch", key=f"sc_{run_id}")
        if top:
            st.caption(f"Bước chênh nhiều nhất giữa các trial (CV cao nhất, do 05_metrics tính): **B{top}**.")


# ============================ Tab 4 — Mura & Muri ============================
@safe("Mura & Muri")
def tab_mura_muri(run_id: str, trial: str | None):
    met = da.load_metrics(run_id)
    if not met:
        st.warning("Run chưa có metrics.json (chưa chạy 05).")
        return
    mura = met.get("mura", {})
    ct = mura.get("all_trials", {}).get("cycle_time", {})
    st.markdown("#### Mura — biến động")
    if (ct.get("n") or 0) < 2:
        st.info("Mura cần ≥ 2 trial trong run. Ví dụ chạy nhiều trial của cùng subject:")
        st.code("bash run_all.sh --subjects U001 --trials all --oracle --dry-run --tag mura", language="bash")
    else:
        c = st.columns(4)
        c[0].metric("Số trial", ct["n"])
        c[1].metric("Cycle time TB", fmt(ct.get("mean"), " s"))
        c[2].metric("Std", fmt(ct.get("std"), " s"))
        c[3].metric("CV cycle time", fmt(ct.get("cv"), "", 3))
        st.plotly_chart(charts.cycle_box({t: m["cycle_time_s"]["value"] for t, m in met["trials"].items()}, ct.get("mean")),
                        width="stretch", key=f"cb_{run_id}")
        rows = [{"Bước": int(s), "n": v.get("n"), "mean (s)": v.get("mean"), "std (s)": v.get("std"), "CV": v.get("cv")}
                for s, v in mura.get("all_trials", {}).get("steps", {}).items()]
        df = pd.DataFrame(rows)
        if len(df):
            st.markdown("CV theo bước (tô đỏ CV > 0.25 — ngưỡng R6):")
            sty = df.style.format({"mean (s)": "{:.2f}", "std (s)": "{:.2f}", "CV": "{:.3f}"}, na_rep="—")
            cell = getattr(sty, "map", None) or sty.applymap      # pandas < 2.1 chỉ có applymap
            st.dataframe(cell(lambda v: "background-color:#fbe3e3;color:#a12b2b"
                              if isinstance(v, float) and v > 0.25 else "", subset=["CV"]),
                         hide_index=True, width="stretch")
        ws = [{"Subject": s, "n": v["cycle_time"].get("n"), "mean (s)": v["cycle_time"].get("mean"),
               "CV": v["cycle_time"].get("cv"), "ghi chú": v["cycle_time"].get("note", "")}
              for s, v in mura.get("within_subject", {}).items()]
        st.markdown("Trong từng subject:")
        st.dataframe(pd.DataFrame(ws), hide_index=True, width="stretch")
        b = mura.get("between_subjects", {}).get("cycle_time", {})
        st.caption(f"Giữa các subject: n={b.get('n')}, CV={b.get('cv')} {b.get('note', '')}")

    st.markdown("#### Muri — tư thế cổ tay")
    st.caption("⚠ ước lượng 2D, chỉ mang tính sàng lọc — cần đánh giá ergonomic chuẩn để kết luận.")
    rows = []
    for st_, m in met["trials"].items():
        mw = m.get("muri_wrist", {})
        for hand, lab in (("left", "trái"), ("right", "phải")):
            h = mw.get(hand, {})
            cov = val(h, "coverage_pct") if h.get("available") else h.get("coverage_pct")
            rows.append({"Trial": st_, "Tay": lab, "Coverage %": cov, "Góc trung vị (°)": val(h, "median_angle_deg"),
                         "% lệch > 30°": val(h, "dev_gt_thr_pct"), "P90 độ lệch (°)": val(h, "dev_p90_deg"),
                         "Tin cậy": "kém (độ phủ thấp)" if (cov or 0) < MURI_COVERAGE_MIN else "được"})
    if not rows:
        st.info(NA)
        return
    df = pd.DataFrame(rows)

    def fade(row):
        low = (row["Coverage %"] or 0) < MURI_COVERAGE_MIN
        return ["color:#a8a7a1" if low else "" for _ in row]
    st.dataframe(df.style.apply(fade, axis=1), hide_index=True, width="stretch")
    st.caption(f"Dòng mờ: coverage < {MURI_COVERAGE_MIN:g}% — độ phủ thấp, kém tin cậy (thường do tay bị thân che).")


# ============================ Tab 5 — Findings ============================
VALID_BADGE = {"passed": ("✅ đạt", "green"), "failed_after_retry": ("❌ lỗi sau retry", "red"),
               "failed": ("❌ lỗi", "red"), "parse_failed": ("❌ không parse được", "red")}


def citation_chips(cites: list[str], key: str):
    if not cites:
        st.caption("Không có citation.")
        return
    cols = st.columns(min(len(cites), 6))
    for i, c in enumerate(cites):
        with cols[i % len(cols)].popover(c, width="stretch"):
            txt = da.kb_lookup(c)
            st.markdown(txt if txt else f"_Không tìm thấy {c} trong kaizen_kb.md / sop.md_")


@safe("Findings")
def tab_findings(run_id: str, trial: str | None):
    fd = da.load_findings(run_id)
    ex = da.load_explained(run_id)
    if not fd:
        st.warning("Run chưa có findings.json (chưa chạy 06).")
        return
    if ex is None:
        st.warning("Run chưa chạy 07 (chưa có explained.json) — chỉ hiện findings rule-based.")
    else:
        c = st.columns(4)
        prov = ex.get("provider", ex.get("mode"))
        c[0].metric("Provider LLM", prov, help=ex.get("llm_model") or "")
        c[1].metric("Pass lần 1", fmt((ex.get("pass_rate_first_attempt") or 0) * 100, " %", 0)
                    if ex.get("pass_rate_first_attempt") is not None else NA)
        c[2].metric("Pass sau retry", fmt((ex.get("pass_rate") or 0) * 100, " %", 0) if ex.get("pass_rate") is not None else NA)
        c[3].metric("Số finding retry", ex.get("n_retried", 0))
        st.caption(f"Validator: {ex.get('validator', '')}")
    exps = {e["finding_id"]: e for e in (ex or {}).get("explanations", [])}
    findings = fd.get("findings", [])
    if not findings:
        st.info("Không có finding nào vượt ngưỡng trong run này.")
    for i, f in enumerate(findings):
        e = f["evidence"]
        sev = {"high": "🔴", "med": "🟠", "low": "🟡"}.get(f["severity"], "⚪")
        with st.expander(f"{sev} **{f['id']}** · {f['rule']} · {f['type']} · severity {f['severity']} · "
                         f"confidence {f['confidence']} — {e['metric']} = {e['value']} {e['unit']}", expanded=(i == 0)):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**Bằng chứng:** {e['metric']} = **{e['value']} {e['unit']}** · trial {', '.join(e['trials'])}")
                st.markdown(f"**Ảnh hưởng:** {f['impact_s_per_cycle']} s/chu kỳ · điểm xếp hạng {f.get('rank_score')}")
                ctx = {k: v for k, v in e.items() if k not in ("metric", "value", "unit", "trials", "timestamps")}
                if ctx:
                    st.caption("Ngữ cảnh: " + " · ".join(f"{k}={v}" for k, v in ctx.items()))
                if e.get("timestamps"):
                    st.caption(f"Mốc thời gian (s): {e['timestamps'][:5]}")
                x = exps.get(f["id"])
                st.markdown("---")
                if ex is None:
                    st.caption("Chưa có giải thích LLM (run chưa chạy 07).")
                elif x is None:
                    st.caption("LLM không có giải thích cho finding này.")
                else:
                    fin = x.get("final", x)
                    status = x.get("validation")
                    status = status.get("status") if isinstance(status, dict) else status
                    if (ex.get("provider") or ex.get("mode")) == "dry-run":
                        st.badge("dry-run (văn bản giả)", color="gray", icon="🧪")
                    lab, colr = VALID_BADGE.get(status, (str(status), "gray"))
                    if status == "passed" and len(x.get("attempts", [])) > 1:
                        lab = "✅ đạt sau retry"
                    st.badge(lab, color=colr)
                    msg = (x.get("final_check") or {}).get("message")
                    if status != "passed" and msg:
                        st.caption(f"Validator: {msg}")
                    st.markdown(f"**Quan sát:** {fin.get('observation', '')}")
                    st.markdown(f"**Giả thuyết:** {fin.get('hypothesis', '')}")
                    st.markdown(f"**Đề xuất:** {fin.get('recommendation', '')}")
                    cites = (x.get("final_check") or {}).get("citations_normalized") or fin.get("citations", [])
                    st.markdown("**Trích dẫn** (click để xem nguyên văn):")
                    citation_chips(cites, f"{run_id}_{f['id']}")
            with right:
                img = da.asset(run_id, f"evidence_{f['id']}.png")
                if img:
                    st.image(str(img), caption="Frame bằng chứng (08_report)", width="stretch")
                else:
                    st.caption("Chưa có ảnh bằng chứng (chạy 08).")
    notes = fd.get("notes", [])
    if notes:
        st.markdown("#### Rule bị bỏ qua / sanity check")
        for n in notes:
            st.markdown(f"<span class='muted'>• {n}</span>", unsafe_allow_html=True)


# ============================ Tab 6 — So sánh run ============================
@safe("So sánh run")
def tab_compare():
    runs = da.list_runs()
    if len(runs) < 2:
        st.info("Cần ≥ 2 run để so sánh.")
        return
    labels = {m["run_id"]: da.run_label(m) for m in runs}
    ids = st.multiselect("Chọn ≥ 2 run", [m["run_id"] for m in runs], default=[m["run_id"] for m in runs[:2]],
                         format_func=lambda r: f"{labels[r]}  ({r})")
    if len(ids) < 2:
        st.info("Chọn ít nhất 2 run.")
        return
    res = da.compare(ids)
    cols = da.COMPARE_COLS
    df = pd.DataFrame(res["summary"])[[k for k, _ in cols]].rename(columns=dict(cols))
    st.markdown("#### Tổng hợp theo run (trung bình các trial — cùng hàm với `compare_runs.py`)")
    st.dataframe(df, hide_index=True, width="stretch")
    st.plotly_chart(charts.compare_bars(res["summary"], [("cycle_s", "Cycle time (s)"), ("va_pct", "%VA"),
                                                          ("f1", "F1@IoU0.5"), ("pass", "Pass LLM")]),
                    width="stretch", key="cmp_bars")
    if res["per_trial"]:
        st.markdown("#### Cycle time theo trial (s)")
        st.dataframe(pd.DataFrame(res["per_trial"]), hide_index=True, width="stretch")
    st.code("python src/tools/compare_runs.py " + " ".join(ids), language="bash")


# ============================ Tab 7 — Chạy pipeline ============================
def job_cmd(f: dict) -> list[str]:
    cmd = ["bash", "run_all.sh", "--provider", f["provider"]]
    if f["oracle"]:
        cmd.append("--oracle")
    for flag in ("subjects", "trials", "tag", "data_root", "llm_model"):
        if f.get(flag):
            cmd += [f"--{flag.replace('_', '-')}", f[flag]]
    return cmd


@st.fragment(run_every=1.0)
def job_panel():
    job = st.session_state.get("job")
    if not job:
        return
    p: subprocess.Popen = job["proc"]
    log = Path(job["log"]).read_text(encoding="utf-8", errors="replace") if Path(job["log"]).exists() else ""
    rc = p.poll()
    state = "đang chạy" if rc is None else ("xong" if rc == 0 else f"lỗi (exit {rc})")
    st.markdown(f"**Job:** `{' '.join(job['cmd'])}` — {state} · {time.time() - job['t0']:.0f}s")
    if rc is None and st.button("⏹ Hủy job", key="cancel_job"):
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    st.code("\n".join(log.splitlines()[-40:]) or "(chưa có log)", language=None)
    if rc is not None and not job.get("handled"):
        job["handled"] = True
        new = da.latest_run()
        if rc == 0 and new and new not in job["before"]:
            st.session_state["pending_run_sel"] = new
            st.success(f"Run mới: {new}")
        st.rerun(scope="app")


@safe("Chạy pipeline")
def tab_run():
    job = st.session_state.get("job")
    running = bool(job) and job["proc"].poll() is None
    with st.form("run_form"):
        c = st.columns(3)
        subjects = c[0].text_input("Subjects", "U001", help="vd U001,U002 hoặc all")
        trials = c[1].text_input("Trials", "all", help="vd V001,V002 hoặc all")
        data_root = c[2].text_input("Data root", "data/ha4m", help="nhiều thư mục cách nhau dấu phẩy")
        c = st.columns(4)
        source = c[0].radio("Nguồn bước", ["oracle", "pred"], horizontal=True)
        provider = c[1].selectbox("LLM provider", ["dry-run", "gemini", "anthropic"])
        llm_model = c[2].text_input("LLM model (tuỳ chọn)", "")
        tag = c[3].text_input("Tag", "ui")
        submitted = st.form_submit_button("▶ Chạy run_all.sh", type="primary", disabled=running)
    need = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    if provider in need and not any(os.environ.get(k) for k in need[provider]):
        st.warning(f"Thiếu {need[provider][0]} trong môi trường của Streamlit — run sẽ dừng ngay. Chọn dry-run "
                   "hoặc export key rồi khởi động lại `streamlit run app/ui.py`.")
    st.caption("Trial chưa có keypoints sẽ được trích trước (~6 phút/video). Chỉ 1 job một lúc.")
    if submitted and not running:
        f = dict(subjects=subjects.strip(), trials=trials.strip(), data_root=data_root.strip(), oracle=source == "oracle",
                 provider=provider, llm_model=llm_model.strip(), tag=tag.strip())
        cmd = job_cmd(f)
        logd = REPO / "outputs" / "ui_jobs"
        logd.mkdir(parents=True, exist_ok=True)
        logp = logd / f"{time.strftime('%Y%m%d-%H%M%S')}.log"
        fh = open(logp, "w", encoding="utf-8")
        env = dict(os.environ, PY=sys.executable, PYTHONUNBUFFERED="1")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        st.session_state["job"] = {"proc": p, "log": str(logp), "cmd": cmd, "t0": time.time(),
                                   "before": {m["run_id"] for m in da.list_runs()}}
        st.rerun()
    job_panel()


# ============================ Main ============================
run_id, trial = sidebar()
TABS = ["📊 Tổng quan", "🎬 Video & Timeline", "🎞️ So sánh video", "🧱 Muda", "📈 Mura & Muri", "💡 Findings & Copilot",
        "⚖️ So sánh run", "▶ Chạy pipeline"]
t = st.tabs(TABS)
if run_id:
    with t[0]:
        tab_overview(run_id, trial)
    with t[1]:
        tab_video(run_id, trial)
    with t[2]:
        tab_video_compare(run_id, trial)
    with t[3]:
        tab_muda(run_id, trial)
    with t[4]:
        tab_mura_muri(run_id, trial)
    with t[5]:
        tab_findings(run_id, trial)
else:
    for i in range(6):
        with t[i]:
            st.info("Chưa có run nào — dùng tab **Chạy pipeline** hoặc `bash run_all.sh`.")
with t[6]:
    tab_compare()
with t[7]:
    tab_run()

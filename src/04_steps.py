"""[04] events (+ nhãn HA4M) → outputs/steps/<subject>_<trial>_{oracle,pred}.csv (+ _eval.json).

Cột: step_id, repetition, name, start_s, end_s, dur_s, source.

--oracle   : đọc Labels.txt (ha4m_io.load_labels) → _oracle.csv. Mỗi đoạn (action_id, repetition) là 1 dòng;
             end_s = (end_frame + 1) / fps.
mặc định   : predicted baseline từ events + steps trong station.yaml:
             duyệt các bước theo thứ tự SOP (mỗi repetition 1 lần); bước có `zone` → mốc bắt đầu là event
             "pick" (hoặc "reach" — lấy nhanh không dừng hẳn) ĐẦU TIÊN (tay bất kỳ) ở zone đó, sau mốc của bước trước + min_gap_s. Kết thúc = mốc bước
             kế tiếp tìm được; bước cuối kết thúc ở event hoạt động cuối cùng (không idle/lost).
             Mốc được lùi về đầu chuỗi transfer/reach liền trước (lúc tay rời vùng lắp).
             Nếu giữa 2 mốc có bước không làm mốc được (không tìm thấy, hoặc không có zone), đoạn đó gán
             "unclassified" (step_id 0) và các bước liên quan ghi vào cột `missing`.
             TODO v2: thay bằng DTW với golden cycle / ruptures change-point.
Eval (khi có cả 2): sai số start/end từng bước, MAE thời lượng, sai số cycle time, F1@IoU0.5.

    python src/04_steps.py --oracle          # chạy trước, không phụ thuộc CV
    python src/04_steps.py                   # predicted + eval nếu đã có oracle
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, can_write, load_config, parse_stem, r, select_stems, stem, write_json  # noqa: E402
import ha4m_io  # noqa: E402

TAG = "04"
COLS = ["step_id", "repetition", "name", "start_s", "end_s", "dur_s", "source"]


def step_names(cfg: dict) -> dict[int, str]:
    names = {int(s["id"]): s.get("name", "") for s in cfg.get("steps", [])}
    for k, v in ha4m_io.ACTION_NAMES.items():
        names.setdefault(k, v)
    return names


def oracle_steps(root: str, subject: str, trial: str, cfg: dict) -> pd.DataFrame:
    seg = ha4m_io.load_labels(root, subject, trial)
    fps = float(cfg["fps"])
    names = step_names(cfg)
    df = pd.DataFrame({
        "step_id": seg["action_id"].astype(int),
        "repetition": seg["repetition"].astype(int),
        "name": [names.get(int(a), n) for a, n in zip(seg["action_id"], seg["action_name"])],
        "start_s": (seg["start_frame"] / fps).round(3),
        "end_s": ((seg["end_frame"] + 1) / fps).round(3),
    })
    df["dur_s"] = (df["end_s"] - df["start_s"]).round(3)
    df["source"] = "oracle"
    return df[COLS]


def lead_in_start(ev: pd.DataFrame, hand: str, t_pick: float) -> float:
    """Lùi mốc về đầu chuỗi transfer/reach/pause liền trước pick của cùng tay
    (= lúc tay rời vùng lắp để đi lấy linh kiện — khớp ngữ nghĩa "Pick up")."""
    he = ev[ev["hand"] == hand].sort_values("start_s")
    t = t_pick
    while True:
        prev = he[(he["end_s"] - t).abs() < 1e-3]
        if prev.empty or prev["event_type"].iloc[-1] not in LEAD_IN:
            return t
        t = float(prev["start_s"].iloc[-1])


LEAD_IN = {"transfer", "reach", "pause"}


def predicted_steps(ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Baseline: mốc bắt đầu bước = pick/reach đầu tiên ở zone đặc trưng của bước (xem docstring)."""
    gap = float(cfg.get("step_min_gap_s", 1.0))
    picks = ev[ev["event_type"].isin(["pick", "reach"])].sort_values("start_s")
    active = ev[~ev["event_type"].isin(["idle", "lost"])]
    t_end = float(active["end_s"].max()) if len(active) else float(ev["end_s"].max())
    plan = []                       # (step_id, rep, name, zone)
    for s in cfg.get("steps", []):
        for rep in range(int(s.get("reps", 1))):
            plan.append((int(s["id"]), rep, s.get("name", ""), s.get("zone")))
    anchors, cursor = [], -1e9      # (k trong plan, t_bắt_đầu)
    for k, (sid, rep, name, zone) in enumerate(plan):
        if not zone:
            continue
        c = picks[(picks["zone"] == zone) & (picks["start_s"] >= cursor)]
        if len(c):
            t_pick = float(c["start_s"].iloc[0])
            t0 = lead_in_start(ev, c["hand"].iloc[0], t_pick)
            if anchors:
                t0 = max(t0, anchors[-1][1])
            anchors.append((k, t0))
            cursor = t_pick + gap
    rows = []
    for i, (k, t0) in enumerate(anchors):
        t1 = anchors[i + 1][1] if i + 1 < len(anchors) else max(t_end, t0)
        k_next = anchors[i + 1][0] if i + 1 < len(anchors) else len(plan)
        between = [f"{plan[j][0]}.{plan[j][1]}" for j in range(k + 1, k_next)]
        sid, rep, name, _ = plan[k]
        if between and i + 1 < len(anchors):
            # có bước ở giữa không làm mốc được → không tách được ranh giới → unclassified
            rows.append(dict(step_id=0, repetition=0, name="unclassified", start_s=t0, end_s=t1,
                             missing=";".join([f"{sid}.{rep}"] + between)))
        else:
            rows.append(dict(step_id=sid, repetition=rep, name=name, start_s=t0, end_s=t1,
                             missing=";".join(between)))
    df = pd.DataFrame(rows, columns=COLS[:-2] + ["missing"])
    df["dur_s"] = (df["end_s"] - df["start_s"]).round(3)
    df["source"] = "pred"
    matched = {(plan[k][0], plan[k][1]) for k, _ in anchors}
    # not_found: bước CÓ zone mà không thấy pick; no_zone: bước không có zone (không thể làm mốc)
    df.attrs["not_found"] = [f"{s}.{rp}" for s, rp, _, z in plan if z and (s, rp) not in matched]
    df.attrs["no_zone"] = [f"{s}.{rp}" for s, rp, _, z in plan if not z]
    df.attrs["matched_order"] = [plan[k][0] for k, _ in anchors]
    return df[COLS + ["missing"]]


def iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def evaluate(pred: pd.DataFrame, orc: pd.DataFrame) -> dict:
    """So khớp theo (step_id, repetition)."""
    o = orc.set_index(["step_id", "repetition"])
    per, tp = [], 0
    p_valid = pred[pred["step_id"] > 0]
    used = set()
    for row in p_valid.itertuples():
        key = (row.step_id, row.repetition)
        if key not in o.index:
            continue
        g = o.loc[key]
        v = iou(row.start_s, row.end_s, g.start_s, g.end_s)
        if v >= 0.5 and key not in used:
            tp += 1; used.add(key)
        per.append(dict(step_id=row.step_id, repetition=row.repetition, start_err_s=r(row.start_s - g.start_s),
                        end_err_s=r(row.end_s - g.end_s), dur_err_s=r(row.dur_s - g.dur_s), iou=r(v, 3)))
    prec = tp / len(p_valid) if len(p_valid) else 0.0
    rec = tp / len(orc) if len(orc) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ct_p = (p_valid["end_s"].max() - p_valid["start_s"].min()) if len(p_valid) else np.nan
    ct_o = orc["end_s"].max() - orc["start_s"].min()
    return dict(
        n_pred=len(p_valid), n_oracle=len(orc), n_unclassified=int((pred["step_id"] == 0).sum()),
        mae_step_dur_s=r(np.mean([abs(x["dur_err_s"]) for x in per])) if per else None,
        mae_start_s=r(np.mean([abs(x["start_err_s"]) for x in per])) if per else None,
        mae_end_s=r(np.mean([abs(x["end_err_s"]) for x in per])) if per else None,
        cycle_time_pred_s=r(ct_p), cycle_time_oracle_s=r(ct_o), cycle_time_err_s=r(ct_p - ct_o),
        precision_iou05=r(prec, 3), recall_iou05=r(rec, 3), f1_iou05=r(f1, 3), per_step=per)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--oracle", action="store_true", help="xuất bước từ nhãn HA4M")
    ap.add_argument("--events-dir", default="outputs/events")
    ap.add_argument("--out-dir", default="outputs/steps")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out_dir = Path(a.out_dir)

    if a.oracle:
        stems = select_stems([stem(s, t) for s, t in ha4m_io.list_trials(a.data_root)], a.subjects, a.trials)
        for st in stems:
            out = out_dir / f"{st}_oracle.csv"
            if not can_write(out, a.force, TAG):
                continue
            df = oracle_steps(a.data_root, *parse_stem(st), cfg)
            df.to_csv(out, index=False)
            print(f"[{TAG}] {st}: {len(df)} đoạn oracle, cycle {df['end_s'].max() - df['start_s'].min():.2f}s → {out}")
        return

    if not cfg.get("steps"):
        sys.exit(f"[{TAG}] station.yaml chưa có steps")
    stems = select_stems([p.stem for p in Path(a.events_dir).glob("*.csv")], a.subjects, a.trials)
    for st in stems:
        out = out_dir / f"{st}_pred.csv"
        if not can_write(out, a.force, TAG):
            continue
        ev = pd.read_csv(Path(a.events_dir) / f"{st}.csv")
        df = predicted_steps(ev, cfg)
        df.to_csv(out, index=False)
        nf = df.attrs.get("not_found", [])
        write_json(out_dir / f"{st}_pred_info.json", {k: df.attrs.get(k, []) for k in ("not_found", "no_zone", "matched_order")})
        print(f"[{TAG}] {st}: {int((df['step_id'] > 0).sum())} bước dự đoán, "
              f"{int((df['step_id'] == 0).sum())} unclassified, không tìm thấy: {nf or 'không'}, "
              f"không có zone: {df.attrs.get('no_zone')} → {out}")
        orc_p = out_dir / f"{st}_oracle.csv"
        if orc_p.exists():
            e = evaluate(df, pd.read_csv(orc_p))
            e["not_found"] = nf
            e["no_zone"] = df.attrs.get("no_zone", [])
            write_json(out_dir / f"{st}_eval.json", e)
            print(f"[{TAG}] {st} eval: MAE dur {e['mae_step_dur_s']}s, cycle err {e['cycle_time_err_s']}s, "
                  f"F1@0.5 {e['f1_iou05']}")


if __name__ == "__main__":
    main()

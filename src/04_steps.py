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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaizen.analyzer.steps import (COLS, LEAD_IN, align_steps, evaluate, iou, lead_in_start,  # noqa: E402,F401
                                   oracle_steps, predicted_steps, step_names)
from kaizen.core.cli import base_parser, setup, stage_dir  # noqa: E402
from kaizen.core.config import load_config  # noqa: E402
from kaizen.core.io import can_write, parse_stem, select_stems, stem, write_json  # noqa: E402
from kaizen.skeleton import ha4m_io  # noqa: E402

TAG = "04"


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--oracle", action="store_true", help="xuất bước từ nhãn HA4M")
    ap.add_argument("--events-dir", default=None, help="mặc định <run>/events")
    ap.add_argument("--out-dir", default=None, help="mặc định <run>/steps")
    a = ap.parse_args()
    ctx = setup(a, TAG)
    events_dir = stage_dir(ctx, a.events_dir, "events", "outputs/events")
    out_dir = stage_dir(ctx, a.out_dir, "steps", "outputs/steps")
    cfg = load_config(a.config)

    if a.oracle:
        stems = select_stems([stem(s, t) for s, t in ha4m_io.list_trials(a.data_root)], a.subjects, a.trials)
        labels, n_ok = {}, 0
        for st in stems:
            sub, tri = parse_stem(st)
            has = ha4m_io.labels_path(a.data_root, sub, tri) is not None
            labels[st] = {"labels_available": has}
            if not has:
                print(f"[{TAG}] cảnh báo: {st} không có Labels*.txt → bỏ qua oracle/eval cho trial này "
                      "(labels_available=false)", file=sys.stderr)
                continue
            out = out_dir / f"{st}_oracle.csv"
            if not can_write(out, a.force, TAG):
                continue
            df = oracle_steps(a.data_root, sub, tri, cfg)
            df.to_csv(out, index=False)
            n_ok += 1
            print(f"[{TAG}] {st}: {len(df)} đoạn oracle, cycle {df['end_s'].max() - df['start_s'].min():.2f}s → {out}")
        if ctx:
            ctx.set_trials(**labels)
            ctx.record_stage("04_steps_oracle", n_trials=n_ok,
                             no_labels=[s for s, v in labels.items() if not v["labels_available"]])
        return

    if not cfg.get("steps"):
        print(f"[{TAG}] station.yaml chưa có steps", file=sys.stderr)
        sys.exit(1)
    stems = select_stems([p.stem for p in events_dir.glob("*.csv")], a.subjects, a.trials)
    for st in stems:
        out = out_dir / f"{st}_pred.csv"
        if not can_write(out, a.force, TAG):
            continue
        res = align_steps(pd.read_csv(events_dir / f"{st}.csv"), "pred", cfg)
        df, nf = res.steps, res.info.not_found
        df.to_csv(out, index=False)
        write_json(out_dir / f"{st}_pred_info.json", res.info.to_json_dict())
        print(f"[{TAG}] {st}: {int((df['step_id'] > 0).sum())} bước dự đoán, "
              f"{int((df['step_id'] == 0).sum())} unclassified, không tìm thấy: {nf or 'không'}, "
              f"không có zone: {res.info.no_zone} → {out}")
        orc_p = out_dir / f"{st}_oracle.csv"
        if orc_p.exists():
            e = evaluate(df, pd.read_csv(orc_p))
            e.not_found = nf
            e.no_zone = res.info.no_zone
            write_json(out_dir / f"{st}_eval.json", e.to_json_dict())
            print(f"[{TAG}] {st} eval: MAE dur {e.mae_step_dur_s}s, cycle err {e.cycle_time_err_s}s, "
                  f"F1@0.5 {e.f1_iou05}")
        else:
            print(f"[{TAG}] {st}: không có {orc_p.name} → bỏ qua eval")
    if ctx:
        ctx.record_stage("04_steps_pred", n_trials=len(stems))


if __name__ == "__main__":
    main()

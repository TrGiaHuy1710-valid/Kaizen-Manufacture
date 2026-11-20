#!/usr/bin/env bash
# Thí nghiệm pilot khi mới có 2 người: CV 2 fold theo video trên U001 (chọn epoch + so cách giải mã) → train cuối trên
# cả U001 → test MỘT LẦN trên U002 (người chưa từng thấy). U002 không tham gia bất kỳ lựa chọn nào trước bước test.
#   cd /storage/huytg/KaizenMotionAnalyzer && conda activate courier310   # GPU (hoặc kaizen-py311: CPU)
#   bash src/tools/exp_u1cv_u2test.sh [--seeds "0 1 2"] [--max-epochs 150] [--threads 3] [--decodes "argmax sop"]
#   bash src/tools/exp_u1cv_u2test.sh --tag _spd --train-args "--speed-min 0.667 --speed-max 1.5"   # augment tốc độ
# Kết quả: models/segmenter/u1cv<TAG>_f{1,2}_s*/ (CV), models/segmenter/u1final<TAG>_s*/test_epoch_XXXX_<decode>/ (test),
#          outputs/experiments/u1cv_u2test<TAG>/ (epoch chọn, CV val + test theo từng cách giải mã).
set -euo pipefail
cd "$(dirname "$0")/../.."
SEEDS="0 1 2"; MAX_EP=150; THREADS=3; TEST_TRIAL="U002_V001"; TAG=""; TRAIN_ARGS=""; DECODES="argmax"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --max-epochs) MAX_EP="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --test) TEST_TRIAL="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    --decodes) DECODES="$2"; shift 2 ;;
    *) echo "tham số lạ $1" >&2; exit 1 ;;
  esac
done
OUT=outputs/experiments/u1cv_u2test$TAG; mkdir -p "$OUT"
SEG="python -m kaizen.train.segmenter"
read -r -a XARGS <<< "$TRAIN_ARGS"
COMMON=(--eval-every 2 --threads "$THREADS" "${XARGS[@]}")
VAL_OF=([1]=U001_V002 [2]=U001_V001)

echo "== 1/5 split cố định (U002 chỉ ở test) =="
mk() { [[ -f "data/splits/$1.json" ]] || python -m kaizen.train.split make --name "$1" --only-ready --strategy trial "${@:2}"; }
mk u1cv_f1 --pin-train U001_V001 --val U001_V002 --test "$TEST_TRIAL"
mk u1cv_f2 --pin-train U001_V002 --val U001_V001 --test "$TEST_TRIAL"
mk u1final --val-frac 0 --test "$TEST_TRIAL"

echo "== 2/5 CV 2 fold × seed trên U001 (không early stop: lấy đủ đường val để chọn epoch) =="
for s in $SEEDS; do for f in 1 2; do
  n="u1cv${TAG}_f${f}_s${s}"
  [[ -f "models/segmenter/$n/model.pt" ]] && { echo "đã có $n"; continue; }
  $SEG train --split-file "data/splits/u1cv_f$f.json" --name "$n" --epochs "$MAX_EP" --patience 100000 \
    --ckpt-every 10 --seed "$s" "${COMMON[@]}" | grep -E "VAL @|đã lưu" || true
done; done

echo "== 3/5 chọn epoch (bội 10, có checkpoint) = argmax val F1@50 trung bình fold × seed (làm mượt 5 điểm) =="
EP=$(python - "$OUT" "$TAG" $SEEDS <<'EOF'
import sys, json, pandas as pd
out, tag, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
logs = []
for s in seeds:
    for f in (1, 2):
        d = pd.read_csv(f"models/segmenter/u1cv{tag}_f{f}_s{s}/train_log.csv").dropna(subset=["val_f1@50"])
        logs.append(d.set_index("epoch")[["val_f1@50", "val_acc", "val_edit", "val_loss"]].add_suffix(f"|f{f}s{s}"))
df = pd.concat(logs, axis=1)
m = pd.DataFrame({k: df[[c for c in df if c.startswith(k + "|")]].mean(1) for k in ("val_f1@50", "val_acc", "val_edit", "val_loss")})
m["val_f1@50_std"] = df[[c for c in df if c.startswith("val_f1@50|")]].std(1)
sm = m["val_f1@50"].rolling(5, center=True, min_periods=1).mean()
ep = int(sm[sm.index % 10 == 0].idxmax())
m.round(3).to_csv(f"{out}/cv_curve.csv")
json.dump({"chosen_epoch": ep, "cv_val_f1@50_at_epoch": round(float(m.loc[ep, "val_f1@50"]), 2),
           "cv_val_f1@50_std": round(float(m.loc[ep, "val_f1@50_std"]), 2),
           "cv_val_acc": round(float(m.loc[ep, "val_acc"]), 2), "cv_val_edit": round(float(m.loc[ep, "val_edit"]), 2),
           "runs": len(logs)}, open(f"{out}/chosen_epoch.json", "w"), indent=2)
print(ep)
EOF
)
CK=$(printf "epoch_%04d" "$EP")
echo "epoch chọn: $EP  ($OUT/chosen_epoch.json)"

echo "== 4/5 CV val theo từng cách giải mã ($DECODES) @ $CK — để CHỌN cách giải mã mà không nhìn test =="
for d in $DECODES; do for s in $SEEDS; do for f in 1 2; do
  n="u1cv${TAG}_f${f}_s${s}"
  $SEG test --model "models/segmenter/$n" --checkpoint "$CK" --trials "${VAL_OF[$f]}" --decode "$d" \
    --out-dir "models/segmenter/$n/cv_val_${CK}_$d" > /dev/null 2>&1
done; done; done

echo "== 5/5 train cuối trên U001_V001+V002 (${EP} epoch) → test $TEST_TRIAL =="
for s in $SEEDS; do
  n="u1final${TAG}_s${s}"
  [[ -f "models/segmenter/$n/model.pt" ]] || $SEG train --split-file data/splits/u1final.json --name "$n" \
    --epochs "$EP" --patience 100000 --ckpt-every 10 --seed "$s" "${COMMON[@]}" | grep -E "đã lưu" || true
  for d in $DECODES; do
    $SEG test --model "models/segmenter/$n" --checkpoint "$CK" --decode "$d" | grep -E "TEST" || true
  done
done
python - "$OUT" "$CK" "$TAG" "$DECODES" $SEEDS <<'EOF'
import sys, json, pandas as pd
out, ck, tag, decodes, seeds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(), sys.argv[5:]
cols = ["acc", "edit", "f1@10", "f1@25", "f1@50", "macro_f1_frames", "f1_iou05", "mae_step_dur_s", "cycle_time_err_s"]
def load(d):
    r = pd.read_csv(f"{d}/per_trial.csv")
    r["macro_f1_frames"] = json.load(open(f"{d}/summary.json"))["macro_f1_frames"]
    return r
res, rows = {}, []
for dec in decodes:
    cv = pd.concat([load(f"models/segmenter/u1cv{tag}_f{f}_s{s}/cv_val_{ck}_{dec}").assign(seed=int(s), fold=f)
                    for s in seeds for f in (1, 2)])
    te = pd.concat([load(f"models/segmenter/u1final{tag}_s{s}/test_{ck}_{dec}").assign(seed=int(s)) for s in seeds])
    for name, df in (("cv_val", cv), ("test", te)):
        rows.append(df.assign(decode=dec, part=name))
        res[f"{name}|{dec}"] = {c: {"mean": round(float(df[c].mean()), 3), "std": round(float(df[c].std(ddof=0)), 3)}
                                for c in cols if c in df}
pd.concat(rows).to_csv(f"{out}/results_long.csv", index=False)
json.dump({"checkpoint": ck, "seeds": seeds, **res}, open(f"{out}/results.json", "w"), indent=2)
tab = pd.DataFrame({k: {c: f"{v[c]['mean']:.2f} ± {v[c]['std']:.2f}" for c in v} for k, v in res.items()}).T
print(tab.to_string())
EOF
echo "xong → $OUT"

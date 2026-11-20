#!/usr/bin/env bash
# Thống kê → trích keypoints còn thiếu → chia split (cố định) → train MS-TCN → test (nếu split có test).
# Chạy từ repo root trên server:
#   cd /storage/huytg/KaizenMotionAnalyzer && conda activate courier310   # GPU (hoặc kaizen-py311: CPU)
#   bash src/tools/train_ha4m.sh                       # split v1, model mstcn_v1
#   bash src/tools/train_ha4m.sh --split v2 --epochs 150 --threads 4
#   bash src/tools/train_ha4m.sh --split v2 --strategy trial --val U001_V002   (đối số lạ → chuyển cho `split make`)
set -euo pipefail

DATA_ROOT="${HA4M_ROOT:-data/ha4m}"
KP_DIR="data/keypoints"
POSE="mediapipe"
SPLIT="v1"
EPOCHS=100
THREADS=0
DEVICE="auto"
EXTRA_SPLIT=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --pose-model) POSE="$2"; shift 2 ;;
    *) EXTRA_SPLIT+=("$1"); shift ;;
  esac
done
SPLIT_FILE="data/splits/${SPLIT}.json"
MODEL="mstcn_${SPLIT}"

echo "== 1/5 thống kê dữ liệu ($DATA_ROOT) =="
python -m kaizen.train.split stats --data-root "$DATA_ROOT" --keypoints-dir "$KP_DIR" --pose-model "$POSE" \
  | sed -n '1,/^## Theo video/p'

echo "== 2/5 trích keypoints cho video có ảnh Color nhưng chưa có .npz (bỏ qua video đã có) =="
python src/01_extract.py --data-root "$DATA_ROOT" --subjects all --trials all --model "$POSE" --out "$KP_DIR"

echo "== 3/5 split $SPLIT_FILE =="
if [[ -f "$SPLIT_FILE" ]]; then
  echo "đã có split, dùng lại (split phải cố định; muốn chia lại thì đổi --split)"
else
  python -m kaizen.train.split make --name "$SPLIT" --data-root "$DATA_ROOT" --keypoints-dir "$KP_DIR" \
    --pose-model "$POSE" --only-ready "${EXTRA_SPLIT[@]}"
fi
python -m kaizen.train.split show "$SPLIT_FILE"

echo "== 4/5 train models/segmenter/$MODEL =="
RESUME=()
[[ -f "models/segmenter/$MODEL/checkpoints/last.pt" && ! -f "models/segmenter/$MODEL/model.pt" ]] && RESUME=(--resume)
python -m kaizen.train.segmenter train --split-file "$SPLIT_FILE" --name "$MODEL" --data-root "$DATA_ROOT" \
  --keypoints-dir "$KP_DIR" --pose-model "$POSE" --epochs "$EPOCHS" --threads "$THREADS" --device "$DEVICE" \
  --eval-every 2 --ckpt-every 10 --patience 20 "${RESUME[@]}"

echo "== 5/5 test =="
if python -c "import json,sys; sys.exit(0 if json.load(open('$SPLIT_FILE'))['test'] else 1)"; then
  python -m kaizen.train.segmenter test --model "models/segmenter/$MODEL" --data-root "$DATA_ROOT" \
    --keypoints-dir "$KP_DIR" --pose-model "$POSE"
else
  echo "split không có tập test (quá ít video) → bỏ qua. Xem val: models/segmenter/$MODEL/train_curve.png"
fi
echo "TensorBoard: tensorboard --logdir models/segmenter --port 6006"

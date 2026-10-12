#!/usr/bin/env bash
# Chạy toàn luồng 00→08. Dừng ngay khi lỗi.
#   bash run_all.sh [--oracle] [--model mediapipe|rtmw] [--subjects U001,U002] [--trials V001]
#                   [--dry-run] [--force] [--data-root data/ha4m]
# --oracle : 05–08 dùng bước từ nhãn HA4M (mặc định: bước dự đoán; eval vs nhãn vẫn được tính nếu có nhãn)
# --dry-run: 07 sinh giải thích giả, không gọi API
# --force  : chạy lại cả 00/01 (trích keypoints chậm ~6 phút/video). 02–08 luôn được tính lại (--force).
# Python: dùng $PY nếu đặt, mặc định `python` (hãy `conda activate kaizen-py311` trước).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
MODEL=mediapipe; ORACLE=0; DRY=""; FORCE=""; SUBJ=""; TRI=""; ROOT="${HA4M_ROOT:-data/ha4m}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --oracle) ORACLE=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --subjects) SUBJ="$2"; shift 2 ;;
    --trials) TRI="$2"; shift 2 ;;
    --dry-run) DRY="--dry-run"; shift ;;
    --force) FORCE="--force"; shift ;;
    --data-root) ROOT="$2"; shift 2 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "Tham số lạ: $1" >&2; exit 1 ;;
  esac
done
SEL=(--data-root "$ROOT" --subjects "$SUBJ" --trials "$TRI")
SOURCE=$([[ $ORACLE == 1 ]] && echo oracle || echo pred)

step() {  # step <tên> <lệnh...>
  local name="$1"; shift
  local t0=$(date +%s.%N)
  echo "=============== $name ==============="
  "$@"
  printf "[run_all] %s xong trong %.1fs\n" "$name" "$(echo "$(date +%s.%N) - $t0" | bc)"
}

T_ALL=$(date +%s)
step "00 inspect"  $PY src/00_inspect.py --data-root "$ROOT" $FORCE
step "01 extract"  $PY src/01_extract.py "${SEL[@]}" --model "$MODEL" $FORCE
step "02 features" $PY src/02_features.py "${SEL[@]}" --model "$MODEL" --force
step "03 events"   $PY src/03_events.py "${SEL[@]}" --force --print-timeline
step "04 oracle"   $PY src/04_steps.py "${SEL[@]}" --oracle --force
step "04 predicted" $PY src/04_steps.py "${SEL[@]}" --force
step "05 metrics ($SOURCE)" $PY src/05_metrics.py "${SEL[@]}" --source "$SOURCE" --force
step "06 findings" $PY src/06_findings.py --force
step "07 explain"  $PY src/07_explain.py --force $DRY
step "08 report"   $PY src/08_report.py "${SEL[@]}" --force
echo "[run_all] Tổng thời gian: $(( $(date +%s) - T_ALL ))s → outputs/report.md"

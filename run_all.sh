#!/usr/bin/env bash
# Chạy toàn luồng 00→08 vào MỘT run mới outputs/runs/<run_id>/ (không ghi đè run cũ). Dừng ngay khi lỗi.
#   bash run_all.sh [--oracle] [--model mediapipe|rtmw] [--subjects U001,U002|all] [--trials V001|all]
#                   [--provider gemini|anthropic|dry-run] [--llm-model NAME] [--dry-run] [--tag TEXT]
#                   [--data-root data/ha4m[,data/ha4m_new]] [--run RUN_ID] [--force]
# --oracle   : 05–08 dùng bước từ nhãn HA4M (mặc định: bước dự đoán; eval vs nhãn vẫn tính nếu có nhãn)
# --provider : LLM cho 07 (mặc định env LLM_PROVIDER, không có thì gemini); --dry-run = --provider dry-run
# --tag      : nối vào run_id, vd _demo1
# --run      : ghi vào run đã có thay vì tạo run mới
# --force    : chạy lại cả 00/01 (trích keypoints chậm ~6 phút/video). 02–08 luôn tính lại trong run.
# Python: dùng $PY nếu đặt, mặc định `python` (hãy `conda activate kaizen-py311` trước).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
MODEL=mediapipe; ORACLE=0; FORCE=""; SUBJ=""; TRI=""; ROOT="${HA4M_ROOT:-data/ha4m}"
PROVIDER="${LLM_PROVIDER:-gemini}"; LLM_MODEL=""; TAG=""; RUN_ID=""
ORIG_ARGS="$*"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --oracle) ORACLE=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --subjects) SUBJ="$2"; shift 2 ;;
    --trials) TRI="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --dry-run) PROVIDER="dry-run"; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --run) RUN_ID="$2"; shift 2 ;;
    --force) FORCE="--force"; shift ;;
    --data-root) ROOT="$2"; shift 2 ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "Tham số lạ: $1" >&2; exit 1 ;;
  esac
done
SOURCE=$([[ $ORACLE == 1 ]] && echo oracle || echo pred)

# Kiểm tra key LLM sớm để không chạy hết luồng rồi mới hỏng ở 07
if [[ "$PROVIDER" == "gemini" && -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; then
  echo "[run_all] thiếu GEMINI_API_KEY cho --provider gemini (hoặc dùng --dry-run / --provider anthropic)" >&2; exit 1
fi
if [[ "$PROVIDER" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  echo "[run_all] thiếu ANTHROPIC_API_KEY cho --provider anthropic" >&2; exit 1
fi

SEL=(--data-root "$ROOT" --subjects "$SUBJ" --trials "$TRI")
LLM_ARGS=(--provider "$PROVIDER"); [[ -n "$LLM_MODEL" ]] && LLM_ARGS+=(--llm-model "$LLM_MODEL")

step() {  # step <tên> <lệnh...>
  local name="$1"; shift
  local t0; t0=$(date +%s.%N)
  echo "=============== $name ==============="
  "$@"
  printf "[run_all] %s xong trong %.1fs\n" "$name" "$(echo "$(date +%s.%N) - $t0" | bc)"
}

T_ALL=$(date +%s)
step "00 inspect" $PY src/00_inspect.py --data-root "$ROOT" $FORCE
step "01 extract" $PY src/01_extract.py "${SEL[@]}" --model "$MODEL" $FORCE

if [[ -z "$RUN_ID" ]]; then
  RUN_ID=$($PY src/run_io.py new --source "$SOURCE" --pose "$MODEL" --llm "$PROVIDER" --llm-model "$LLM_MODEL" \
           --tag "$TAG" --data-root "$ROOT" --subjects "$SUBJ" --trials "$TRI" --cmd "bash run_all.sh $ORIG_ARGS")
fi
RUN=(--run "$RUN_ID")
echo "[run_all] run: $RUN_ID → outputs/runs/$RUN_ID/"

step "02 features"  $PY src/02_features.py "${RUN[@]}" "${SEL[@]}" --model "$MODEL" --force
step "03 events"    $PY src/03_events.py "${RUN[@]}" "${SEL[@]}" --force --print-timeline
step "04 oracle"    $PY src/04_steps.py "${RUN[@]}" "${SEL[@]}" --oracle --force
step "04 predicted" $PY src/04_steps.py "${RUN[@]}" "${SEL[@]}" --force
step "05 metrics ($SOURCE)" $PY src/05_metrics.py "${RUN[@]}" "${SEL[@]}" --source "$SOURCE" --force
step "06 findings"  $PY src/06_findings.py "${RUN[@]}" --force
step "07 explain ($PROVIDER)" $PY src/07_explain.py "${RUN[@]}" --force "${LLM_ARGS[@]}"
step "08 report"    $PY src/08_report.py "${RUN[@]}" "${SEL[@]}" --force
echo "[run_all] Tổng thời gian: $(( $(date +%s) - T_ALL ))s"
echo "[run_all] Report: outputs/runs/$RUN_ID/report/report.md"

#!/usr/bin/env bash
# Two-GPU task queue for BUSI_5fold multi-task ablations.
#
# Foreground:
#   PYTHON_BIN=/path/to/python GPU0=0 GPU1=1 bash run_all_multitask_ablations.sh [all|1|2|3|4|5]
# Background (detached, with a timestamped log and PID):
#   PYTHON_BIN=/path/to/python GPU0=0 GPU1=1 bash run_all_multitask_ablations.sh background [all|1|2|3|4|5]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKGROUND_MODE=0
if [[ "${1:-}" == "background" ]]; then
  BACKGROUND_MODE=1
  shift
fi

FOLD_SELECTOR="${1:-all}"
if [[ "$BACKGROUND_MODE" -eq 1 ]]; then
  LOG_DIR="$SCRIPT_DIR/logs"
  mkdir -p "$LOG_DIR"
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="$LOG_DIR/multitask_ablations_${FOLD_SELECTOR}_${RUN_ID}.log"
  PID_FILE="$LOG_DIR/multitask_ablations_${FOLD_SELECTOR}_${RUN_ID}.pid"
  nohup env PYTHON_BIN="${PYTHON_BIN:-python3}" GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" \
    bash "$SCRIPT_DIR/run_all_multitask_ablations.sh" "$FOLD_SELECTOR" \
    >"$LOG_FILE" 2>&1 < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$PID_FILE"
  echo "Background multi-task training started."
  echo "PID: $PID"
  echo "Log: $LOG_FILE"
  echo "PID file: $PID_FILE"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
EXPERIMENTS=(
  "1_Classification_Segmentation"
  "2_Edge_Segmentation"
  "3_Edge_Classification_Segmentation"
  "4_Edge_EdgeAttention_Segmentation"
  "5_Edge_EdgeAttention_classification_Segmentation"
)

TASKS=()
if [[ "$FOLD_SELECTOR" == "all" ]]; then
  FOLDS=(1 2 3 4 5)
elif [[ "$FOLD_SELECTOR" =~ ^[1-5]$ ]]; then
  FOLDS=("$FOLD_SELECTOR")
else
  echo "Usage: $0 [background] [all|1|2|3|4|5]" >&2
  exit 2
fi
for experiment in "${EXPERIMENTS[@]}"; do
  for fold in "${FOLDS[@]}"; do TASKS+=("${experiment}:${fold}"); done
done

run_worker() {
  local gpu="$1" start="$2" task experiment fold
  for ((i=start; i<${#TASKS[@]}; i+=2)); do
    task="${TASKS[i]}"; experiment="${task%%:*}"; fold="${task##*:}"
    echo "===== GPU ${gpu} | multi-task | ${experiment} | fold ${fold} ====="
    CUDA_VISIBLE_DEVICES="$gpu" PYTHON_BIN="$PYTHON_BIN" \
      bash "$SCRIPT_DIR/$experiment/run_busi_5fold.sh" "$fold"
  done
}

echo "Launching ${#TASKS[@]} multi-task-ablation tasks concurrently on GPUs ${GPU0} and ${GPU1}."
run_worker "$GPU0" 0 & worker0_pid=$!
run_worker "$GPU1" 1 & worker1_pid=$!
wait "$worker0_pid"
wait "$worker1_pid"
echo "All requested BUSI_5fold multi-task ablations completed."

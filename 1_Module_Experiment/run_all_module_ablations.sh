#!/usr/bin/env bash
# Two-GPU task queue for BUSI_5fold module ablations.
# Every worker runs independent (experiment, fold) tasks, so both GPUs train
# concurrently and no result directory is shared.
# Usage:
#   ./run_all_module_ablations.sh start [all|1|2|3|4|5]  # detached
#   ./run_all_module_ablations.sh status|log|stop
#   ./run_all_module_ablations.sh [all|1|2|3|4|5]        # foreground

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/.module_ablation_run}"
PID_FILE="$STATE_DIR/run.pid"
CURRENT_LOG_FILE="$STATE_DIR/current.log"
LOG_DIR="$SCRIPT_DIR/logs"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_background() {
  local selector="${1:-all}" stamp log_file pid
  if is_running; then
    echo "Module ablations are already running (PID $(cat "$PID_FILE"))."
    echo "Log: $(cat "$CURRENT_LOG_FILE" 2>/dev/null || echo unknown)"
    exit 1
  fi
  mkdir -p "$STATE_DIR" "$LOG_DIR"
  rm -f "$PID_FILE"
  stamp="$(date '+%Y%m%d_%H%M%S')"
  log_file="$LOG_DIR/module_ablations_${selector}_${stamp}.log"
  printf '%s\n' "$log_file" > "$CURRENT_LOG_FILE"
  nohup setsid env PYTHON_BIN="$PYTHON_BIN" GPU0="$GPU0" GPU1="$GPU1" \
    bash "$SCRIPT_DIR/$(basename "$0")" run "$selector" > "$log_file" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Failed to start. Check: $log_file" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  echo "Started module ablations in background."
  echo "PID: $pid"
  echo "Log: $log_file"
  echo "Status: $0 status"
  echo "Follow log: $0 log"
}

show_status() {
  if is_running; then
    echo "RUNNING (PID $(cat "$PID_FILE"))"
    echo "Log: $(cat "$CURRENT_LOG_FILE" 2>/dev/null || echo unknown)"
  else
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
    echo "NOT RUNNING"
    [[ -f "$CURRENT_LOG_FILE" ]] && echo "Last log: $(cat "$CURRENT_LOG_FILE")"
  fi
}

follow_log() {
  [[ -f "$CURRENT_LOG_FILE" ]] || { echo "No run log is recorded yet." >&2; exit 1; }
  local log_file
  log_file="$(cat "$CURRENT_LOG_FILE")"
  [[ -f "$log_file" ]] || { echo "Log file not found: $log_file" >&2; exit 1; }
  tail -n 80 -f "$log_file"
}

stop_background() {
  if ! is_running; then
    echo "Module ablations are not running."
    rm -f "$PID_FILE"
    exit 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping process group $pid..."
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "Process did not stop within 20 seconds; forcing termination."
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "Stopped."
}

COMMAND="${1:-all}"
case "$COMMAND" in
  start) start_background "${2:-all}"; exit 0 ;;
  status) show_status; exit 0 ;;
  log) follow_log; exit 0 ;;
  stop) stop_background; exit 0 ;;
  run) FOLD_SELECTOR="${2:-all}" ;;
  all|1|2|3|4|5) FOLD_SELECTOR="$COMMAND" ;;
  *) echo "Usage: $0 {start [all|1|2|3|4|5]|status|log|stop|all|1|2|3|4|5}" >&2; exit 2 ;;
esac
EXPERIMENTS=(
  "1_Base_model"
  "2_Base_HFD"
  "3_Base_HFD_DetailFusion"
  "4_Base_MSAG"
  "5_Base_HFD_DetailFusion_MSAG"
)

TASKS=()
if [[ "$FOLD_SELECTOR" == "all" ]]; then
  FOLDS=(1 2 3 4 5)
elif [[ "$FOLD_SELECTOR" =~ ^[1-5]$ ]]; then
  FOLDS=("$FOLD_SELECTOR")
else
  echo "Usage: $0 [all|1|2|3|4|5]" >&2
  exit 2
fi
for experiment in "${EXPERIMENTS[@]}"; do
  for fold in "${FOLDS[@]}"; do TASKS+=("${experiment}:${fold}"); done
done

run_worker() {
  local gpu="$1" start="$2" task experiment fold
  for ((i=start; i<${#TASKS[@]}; i+=2)); do
    task="${TASKS[i]}"; experiment="${task%%:*}"; fold="${task##*:}"
    echo "===== GPU ${gpu} | module | ${experiment} | fold ${fold} ====="
    CUDA_VISIBLE_DEVICES="$gpu" PYTHON_BIN="$PYTHON_BIN" \
      bash "$SCRIPT_DIR/$experiment/run_busi_5fold.sh" "$fold"
  done
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  [[ -n "${worker0_pid:-}" ]] && kill "$worker0_pid" 2>/dev/null || true
  [[ -n "${worker1_pid:-}" ]] && kill "$worker1_pid" 2>/dev/null || true
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$PID_FILE"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "Launching ${#TASKS[@]} module-ablation tasks concurrently on GPUs ${GPU0} and ${GPU1}."
run_worker "$GPU0" 0 & worker0_pid=$!
run_worker "$GPU1" 1 & worker1_pid=$!
wait "$worker0_pid"
wait "$worker1_pid"
echo "All requested BUSI_5fold module ablations completed."

#!/usr/bin/env bash
# 5_Edge_EdgeAttention_classification_Segmentation
# TEST ONLY for BUSI 5-fold
#
# Usage:
#   PYTHON_BIN=/path/to/python bash run_busi_5fold_test_only.sh [all|1|2|3|4|5]

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/BUSI_5fold}"

# 已训练好的权重目录
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-256}"

FOLD_SELECTOR="${1:-all}"

# ============================================================
# Select folds
# ============================================================

case "$FOLD_SELECTOR" in
  all)
    folds=(1 2 3 4 5)
    ;;
  1|2|3|4|5)
    folds=("$FOLD_SELECTOR")
    ;;
  *)
    echo "Usage: $0 [all|1|2|3|4|5]" >&2
    exit 2
    ;;
esac

# ============================================================
# Check environment
# ============================================================

"$PYTHON_BIN" -c 'import torch, timm, matplotlib, cv2, PIL, torchvision' >/dev/null 2>&1 || {
  echo "Set PYTHON_BIN to the training/test environment." >&2
  exit 1
}

cd "$PROJECT_DIR"

echo "============================================================"
echo "TEST ONLY"
echo "Experiment : 5_Edge_EdgeAttention_classification_Segmentation"
echo "Dataset    : $DATASET_ROOT"
echo "Results    : $RESULTS_ROOT"
echo "Folds      : ${folds[*]}"
echo "============================================================"

# ============================================================
# Test each fold
# ============================================================

for fold in "${folds[@]}"; do

  data_root="$DATASET_ROOT/fold_${fold}"
  output_dir="$RESULTS_ROOT/fold_${fold}"
  weight_path="$output_dir/best_model_all_case.pth"

  echo
  echo "============================================================"
  echo "Testing Fold $fold"
  echo "Data   : $data_root"
  echo "Weight : $weight_path"
  echo "============================================================"

  # 检查数据目录
  if [[ ! -f "$data_root/labels.csv" ]]; then
    echo "[ERROR] Missing fold data:"
    echo "        $data_root/labels.csv"
    exit 1
  fi

  # 检查训练好的权重
  if [[ ! -f "$weight_path" ]]; then
    echo "[ERROR] Missing trained weight:"
    echo "        $weight_path"
    exit 1
  fi

  # 只运行 test
  "$PYTHON_BIN" "$SCRIPT_DIR/test.py" \
    --weight-path "$weight_path" \
    --data-root "$data_root" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --select-metric dice_mean \
    --use-checkpoint-threshold 1

done

echo
echo "============================================================"
echo "Completed TEST ONLY"
echo "Experiment : 5_Edge_EdgeAttention_classification_Segmentation"
echo "Results    : $RESULTS_ROOT"
echo "============================================================"
#!/usr/bin/env bash
# module_ablation_2_Base_HFD: test only for BUSI 5 folds.
# Usage:
#   PYTHON_BIN=/path/to/python bash run_busi_5fold_test_only.sh [all|1|2|3|4|5]

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/BUSI_5fold1004}"
# DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/BUSI_5fold}"

# 训练好的权重目录
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/0results}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-256}"

FOLD_SELECTOR="${1:-all}"

# ============================================================
# Check Python environment
# ============================================================

if ! "$PYTHON_BIN" -c 'import torch, timm, matplotlib, cv2, PIL, torchvision' >/dev/null 2>&1; then
  echo "Set PYTHON_BIN to the Python environment with torch, timm, matplotlib, cv2, PIL, and torchvision." >&2
  exit 1
fi

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

cd "$PROJECT_DIR"

echo "============================================================"
echo "TEST ONLY: module_ablation_2_Base_HFD"
echo "Dataset root : $DATASET_ROOT"
echo "Results root : $RESULTS_ROOT"
echo "Folds        : ${folds[*]}"
echo "============================================================"

# ============================================================
# Test each fold
# ============================================================

for fold in "${folds[@]}"; do

  data_root="$DATASET_ROOT/fold_${fold}"
  output_dir="$RESULTS_ROOT/fold_${fold}"
  weight_path="$output_dir/best_model_all_case.pth"
  # weight_path="$output_dir/best_model_lesion_only.pth"


  echo
  echo "============================================================"
  echo "Testing Fold $fold"
  echo "Data   : $data_root"
  echo "Weight : $weight_path"
  echo "============================================================"

  # Check dataset
  if [[ ! -f "$data_root/labels.csv" ]]; then
    echo "[ERROR] Missing fold data:"
    echo "        $data_root/labels.csv"
    exit 1
  fi

  # Check trained weight
  if [[ ! -f "$weight_path" ]]; then
    echo "[ERROR] Missing trained weight:"
    echo "        $weight_path"
    exit 1
  fi

  # Test only
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
echo "Completed TEST ONLY: module_ablation_2_Base_HFD"
echo "Results root: $RESULTS_ROOT"
echo "============================================================"
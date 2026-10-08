#!/usr/bin/env bash
# module_ablation_5_Base_HFD_DetailFusion_MSAG: train and test all BUSI_5fold 5 folds.
# Usage: PYTHON_BIN=/path/to/python bash run_busi_5fold.sh [all|1|2|3|4|5]
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/BUSI_5fold1004}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-256}"
SEED_BASE="${SEED_BASE:-2025}"
FOLD_SELECTOR="${1:-all}"

if ! "$PYTHON_BIN" -c 'import torch, timm, matplotlib, cv2, PIL, torchvision' >/dev/null 2>&1; then
  echo "Set PYTHON_BIN to the Python environment with torch, timm, matplotlib, cv2, PIL, and torchvision." >&2
  exit 1
fi
case "$FOLD_SELECTOR" in
  all) folds=(1 2 3 4 5) ;;
  1|2|3|4|5) folds=("$FOLD_SELECTOR") ;;
  *) echo "Usage: $0 [all|1|2|3|4|5]" >&2; exit 2 ;;
esac

mkdir -p "$RESULTS_ROOT"
cd "$PROJECT_DIR"
for fold in "${folds[@]}"; do
  data_root="$DATASET_ROOT/fold_${fold}"
  output_dir="$RESULTS_ROOT/fold_${fold}"
  [[ -f "$data_root/labels.csv" ]] || { echo "Missing fold data: $data_root" >&2; exit 1; }
  "$PYTHON_BIN" "$SCRIPT_DIR/train.py" \
    --data-root "$data_root" --output-dir "$output_dir" \
    --experiment-name "module_ablation_5_Base_HFD_DetailFusion_MSAG_fold_${fold}" \
    --img-size "$IMG_SIZE" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" \
    --seed "$((SEED_BASE + fold))" --stem-channels 16 --export-visuals 0
  "$PYTHON_BIN" "$SCRIPT_DIR/test.py" \
    --weight-path "$output_dir/best_model_all_case.pth" \
    --data-root "$data_root" --img-size "$IMG_SIZE" --batch-size "$BATCH_SIZE" \
    --select-metric dice_mean --use-checkpoint-threshold 1
done
echo "Completed module_ablation_5_Base_HFD_DetailFusion_MSAG: $RESULTS_ROOT"

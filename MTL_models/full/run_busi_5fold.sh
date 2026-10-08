#!/usr/bin/env bash
# Train and evaluate HighFreqShallow V8 ResNet-18 on BUSI 5-fold splits.
# Usage:
#   PYTHON_BIN=/path/to/python bash run_busi_5fold.sh all
#   CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/path/to/python bash run_busi_5fold.sh 3

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_DIR/data/BUSI_5fold}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results/BUSI_5fold_HighFreqShallowV8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Override these without editing this file, e.g. EPOCHS=200 BATCH_SIZE=8 bash ...
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-256}"
SEED_BASE="${SEED_BASE:-2025}"
FOLD_SELECTOR="${1:-all}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory not found: $PROJECT_DIR" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset directory not found: $DATASET_ROOT" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import torch, timm, matplotlib, cv2, PIL, torchvision' >/dev/null 2>&1; then
  echo "PYTHON_BIN=$PYTHON_BIN is missing one or more required packages." >&2
  echo "Activate the training environment or pass PYTHON_BIN=/path/to/its/python." >&2
  exit 1
fi

case "$FOLD_SELECTOR" in
  all) folds=(1 2 3 4 5) ;;
  1|2|3|4|5) folds=("$FOLD_SELECTOR") ;;
  *)
    echo "Usage: $0 [all|1|2|3|4|5]" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULTS_ROOT"
cd "$PROJECT_DIR"

for fold in "${folds[@]}"; do
  data_root="$DATASET_ROOT/fold_${fold}"
  output_dir="$RESULTS_ROOT/fold_${fold}"
  seed=$((SEED_BASE + fold))

  for required_path in "$data_root/labels.csv" "$data_root/train/images" "$data_root/val/images" "$data_root/test/images"; do
    if [[ ! -e "$required_path" ]]; then
      echo "Fold ${fold} is incomplete; missing: $required_path" >&2
      exit 1
    fi
  done

  echo "===== Fold ${fold}: training HighFreqShallow V8 ResNet-18 ====="
  "$PYTHON_BIN" train.py \
    --data-root "$data_root" \
    --output-dir "$output_dir" \
    --experiment-name "BUSI_5fold_fold_${fold}_HighFreqShallowV8" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --seed "$seed" \
    --stem-channels 16 \
    --detail-strength 1.0 \
    --raw-branch-bias 0.50 \
    --fusion-strength 1.0 \
    --export-visuals 0

  checkpoint="$output_dir/best_model_all_case.pth"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Expected checkpoint was not produced: $checkpoint" >&2
    exit 1
  fi

  echo "===== Fold ${fold}: testing best all-case checkpoint ====="
  "$PYTHON_BIN" test.py \
    --weight-path "$checkpoint" \
    --data-root "$data_root" \
    --img-size "$IMG_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --select-metric dice_mean \
    --use-checkpoint-threshold 1
done

echo "Completed folds: ${folds[*]}"
echo "Results: $RESULTS_ROOT"

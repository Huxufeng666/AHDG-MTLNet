# AHDG_MTLNET_LAST

This project implements BUSI ultrasound image segmentation, edge prediction, and joint classification learning. It includes the proposed model, module-ablation experiments, and multi-task ablation experiments.

## Project Layout

```text
AHDG_MTLNET_LAST/
├── MTL_models/                         # Core implementation of the proposed model
│   ├── MTL_DualBranch_HighFreqShallowV8_Resnet18.py
│   ├── Propose_Module.py
│   └── full/                           # Standalone full-model training and testing entry points
├── 1_Module_Experiment/                # Architectural module ablation experiments
│   ├── 1_Base_model/
│   ├── 2_Base_HFD/
│   ├── 3_Base_HFD_DetailFusion/
│   ├── 4_Base_MSAG/
│   ├── 5_Base_HFD_DetailFusion_MSAG/
│   └── run_all_module_ablations.sh
├── 2_Multi_task_Experiment/            # Multi-task ablation experiments
│   ├── 1_Classification_Segmentation/
│   ├── 2_Edge_Segmentation/
│   ├── 3_Edge_Classification_Segmentation/
│   ├── 4_Edge_EdgeAttention_Segmentation/
│   ├── 5_Edge_EdgeAttention_classification_Segmentation/
│   └── run_all_multitask_ablations.sh
└── utils/                              # Data loading, losses, metrics, logging, saving, and plotting utilities
```

## Proposed Model

`MTL_models/MTL_DualBranch_HighFreqShallowV8_Resnet18.py` defines the primary dual-branch ResNet18-based network.

`MTL_models/Propose_Module.py` provides the reusable proposed components:

- **HFD**: high-frequency detail extraction;
- **Detail Fusion**: direct fusion and ADF residual detail fusion;
- **MSAG**: multi-scale attention gate;
- **EdgeAttention**: edge-aware attention;
- Classification-guided channel attention, ConvMixer blocks, attention gates, and decoder blocks.

The same backbone supports the complete model and all ablations through `use_hfd`, `fusion_mode`, `use_msag`, `use_edge_attention`, and `use_cls` switches.

## Paper-aligned defaults

The default five-fold launchers implement the manuscript configuration: a 16-channel input stem, a four-layer ConvMixer bottleneck, and ADF using direct raw/detail concatenation (`raw_branch_bias=0.50`) with a bounded residual coefficient of `fusion_strength=1.0`. EMA and validation-time flip TTA are disabled by default because they are not part of the reported experimental protocol. The corresponding command-line switches remain available only for explicit, separately reported extensions.

> **Reproducibility note:** Existing checkpoints trained before this alignment may use the previous ADF and evaluation settings. Retrain all five folds before using them to support manuscript tables.


## Dataset and Environment

The current training scripts use the following five-fold BUSI dataset root:

```text
/path/to/BUSI_5fold1004
```

Each fold directory must contain the images, labels, and `labels.csv` required by the data-loading code.

Use the existing training environment on the server:

```bash
export PYTHON_BIN=/path/to/python
```

This environment must provide `torch`, `timm`, `matplotlib`, `cv2`, `PIL`, and `torchvision`. Do not rely on the system `/usr/bin/python3`, which currently lacks required training dependencies.

## Module Ablation Experiments

`1_Module_Experiment` evaluates the contribution of HFD, Detail Fusion, and MSAG:

| Directory | Configuration |
| --- | --- |
| `1_Base_model` | Baseline segmentation model |
| `2_Base_HFD` | Baseline + HFD |
| `3_Base_HFD_DetailFusion` | Baseline + HFD + Detail Fusion |
| `4_Base_MSAG` | Baseline + MSAG |
| `5_Base_HFD_DetailFusion_MSAG` | Baseline + HFD + Detail Fusion + MSAG |

Each experiment directory contains:

- `train.py`: training, validation, threshold search, and checkpoint saving;
- `test.py`: evaluation of the selected checkpoint for one fold;
- `run_busi_5fold.sh`: sequential training and testing for folds 1--5;
- `run_busi_5fold_test_only.sh`: evaluation of existing checkpoints only;
- `results/fold_*`: checkpoints, training logs, and test outputs.

Run all module ablations with the two-GPU queue in the background:

```bash
cd /path/to/AHDG-MTLNet
PYTHON_BIN="$PYTHON_BIN" bash 1_Module_Experiment/run_all_module_ablations.sh start all
```

Run the five folds for a single module experiment:

```bash
PYTHON_BIN="$PYTHON_BIN" bash 1_Module_Experiment/5_Base_HFD_DetailFusion_MSAG/run_busi_5fold.sh all
```

## Multi-task Ablation Experiments

`2_Multi_task_Experiment` progressively adds classification, edge supervision, and edge attention to the complete backbone:

| Directory | Task configuration |
| --- | --- |
| `1_Classification_Segmentation` | Classification + segmentation |
| `2_Edge_Segmentation` | Edge supervision + segmentation |
| `3_Edge_Classification_Segmentation` | Edge supervision + classification + segmentation |
| `4_Edge_EdgeAttention_Segmentation` | Edge supervision + edge attention + segmentation |
| `5_Edge_EdgeAttention_classification_Segmentation` | Edge supervision + edge attention + classification + segmentation (full multi-task model) |

Every directory includes `train.py`, `test.py`, five-fold training and test-only scripts, and a `results/fold_*` output directory.

Run all multi-task ablations with the two-GPU queue in the background:

```bash
cd /path/to/AHDG-MTLNet
PYTHON_BIN="$PYTHON_BIN" bash 2_Multi_task_Experiment/run_all_multitask_ablations.sh background all
```

Run the complete multi-task model only:

```bash
PYTHON_BIN="$PYTHON_BIN" bash 2_Multi_task_Experiment/5_Edge_EdgeAttention_classification_Segmentation/run_busi_5fold.sh all
```

## Testing Existing Checkpoints

Use the test-only script in the desired experiment directory. For example:

```bash
PYTHON_BIN="$PYTHON_BIN" bash 2_Multi_task_Experiment/5_Edge_EdgeAttention_classification_Segmentation/run_busi_5fold_test_only.sh all
```

Testing loads `results/fold_*/best_model_all_case.pth` and writes outputs such as `test_metrics_global.csv`.

> **Note:** The multi-task `run_busi_5fold_test_only.sh` scripts currently default to `BUSI_5fold`, while the training scripts use `BUSI_5fold1004`. Before evaluating an existing checkpoint, make sure the test-only script points to the same dataset version used during training.

## Common Outputs

- `train_log.csv`: per-epoch training and validation losses, Dice, IoU, Precision, and Recall;
- `best_model_all_case.pth`: checkpoint selected by all-case Dice;
- `best_model_lesion_only.pth`: checkpoint selected by lesion-only Dice;
- `test_metrics_global.csv`: summary metrics for the full test fold;
- `visualizations/`: intermediate prediction images when visualization export is enabled.

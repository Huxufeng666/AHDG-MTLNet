import argparse
import copy
import csv
import datetime
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.get_data import BUSI_Data
from utils.loss import dice_loss, boundary_loss
from MTL_models.MTL_DualBranch_HighFreqShallowV8_Resnet18 import DualBranchHighFreqShallowV8_Resnet18

# BUSI class_id convention: 0=normal, 1=benign, 2=malignant.
NORMAL_CLASS_ID = 0


ABLATION_MODEL_KWARGS = dict(use_hfd=True, use_detail_fusion=True, hfd_input_injection=False, fusion_mode='adf', use_msag=True)

MODEL_ARCH = 'module_ablation_5_Base_HFD_DetailFusion_MSAG'

ABLATION_SPECS = [
    ('full_model', 1, 1, 1, 1, 1),
    ('w_o_mixer', 0, 1, 1, 1, 1),
    ('w_o_msag', 1, 0, 1, 1, 1),
    ('w_o_att_gate', 1, 1, 0, 1, 1),
    ('w_o_edge_attention', 1, 1, 1, 0, 1),
    ('w_o_cls', 1, 1, 1, 1, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Train train_G variant with all-case and lesion-only validation metrics.')
    parser.add_argument('--data-root', required=True, help='Path to the dataset root.')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--augment-profile', default='fixed', choices=['current', 'fixed', 'simplified', 'none'])
    parser.add_argument('--stem-channels', type=int, default=1)
    parser.add_argument('--denoise-strength', type=float, default=0.0)
    parser.add_argument('--raw-branch-bias', type=float, default=0.65)
    parser.add_argument('--fusion-strength', type=float, default=0.35)
    parser.add_argument('--use-spatial-fusion', type=int, default=1)
    parser.add_argument('--detail-strength', type=float, default=1.0)
    parser.add_argument('--fusion-mode', choices=['none', 'direct_add', 'concat_conv', 'adf'], default='adf')
    parser.add_argument('--deep-supervision-weight', type=float, default=0.4)
    parser.add_argument('--edge-loss-weight', type=float, default=0.04)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--val-flip-tta', type=int, default=1)
    parser.add_argument('--threshold-start', type=float, default=0.35)
    parser.add_argument('--threshold-end', type=float, default=0.56)
    parser.add_argument('--threshold-step', type=float, default=0.01)
    parser.add_argument('--device', default=None)
    parser.add_argument('--use-mixer', type=int, default=1)
    parser.add_argument('--use-msag', type=int, default=1)
    parser.add_argument('--use-att-gate', type=int, default=1)
    parser.add_argument('--use-edge-attention', type=int, default=0, help='Disabled: this ablation is segmentation-only.')
    parser.add_argument('--use-cls', type=int, default=0, help='Disabled: this ablation is segmentation-only.')
    parser.add_argument('--cls-loss-weight', type=float, default=0.05)
    parser.add_argument('--cls-normal-weight', type=float, default=1.0)
    parser.add_argument('--detach-cls-gate', type=int, default=0)
    parser.add_argument('--use-soft-cls-gate', type=int, default=0, help='Disabled: this ablation is segmentation-only.')
    parser.add_argument('--cls-gate-strength', type=float, default=0.75)
    parser.add_argument('--normal-penalty-weight', type=float, default=0.20)
    parser.add_argument('--presence-loss-weight', type=float, default=0.05)
    parser.add_argument('--presence-normal-weight', type=float, default=2.0)
    parser.add_argument('--normal-topk-weight', type=float, default=0.10)
    parser.add_argument('--normal-topk-ratio', type=float, default=0.01)
    parser.add_argument('--lesion-balanced-sampler', type=int, default=0)
    parser.add_argument('--lesion-crop-prob', type=float, default=0.0)
    parser.add_argument('--lesion-crop-margin', type=float, default=0.35)
    parser.add_argument('--size-aware-loss', type=int, default=0)
    parser.add_argument('--small-lesion-ratio', type=float, default=0.02)
    parser.add_argument('--medium-lesion-ratio', type=float, default=0.10)
    parser.add_argument('--small-lesion-loss-weight', type=float, default=1.8)
    parser.add_argument('--medium-lesion-loss-weight', type=float, default=1.3)
    parser.add_argument('--hard-sample-weight', type=float, default=2.0)
    parser.add_argument('--hard-normal-weight', type=float, default=1.8)
    parser.add_argument('--hard-lesion-dice-threshold', type=float, default=0.50)
    parser.add_argument('--hard-normal-prob-threshold', type=float, default=0.02)
    parser.add_argument('--hard-weight-momentum', type=float, default=0.7)
    parser.add_argument('--experiment-name', default='full_model')
    parser.add_argument('--export-visuals', type=int, default=1)
    parser.add_argument('--visualize-split', default='val', choices=['train', 'val'])
    parser.add_argument('--max-visual-samples', type=int, default=32)
    parser.add_argument('--list-ablations', action='store_true')
    parser.add_argument('--generate-ablation-sh', default=None, help='Write a shell script that runs the standard G2 ablation suite.')
    parser.add_argument('--ablation-results-root', default='results/train_G2_ablation')
    parser.add_argument('--ablation-data-root', default=None)
    parser.add_argument('--ablation-gpus', nargs='*', default=None)
    parser.add_argument('--ablation-epochs', type=int, default=None)
    parser.add_argument('--ablation-batch-size', type=int, default=None)
    return parser.parse_args()


def build_ablation_command(script_path, args, spec):
    name, use_mixer, use_msag, use_att_gate, use_edge_attention, use_cls = spec
    data_root = args.ablation_data_root or args.data_root
    out_dir = Path(args.ablation_results_root) / name
    cmd = [
        'python', script_path,
        '--data-root', str(data_root),
        '--output-dir', str(out_dir),
        '--img-size', str(args.img_size),
        '--batch-size', str(args.ablation_batch_size or args.batch_size),
        '--epochs', str(args.ablation_epochs or args.epochs),
        '--patience', str(args.patience),
        '--seed', str(args.seed),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--augment-profile', args.augment_profile,
        '--stem-channels', str(args.stem_channels),
        '--denoise-strength', str(args.denoise_strength),
        '--raw-branch-bias', str(args.raw_branch_bias),
        '--fusion-strength', str(args.fusion_strength),
        '--threshold-start', str(args.threshold_start),
        '--threshold-end', str(args.threshold_end),
        '--threshold-step', str(args.threshold_step),
        '--experiment-name', name,
        '--use-mixer', str(use_mixer),
        '--use-msag', str(use_msag),
        '--use-att-gate', str(use_att_gate),
        '--use-edge-attention', str(use_edge_attention),
        '--use-cls', str(use_cls),
        '--export-visuals', str(args.export_visuals),
        '--visualize-split', args.visualize_split,
        '--max-visual-samples', str(args.max_visual_samples),
    ]
    return cmd


def write_ablation_shell_script(script_path, args):
    output_path = Path(args.generate_ablation_sh)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        '',
        f'# Generated by {Path(__file__).name}',
        f'RESULTS_ROOT="{args.ablation_results_root}"',
        f'DATA_ROOT="{args.ablation_data_root or args.data_root}"',
        '',
    ]
    if args.ablation_gpus:
        gpu_count = len(args.ablation_gpus)
        lines.extend([
            f'GPUS=({" ".join(args.ablation_gpus)})',
            f'MAX_JOBS={gpu_count}',
            '',
            'launch_job() {',
            '  local gpu="$1"',
            '  shift',
            '  CUDA_VISIBLE_DEVICES="$gpu" "$@" &',
            '  if [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; then',
            '    wait -n',
            '  fi',
            '}',
            '',
        ])
        for idx, spec in enumerate(ABLATION_SPECS):
            cmd = build_ablation_command(script_path, args, spec)
            line = ' '.join(f'"{part}"' if ' ' in part else part for part in cmd)
            gpu_expr = f'${{GPUS[{idx % gpu_count}]}}'
            lines.append(f'launch_job {gpu_expr} {line}')
        lines.extend(['', 'wait'])
    else:
        for spec in ABLATION_SPECS:
            cmd = build_ablation_command(script_path, args, spec)
            line = ' '.join(f'"{part}"' if ' ' in part else part for part in cmd)
            lines.append(line)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    output_path.chmod(0o755)
    print(f'Generated ablation shell script: {output_path}')


def seed_everything(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker_factory(base_seed):
    def seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            current = source[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(current, alpha=1.0 - decay)
            else:
                value.copy_(current)


def build_model_from_args(args):
    """Build the project's only supported architecture: HighFreqShallow V8 ResNet-18."""
    return DualBranchHighFreqShallowV8_Resnet18(
        img_ch=1, output_ch=1, cls_classes=3, use_cls=False,
        mixer_depth=4, mixer_kernel=7, edge_alpha=0.08,
        use_mixer=bool(args.use_mixer),
        use_att_gate=bool(args.use_att_gate), use_edge_attention=False,
        stem_channels=int(args.stem_channels), detail_strength=float(args.detail_strength),
        raw_branch_bias=float(args.raw_branch_bias), fusion_strength=float(args.fusion_strength),
        **ABLATION_MODEL_KWARGS,
    )

def ensure_mask_channel(tensor):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    tensor = tensor.float()
    return tensor / 255.0 if tensor.max() > 1 else tensor


def tversky_loss(logits, targets, alpha=0.25, beta=0.82, smooth=1e-6):
    probs = torch.sigmoid(logits)
    targets = ensure_mask_channel(targets)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = ensure_mask_channel(targets)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def probs_to_logits(probs, eps=1e-4):
    probs = probs.clamp(eps, 1.0 - eps)
    return torch.logit(probs)


def apply_soft_cls_gate(seg_out, cls_out, args):
    seg_prob = torch.sigmoid(seg_out)
    gate = torch.ones(seg_prob.size(0), 1, 1, 1, device=seg_prob.device, dtype=seg_prob.dtype)
    if bool(args.use_soft_cls_gate) and cls_out is not None:
        cls_prob = torch.softmax(cls_out, dim=1)
        lesion_prob = 1.0 - cls_prob[:, NORMAL_CLASS_ID]
        if bool(args.detach_cls_gate) or is_presence_v7(args):
            lesion_prob = lesion_prob.detach()
        gate = (1.0 - float(args.cls_gate_strength)) + float(args.cls_gate_strength) * lesion_prob
        gate = gate.view(-1, 1, 1, 1)
        seg_prob = seg_prob * gate
    return probs_to_logits(seg_prob), seg_prob, gate


def compute_bce_per_sample(logits, targets):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return loss.reshape(loss.size(0), -1).mean(dim=1)


def compute_focal_per_sample(logits, targets, alpha=0.35, gamma=2.5):
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.reshape(loss.size(0), -1).mean(dim=1)


def compute_dice_per_sample(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1.0 - dice


def compute_tversky_per_sample(logits, targets, alpha=0.2, beta=0.82, smooth=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - score


def compute_boundary_per_sample(logits, targets):
    pred = torch.sigmoid(logits)
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_dx = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_dy = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
    loss_x = torch.abs(pred_dx - target_dx).reshape(pred.size(0), -1).mean(dim=1)
    loss_y = torch.abs(pred_dy - target_dy).reshape(pred.size(0), -1).mean(dim=1)
    return loss_x + loss_y


def compute_seg_loss(seg_out, masks, sample_weights=None, args=None):
    masks = ensure_mask_channel(masks)
    if args is not None and bool(args.size_aware_loss):
        # Give overlap and contour terms more influence for small BUSI lesions.
        per_sample = (
            0.20 * compute_bce_per_sample(seg_out, masks)
            + 0.40 * compute_dice_per_sample(seg_out, masks)
            + 0.25 * compute_tversky_per_sample(seg_out, masks, alpha=0.25, beta=0.75)
            + 0.15 * compute_boundary_per_sample(seg_out, masks)
        )
        area_ratio = masks.reshape(masks.size(0), -1).mean(dim=1)
        size_weights = torch.ones_like(area_ratio)
        lesion = area_ratio > 0
        size_weights = torch.where(
            lesion & (area_ratio <= float(args.small_lesion_ratio)),
            torch.full_like(size_weights, float(args.small_lesion_loss_weight)),
            size_weights,
        )
        size_weights = torch.where(
            lesion & (area_ratio > float(args.small_lesion_ratio))
            & (area_ratio <= float(args.medium_lesion_ratio)),
            torch.full_like(size_weights, float(args.medium_lesion_loss_weight)),
            size_weights,
        )
        per_sample = per_sample * (size_weights / size_weights.mean().clamp_min(1e-6))
    else:
        per_sample = (
            0.25 * compute_bce_per_sample(seg_out, masks)
            + 0.35 * compute_dice_per_sample(seg_out, masks)
            + 0.20 * compute_focal_per_sample(seg_out, masks)
            + 0.15 * compute_tversky_per_sample(seg_out, masks, alpha=0.2, beta=0.82)
            + 0.05 * compute_boundary_per_sample(seg_out, masks)
        )
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    return per_sample.mean(), per_sample.detach()


def build_lesion_balanced_sampler(dataset, args, generator):
    """Balance normal/small/medium/large groups using inverse-frequency weights."""
    ratios = np.asarray(dataset.get_mask_area_ratios(), dtype=np.float64)
    groups = np.full(len(ratios), 3, dtype=np.int64)  # large lesion
    groups[ratios <= float(args.medium_lesion_ratio)] = 2
    groups[ratios <= float(args.small_lesion_ratio)] = 1
    groups[ratios <= 0] = 0  # normal/empty mask
    counts = np.bincount(groups, minlength=4)
    nonempty_counts = counts[counts > 0]
    target_count = float(nonempty_counts.mean()) if len(nonempty_counts) else 1.0
    group_weights = np.ones(4, dtype=np.float64)
    for group_id, count in enumerate(counts):
        if count > 0:
            group_weights[group_id] = np.clip(target_count / float(count), 0.5, 3.0)
    sample_weights = torch.as_tensor(group_weights[groups], dtype=torch.double)
    print(
        'Lesion-size sampler | '
        f'normal={counts[0]} small={counts[1]} medium={counts[2]} large={counts[3]} | '
        f'weights={group_weights.round(3).tolist()}'
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def build_sample_weights(names, sample_weight_map, device):
    weights = [float(sample_weight_map.get(name, 1.0)) for name in names]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_normal_penalty(seg_prob, cls_gt, sample_weights=None):
    normal_mask = (cls_gt == NORMAL_CLASS_ID)
    if normal_mask.sum() == 0:
        zero = seg_prob.new_tensor(0.0)
        return zero, seg_prob.new_zeros(seg_prob.size(0))
    per_sample = seg_prob.reshape(seg_prob.size(0), -1).mean(dim=1)
    per_sample = per_sample * normal_mask.float()
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    penalty = per_sample.sum() / normal_mask.float().sum().clamp_min(1.0)
    return penalty, per_sample.detach()


def is_presence_v7(args):
    return False


def classification_targets(cls_gt, args):
    """Map normal/benign/malignant to normal/lesion for the V7 head."""
    if is_presence_v7(args):
        return (cls_gt != NORMAL_CLASS_ID).long()
    return cls_gt


def compute_normal_topk_penalty(seg_prob, cls_gt, topk_ratio=0.01, sample_weights=None):
    """Penalize concentrated false-positive responses on normal images."""
    normal_mask = cls_gt == NORMAL_CLASS_ID
    per_sample = seg_prob.new_zeros(seg_prob.size(0))
    if not bool(normal_mask.any()):
        return seg_prob.new_tensor(0.0), per_sample

    flat = seg_prob[normal_mask].reshape(int(normal_mask.sum().item()), -1)
    ratio = min(max(float(topk_ratio), 0.0), 1.0)
    k = min(flat.size(1), max(1, int(math.ceil(flat.size(1) * ratio))))
    normal_values = torch.topk(flat, k=k, dim=1, largest=True).values.mean(dim=1)
    if sample_weights is not None:
        normal_values = normal_values * sample_weights[normal_mask]
    per_sample[normal_mask] = normal_values.detach()
    return normal_values.mean(), per_sample


def update_hard_sample_weights(sample_weight_map, epoch_stats, args):
    momentum = float(args.hard_weight_momentum)
    for name, stats in epoch_stats.items():
        current = float(sample_weight_map.get(name, 1.0))
        target = 1.0
        if stats['cls_id'] == NORMAL_CLASS_ID:
            if stats['normal_score'] > float(args.hard_normal_prob_threshold):
                target = float(args.hard_normal_weight)
        else:
            if stats['dice_score'] < float(args.hard_lesion_dice_threshold):
                target = float(args.hard_sample_weight)
        sample_weight_map[name] = momentum * current + (1.0 - momentum) * target


def compute_metrics_per_image(pred_bin, gt_bin, smooth=1e-6):
    p = pred_bin.reshape(-1).float()
    g = gt_bin.reshape(-1).float()
    tp = (p * g).sum()
    fp = (p * (1.0 - g)).sum()
    fn = ((1.0 - p) * g).sum()
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return float(dice.item()), float(iou.item()), float(precision.item()), float(recall.item())


def unpack_outputs(outputs):
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            return outputs[0], outputs[1], outputs[2]
        if len(outputs) == 2:
            return outputs[0], outputs[1], None
    raise ValueError('Unexpected model output.')


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def evaluate_with_threshold(model, loader, device, threshold, edge_bce_fn, ce_fn, args):
    model.eval()
    dice_all, iou_all, prec_all, rec_all = [], [], [], []
    dice_lesion, iou_lesion, prec_lesion, rec_lesion = [], [], [], []
    val_seg_loss = val_edge_loss = val_cls_loss = 0.0

    for imgs, masks, cls_gt, edge_gt, *_ in loader:
        imgs = imgs.to(device)
        masks = ensure_mask_channel(masks.to(device))
        edge_gt = ensure_mask_channel(edge_gt.to(device))
        cls_gt = cls_gt.to(device).long()

        seg_out, edge_out, cls_out = unpack_outputs(model(imgs))
        seg_out = F.interpolate(seg_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        edge_out = F.interpolate(edge_out, size=edge_gt.shape[-2:], mode='bilinear', align_corners=False)

        if bool(args.val_flip_tta):
            flipped_imgs = torch.flip(imgs, dims=[-1])
            flipped_seg_out, flipped_edge_out, flipped_cls_out = unpack_outputs(model(flipped_imgs))
            flipped_seg_out = F.interpolate(flipped_seg_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            flipped_edge_out = F.interpolate(flipped_edge_out, size=edge_gt.shape[-2:], mode='bilinear', align_corners=False)
            flipped_seg_prob = torch.flip(torch.sigmoid(flipped_seg_out), dims=[-1])
            seg_out = probs_to_logits(0.5 * (torch.sigmoid(seg_out) + flipped_seg_prob))
            edge_out = 0.5 * (edge_out + torch.flip(flipped_edge_out, dims=[-1]))
            if cls_out is not None and flipped_cls_out is not None:
                cls_out = 0.5 * (cls_out + flipped_cls_out)

        gated_seg_out, gated_seg_prob, _ = apply_soft_cls_gate(seg_out, cls_out, args)
        loss_seg_out = seg_out if is_presence_v7(args) else gated_seg_out
        val_seg_loss += compute_seg_loss(loss_seg_out, masks, args=args)[0].item()
        # Segmentation-only ablation: edge and classification losses are disabled.
        val_edge_loss += 0.0
        val_cls_loss += 0.0

        pred = (gated_seg_prob > threshold).float()
        for i in range(pred.size(0)):
            d, j, p, r = compute_metrics_per_image(pred[i], masks[i])
            dice_all.append(d)
            iou_all.append(j)
            prec_all.append(p)
            rec_all.append(r)
            if int(cls_gt[i].item()) != NORMAL_CLASS_ID:
                dice_lesion.append(d)
                iou_lesion.append(j)
                prec_lesion.append(p)
                rec_lesion.append(r)

    n_batches = max(len(loader), 1)
    stats = {
        'val_seg_loss': val_seg_loss / n_batches,
        'val_edge_loss': val_edge_loss / n_batches,
        'val_cls_loss': val_cls_loss / n_batches,
        'dice_mean': safe_mean(dice_all),
        'iou_mean': safe_mean(iou_all),
        'precision_mean': safe_mean(prec_all),
        'recall_mean': safe_mean(rec_all),
        'lesion_dice_mean': safe_mean(dice_lesion),
        'lesion_iou_mean': safe_mean(iou_lesion),
        'lesion_precision_mean': safe_mean(prec_lesion),
        'lesion_recall_mean': safe_mean(rec_lesion),
    }
    return stats


@torch.no_grad()
def sweep_threshold(model, loader, device, args, edge_bce_fn, ce_fn, optimize_metric):
    thresholds = np.arange(args.threshold_start, args.threshold_end + 1e-9, args.threshold_step)
    best = None
    for th in thresholds:
        stats = evaluate_with_threshold(model, loader, device, float(th), edge_bce_fn, ce_fn, args)
        candidate = {'threshold': float(th), 'optimize_metric': optimize_metric, **stats}
        if best is None or candidate[optimize_metric] > best[optimize_metric]:
            best = candidate
    return best


def save_history(rows, out_csv):
    fields = [
        'experiment_name', 'epoch', 'lr', 'train_loss', 'train_seg_loss', 'train_edge_loss', 'train_cls_loss',
        'val_loss',
        'val_threshold_all_case', 'val_threshold_lesion_only',
        'val_loss_all_case', 'val_loss_lesion_only',
        'val_seg_loss_all_case', 'val_edge_loss_all_case', 'val_cls_loss_all_case',
        'val_seg_loss_lesion_only', 'val_edge_loss_lesion_only', 'val_cls_loss_lesion_only',
        'dice_mean', 'iou_mean', 'precision_mean', 'recall_mean',
        'lesion_dice_mean', 'lesion_iou_mean', 'lesion_precision_mean', 'lesion_recall_mean',
    ]
    for row in rows:
        row['experiment_name'] = row.get('experiment_name', '')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows, out_png):
    if not rows:
        return
    epochs = [r['epoch'] for r in rows]
    plt.figure(figsize=(16, 12))

    plt.subplot(3, 2, 1)
    plt.plot(epochs, [r['train_loss'] for r in rows], label='train_loss')
    plt.plot(epochs, [r['val_loss_all_case'] for r in rows], label='val_loss_all_case')
    plt.plot(epochs, [r['val_loss_lesion_only'] for r in rows], label='val_loss_lesion_only')
    plt.legend(); plt.title('Total Loss')

    plt.subplot(3, 2, 2)
    plt.plot(epochs, [r['train_seg_loss'] for r in rows], label='train_seg_loss')
    plt.plot(epochs, [r['val_seg_loss_all_case'] for r in rows], label='val_seg_loss_all_case')
    plt.plot(epochs, [r['val_seg_loss_lesion_only'] for r in rows], label='val_seg_loss_lesion_only')
    plt.legend(); plt.title('Seg Loss')

    plt.subplot(3, 2, 3)
    plt.plot(epochs, [r['train_edge_loss'] for r in rows], label='train_edge_loss')
    plt.plot(epochs, [r['val_edge_loss_all_case'] for r in rows], label='val_edge_loss_all_case')
    plt.plot(epochs, [r['val_edge_loss_lesion_only'] for r in rows], label='val_edge_loss_lesion_only')
    plt.legend(); plt.title('Edge Loss')

    plt.subplot(3, 2, 4)
    plt.plot(epochs, [r['train_cls_loss'] for r in rows], label='train_cls_loss')
    plt.plot(epochs, [r['val_cls_loss_all_case'] for r in rows], label='val_cls_loss_all_case')
    plt.plot(epochs, [r['val_cls_loss_lesion_only'] for r in rows], label='val_cls_loss_lesion_only')
    plt.legend(); plt.title('Cls Loss')

    plt.subplot(3, 2, 5)
    plt.plot(epochs, [r['dice_mean'] for r in rows], label='dice_mean')
    plt.plot(epochs, [r['lesion_dice_mean'] for r in rows], label='lesion_dice_mean')
    plt.plot(epochs, [r['iou_mean'] for r in rows], label='iou_mean')
    plt.plot(epochs, [r['lesion_iou_mean'] for r in rows], label='lesion_iou_mean')
    plt.legend(); plt.title('Dice / IoU')

    plt.subplot(3, 2, 6)
    plt.plot(epochs, [r['precision_mean'] for r in rows], label='precision_mean')
    plt.plot(epochs, [r['recall_mean'] for r in rows], label='recall_mean')
    plt.plot(epochs, [r['lesion_precision_mean'] for r in rows], label='lesion_precision_mean')
    plt.plot(epochs, [r['lesion_recall_mean'] for r in rows], label='lesion_recall_mean')
    plt.legend(); plt.title('Precision / Recall')

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()




def save_best_checkpoint_summary(train_rows, best_all_epoch, best_lesion_epoch, output_dir):
    row_by_epoch = {row['epoch']: row for row in train_rows}
    summary_rows = []

    if best_all_epoch in row_by_epoch:
        row = row_by_epoch[best_all_epoch]
        summary_rows.append({
            'checkpoint_type': 'all_case',
            'best_epoch': best_all_epoch,
            'best_metric_name': 'dice_mean',
            'best_metric_value': row['dice_mean'],
            'best_threshold': row['val_threshold_all_case'],
            'train_loss': row['train_loss'],
            'train_seg_loss': row['train_seg_loss'],
            'train_edge_loss': row['train_edge_loss'],
            'train_cls_loss': row['train_cls_loss'],
            'val_loss': row['val_loss_all_case'],
            'val_seg_loss': row['val_seg_loss_all_case'],
            'val_edge_loss': row['val_edge_loss_all_case'],
            'val_cls_loss': row['val_cls_loss_all_case'],
            'dice_mean': row['dice_mean'],
            'iou_mean': row['iou_mean'],
            'precision_mean': row['precision_mean'],
            'recall_mean': row['recall_mean'],
            'lesion_dice_mean': row['lesion_dice_mean'],
            'lesion_iou_mean': row['lesion_iou_mean'],
            'lesion_precision_mean': row['lesion_precision_mean'],
            'lesion_recall_mean': row['lesion_recall_mean'],
        })

    if best_lesion_epoch in row_by_epoch:
        row = row_by_epoch[best_lesion_epoch]
        summary_rows.append({
            'checkpoint_type': 'lesion_only',
            'best_epoch': best_lesion_epoch,
            'best_metric_name': 'lesion_dice_mean',
            'best_metric_value': row['lesion_dice_mean'],
            'best_threshold': row['val_threshold_lesion_only'],
            'train_loss': row['train_loss'],
            'train_seg_loss': row['train_seg_loss'],
            'train_edge_loss': row['train_edge_loss'],
            'train_cls_loss': row['train_cls_loss'],
            'val_loss': row['val_loss_lesion_only'],
            'val_seg_loss': row['val_seg_loss_lesion_only'],
            'val_edge_loss': row['val_edge_loss_lesion_only'],
            'val_cls_loss': row['val_cls_loss_lesion_only'],
            'dice_mean': row['dice_mean'],
            'iou_mean': row['iou_mean'],
            'precision_mean': row['precision_mean'],
            'recall_mean': row['recall_mean'],
            'lesion_dice_mean': row['lesion_dice_mean'],
            'lesion_iou_mean': row['lesion_iou_mean'],
            'lesion_precision_mean': row['lesion_precision_mean'],
            'lesion_recall_mean': row['lesion_recall_mean'],
        })

    if not summary_rows:
        return

    out_csv = os.path.join(output_dir, 'best_checkpoint_summary.csv')
    for row in summary_rows:
        row['experiment_name'] = row.get('experiment_name', '')
    fields = list(summary_rows[0].keys())
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)



def save_sample_visualization(save_path, img, gt_mask, pred_mask, pred_prob, gt_edge, pred_edge_prob, gt_cls, pred_cls, metrics, experiment_name):
    img_np = img.squeeze().detach().cpu().numpy()
    gt_mask_np = gt_mask.squeeze().detach().cpu().numpy()
    pred_mask_np = pred_mask.squeeze().detach().cpu().numpy()
    pred_prob_np = pred_prob.squeeze().detach().cpu().numpy()
    gt_edge_np = gt_edge.squeeze().detach().cpu().numpy()
    pred_edge_np = pred_edge_prob.squeeze().detach().cpu().numpy()

    overlay = np.stack([img_np, img_np, img_np], axis=-1)
    overlay[..., 1] = np.clip(overlay[..., 1] + 0.45 * pred_mask_np, 0, 1)
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.45 * gt_mask_np, 0, 1)

    class_names = ['normal', 'benign', 'malignant']
    gt_name = class_names[gt_cls] if 0 <= gt_cls < len(class_names) else str(gt_cls)
    pred_name = class_names[pred_cls] if 0 <= pred_cls < len(class_names) else str(pred_cls)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(img_np, cmap='gray')
    axes[0, 0].set_title('Image')
    axes[0, 1].imshow(gt_mask_np, cmap='gray')
    axes[0, 1].set_title('GT Mask')
    axes[0, 2].imshow(pred_prob_np, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('Pred Mask Prob')
    axes[1, 0].imshow(pred_mask_np, cmap='gray')
    axes[1, 0].set_title('Pred Mask')
    axes[1, 1].imshow(gt_edge_np, cmap='gray')
    axes[1, 1].set_title('GT Edge')
    axes[1, 2].imshow(pred_edge_np, cmap='magma', vmin=0, vmax=1)
    axes[1, 2].set_title('Pred Edge Prob')

    for ax in axes.ravel():
        ax.axis('off')

    fig.suptitle(
        f"{experiment_name} | GT={gt_name} Pred={pred_name} | "
        f"Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    overlay_path = save_path.with_name(save_path.stem + '_overlay.png')
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=160, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def export_visualizations(model, loader, device, threshold, output_dir, experiment_name, args, max_samples=None):
    model.eval()
    vis_dir = Path(output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for imgs, masks, cls_gt, edge_gt, _, names in loader:
        imgs = imgs.to(device)
        masks = ensure_mask_channel(masks.to(device))
        edge_gt = ensure_mask_channel(edge_gt.to(device))
        cls_gt = cls_gt.to(device).long()

        seg_out, edge_out, cls_out = unpack_outputs(model(imgs))
        seg_out = F.interpolate(seg_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        edge_out = F.interpolate(edge_out, size=edge_gt.shape[-2:], mode='bilinear', align_corners=False)
        gated_seg_out, seg_prob, _ = apply_soft_cls_gate(seg_out, cls_out, args)
        edge_prob = torch.sigmoid(edge_out)
        pred_mask = (seg_prob > threshold).float()
        if cls_out is not None:
            cls_pred = torch.argmax(cls_out, dim=1)
        else:
            cls_pred = torch.full_like(cls_gt, -1)

        for i in range(pred_mask.size(0)):
            d, j, p, r = compute_metrics_per_image(pred_mask[i], masks[i])
            metrics = {'dice': d, 'iou': j, 'precision': p, 'recall': r}
            name = names[i] if isinstance(names, (list, tuple)) else str(names)
            save_sample_visualization(
                vis_dir / f'{name}.png',
                imgs[i].detach().cpu(),
                masks[i].detach().cpu(),
                pred_mask[i].detach().cpu(),
                seg_prob[i].detach().cpu(),
                edge_gt[i].detach().cpu(),
                edge_prob[i].detach().cpu(),
                int(cls_gt[i].detach().cpu().item()),
                int(cls_pred[i].detach().cpu().item()),
                metrics,
                experiment_name,
            )
            saved += 1
            if max_samples is not None and saved >= max_samples:
                return

def build_fixed_visualizer(val_loader, device, args):
    fixed_data = val_loader.dataset[min(45, len(val_loader.dataset) - 1)]
    fixed_imgs = fixed_data[0].unsqueeze(0).to(device)
    fixed_masks = fixed_data[1]
    if fixed_masks.dim() == 2:
        fixed_masks = fixed_masks.unsqueeze(0).unsqueeze(0)
    elif fixed_masks.dim() == 3:
        fixed_masks = fixed_masks.unsqueeze(0)
    fixed_masks = fixed_masks.float().to(device)
    fixed_edge = fixed_data[3].to(device)
    fixed_cls = int(fixed_data[2])

    def visualize(epoch, model, log_dir):
        model.eval()
        with torch.no_grad():
            seg_out, edge_out, cls_out = unpack_outputs(model(fixed_imgs))
            logits = F.interpolate(seg_out, size=fixed_masks.shape[-2:], mode='bilinear', align_corners=False)
            edge_logits = F.interpolate(edge_out, size=fixed_masks.shape[-2:], mode='bilinear', align_corners=False)
            gated_logits, gated_prob, _ = apply_soft_cls_gate(logits, cls_out, args)
            pred_mask = (gated_prob > 0.5).float()
            pred_edge = (torch.sigmoid(edge_logits) > 0.5).float()
            imgs_vis = fixed_imgs[:1].cpu().repeat(1, 3, 1, 1)
            masks_vis = fixed_masks[:1].cpu()
            pred_vis = pred_mask[:1].cpu()
            edge_gt_vis = fixed_edge[:1].float().cpu()
            if edge_gt_vis.dim() == 3:
                edge_gt_vis = edge_gt_vis.unsqueeze(1)
            edge_pred_vis = pred_edge[:1].cpu()

            def overlay(img, mask, color='green'):
                img = img.clone()
                if color == 'red':
                    img[:, 0] = torch.clamp(img[:, 0] + mask.squeeze(1) * 0.7, 0, 1)
                else:
                    img[:, 1] = torch.clamp(img[:, 1] + mask.squeeze(1) * 0.7, 0, 1)
                return img

            edge_gt_rgb = edge_gt_vis.repeat(1, 3, 1, 1)
            edge_pred_rgb = edge_pred_vis.repeat(1, 3, 1, 1)
            vis = torch.cat([imgs_vis, overlay(imgs_vis, masks_vis, 'red'), overlay(imgs_vis, pred_vis, 'green'), edge_gt_rgb, edge_pred_rgb], dim=3)
            save_path = Path(log_dir) / 'visualizations' / f'vis_epoch_{epoch:03d}.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vutils.save_image(vis, str(save_path), normalize=True)
            img_pil = Image.open(save_path)
            draw = ImageDraw.Draw(img_pil)
            class_names = ['normal', 'benign', 'malignant']
            cls_pred = int(torch.argmax(torch.softmax(cls_out, dim=1), dim=1)[0].item()) if cls_out is not None else -1
            draw.text((20, 20), f'Exp: {getattr(model, "experiment_name", "unknown")}', fill=(255, 255, 100))
            draw.text((20, 45), f'GT cls: {class_names[fixed_cls]} | Pred cls: {class_names[cls_pred]}', fill=(255, 255, 100))
            draw.text((20, 70), 'Panels: image | gt seg | pred seg | gt edge | pred edge', fill=(255, 255, 100))
            img_pil.save(save_path)
    return visualize


def main():
    args = parse_args()
    script_path = str(Path(__file__).resolve())

    if args.list_ablations:
        for spec in ABLATION_SPECS:
            print(spec[0])
        return

    if args.generate_ablation_sh:
        write_ablation_shell_script(script_path, args)
        return

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f'Using device: {device}')
    print(f'Model arch: {MODEL_ARCH}')
    print(f'Augment profile: {args.augment_profile}')

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'results/train_G2_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator().manual_seed(args.seed)
    seed_worker = seed_worker_factory(args.seed)

    train_data = BUSI_Data(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size,
        seed=args.seed,
        train=True,
        augment_profile=args.augment_profile,
        lesion_crop_prob=args.lesion_crop_prob,
        lesion_crop_margin=args.lesion_crop_margin,
    )
    val_data = BUSI_Data(
        root_dir=args.data_root,
        split='val',
        img_size=args.img_size,
        seed=args.seed,
        train=False,
        augment_profile='none',
    )

    train_sampler = build_lesion_balanced_sampler(train_data, args, generator) if bool(args.lesion_balanced_sampler) else None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=generator, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=generator, drop_last=False)
    vis_loader = val_loader if args.visualize_split == 'val' else DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=generator, drop_last=False)

    model = build_model_from_args(args).to(device)
    model.experiment_name = args.experiment_name
    model.model_arch = MODEL_ARCH

    if is_presence_v7(args):
        cls_weights = torch.tensor([float(args.presence_normal_weight), 1.0], dtype=torch.float32, device=device)
    else:
        cls_weights = torch.tensor([float(args.cls_normal_weight), 1.0, 1.0], dtype=torch.float32, device=device)
    ce_fn = nn.CrossEntropyLoss(weight=cls_weights)
    edge_bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0], device=device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEMA(model, args.ema_decay) if float(args.ema_decay) > 0 else None
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=5e-6)

    visualize_fixed = build_fixed_visualizer(val_loader, device, args)
    train_rows = []
    sample_weight_map = {}
    best_all = -1.0
    best_lesion = -1.0
    best_all_epoch = None
    best_lesion_epoch = None
    early_stop = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = train_seg_total = train_edge_total = train_cls_total = 0.0

        epoch_sample_stats = {}
        for imgs, masks, cls_gt, edge_gt, _, names in tqdm(train_loader, desc=f'Epoch {epoch}'):
            imgs = imgs.to(device)
            masks = ensure_mask_channel(masks.to(device))
            edge_gt = ensure_mask_channel(edge_gt.to(device))
            cls_gt = cls_gt.to(device).long()
            batch_names = [str(name) for name in names]
            sample_weights = build_sample_weights(batch_names, sample_weight_map, device)

            seg_out, edge_out, cls_out = unpack_outputs(model(imgs))
            seg_out = F.interpolate(seg_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            edge_out = F.interpolate(edge_out, size=edge_gt.shape[-2:], mode='bilinear', align_corners=False)
            gated_seg_out, gated_seg_prob, _ = apply_soft_cls_gate(seg_out, cls_out, args)

            loss_seg_out = seg_out if is_presence_v7(args) else gated_seg_out
            seg_loss, seg_loss_per_sample = compute_seg_loss(loss_seg_out, masks, sample_weights=sample_weights, args=args)
            aux_outputs = getattr(model, 'aux_outputs', ())
            if aux_outputs and float(args.deep_supervision_weight) > 0:
                aux_weights = (0.5, 0.3, 0.2)
                aux_loss = sum(
                    weight * compute_seg_loss(aux_out, masks, sample_weights=sample_weights, args=args)[0]
                    for weight, aux_out in zip(aux_weights, aux_outputs)
                )
                seg_loss = seg_loss + float(args.deep_supervision_weight) * aux_loss
            # Segmentation-only ablation: do not optimize edge or classification tasks.
            edge_loss = torch.zeros((), device=device)
            cls_loss = torch.zeros((), device=device)
            normal_penalty = torch.zeros((), device=device)
            normal_penalty_per_sample = torch.zeros_like(seg_loss_per_sample)
            loss = seg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            train_loss_total += loss.item()
            train_seg_total += seg_loss.item()
            train_edge_total += edge_loss.item()
            train_cls_total += cls_loss.item()

            pred_mask = (gated_seg_prob.detach() > 0.5).float()
            for i, name in enumerate(batch_names):
                dice_i, _, _, _ = compute_metrics_per_image(pred_mask[i], masks[i])
                normal_score = float(gated_seg_prob[i].detach().mean().item())
                epoch_sample_stats[name] = {
                    'cls_id': int(cls_gt[i].detach().cpu().item()),
                    'dice_score': float(dice_i),
                    'normal_score': normal_score,
                    'seg_loss': float(seg_loss_per_sample[i].detach().cpu().item()),
                    'normal_penalty': float(normal_penalty_per_sample[i].detach().cpu().item()),
                }

        update_hard_sample_weights(sample_weight_map, epoch_sample_stats, args)
        avg_train_loss = train_loss_total / max(len(train_loader), 1)
        avg_seg = train_seg_total / max(len(train_loader), 1)
        avg_edge = train_edge_total / max(len(train_loader), 1)
        avg_cls = train_cls_total / max(len(train_loader), 1)

        eval_model = ema.module if ema is not None else model
        val_best_all = sweep_threshold(eval_model, val_loader, device, args, edge_bce_fn, ce_fn, optimize_metric='dice_mean')
        val_best_lesion = sweep_threshold(eval_model, val_loader, device, args, edge_bce_fn, ce_fn, optimize_metric='lesion_dice_mean')
        avg_val_loss_all = val_best_all['val_seg_loss'] + 0.06 * val_best_all['val_edge_loss'] + 0.03 * val_best_all['val_cls_loss']
        avg_val_loss_lesion = val_best_lesion['val_seg_loss'] + 0.06 * val_best_lesion['val_edge_loss'] + 0.03 * val_best_lesion['val_cls_loss']
        lr = optimizer.param_groups[0]['lr']

        row = {
            'experiment_name': args.experiment_name,
            'epoch': epoch,
            'lr': lr,
            'train_loss': avg_train_loss,
            'train_seg_loss': avg_seg,
            'train_edge_loss': avg_edge,
            'train_cls_loss': avg_cls,
            'val_loss': avg_val_loss_all,
            'val_threshold_all_case': val_best_all['threshold'],
            'val_threshold_lesion_only': val_best_lesion['threshold'],
            'val_loss_all_case': avg_val_loss_all,
            'val_loss_lesion_only': avg_val_loss_lesion,
            'val_seg_loss_all_case': val_best_all['val_seg_loss'],
            'val_edge_loss_all_case': val_best_all['val_edge_loss'],
            'val_cls_loss_all_case': val_best_all['val_cls_loss'],
            'val_seg_loss_lesion_only': val_best_lesion['val_seg_loss'],
            'val_edge_loss_lesion_only': val_best_lesion['val_edge_loss'],
            'val_cls_loss_lesion_only': val_best_lesion['val_cls_loss'],
            'dice_mean': val_best_all['dice_mean'],
            'iou_mean': val_best_all['iou_mean'],
            'precision_mean': val_best_all['precision_mean'],
            'recall_mean': val_best_all['recall_mean'],
            'lesion_dice_mean': val_best_lesion['lesion_dice_mean'],
            'lesion_iou_mean': val_best_lesion['lesion_iou_mean'],
            'lesion_precision_mean': val_best_lesion['lesion_precision_mean'],
            'lesion_recall_mean': val_best_lesion['lesion_recall_mean'],
        }
        train_rows.append(row)
        save_history(train_rows, os.path.join(output_dir, 'train_log.csv'))
        plot_history(train_rows, os.path.join(output_dir, 'training_curve.png'))

        print(
            f"Epoch {epoch:3d} | "
            f"train_total={avg_train_loss:.4f} train_seg={avg_seg:.4f} train_edge={avg_edge:.4f} train_cls={avg_cls:.4f} | "
            f"val_all_total={avg_val_loss_all:.4f} val_all_seg={val_best_all['val_seg_loss']:.4f} val_all_edge={val_best_all['val_edge_loss']:.4f} val_all_cls={val_best_all['val_cls_loss']:.4f} | "
            f"val_lesion_total={avg_val_loss_lesion:.4f} val_lesion_seg={val_best_lesion['val_seg_loss']:.4f} val_lesion_edge={val_best_lesion['val_edge_loss']:.4f} val_lesion_cls={val_best_lesion['val_cls_loss']:.4f} | "
            f"dice={row['dice_mean']:.4f}@{row['val_threshold_all_case']:.3f} lesion_dice={row['lesion_dice_mean']:.4f}@{row['val_threshold_lesion_only']:.3f} | "
            f"best_all={best_all:.4f} best_lesion={best_lesion:.4f}"
        )

        if epoch % 10 == 0 or row['dice_mean'] > 0.81 or row['lesion_dice_mean'] > 0.81:
            visualize_fixed(epoch, model, output_dir)

        improved = False
        if row['dice_mean'] > best_all:
            best_all = row['dice_mean']
            torch.save({
                'model_state_dict': eval_model.state_dict(),
                'experiment_name': args.experiment_name,
            'epoch': epoch,
                'best_metric_name': 'dice_mean',
                'best_metric_value': best_all,
                'best_threshold': row['val_threshold_all_case'],
                'selection_mode': 'all_case',
                'selection_threshold_metric': 'dice_mean',
                'experiment_name': args.experiment_name,
                'model_name': MODEL_ARCH,
                'model_arch': MODEL_ARCH,
                'use_cls': False,
                'cls_loss_weight': float(args.cls_loss_weight),
                'cls_normal_weight': float(args.cls_normal_weight),
                'detach_cls_gate': bool(args.detach_cls_gate),
                'use_soft_cls_gate': False,
                'cls_gate_strength': float(args.cls_gate_strength),
                'normal_penalty_weight': float(args.normal_penalty_weight),
                'presence_loss_weight': float(args.presence_loss_weight),
                'presence_normal_weight': float(args.presence_normal_weight),
                'normal_topk_weight': float(args.normal_topk_weight),
                'normal_topk_ratio': float(args.normal_topk_ratio),
                'classification_mode': 'binary_presence' if is_presence_v7(args) else 'three_class',
                'use_mixer': bool(args.use_mixer),
                'use_msag': bool(args.use_msag),
                'use_att_gate': bool(args.use_att_gate),
                'use_edge_attention': False,
                'stem_channels': int(args.stem_channels),
                'denoise_strength': float(args.denoise_strength),
                'raw_branch_bias': float(args.raw_branch_bias),
                'fusion_strength': float(args.fusion_strength),
                'fusion_mode': args.fusion_mode,
                'use_spatial_fusion': bool(args.use_spatial_fusion),
                'detail_strength': float(args.detail_strength),
                'deep_supervision_weight': float(args.deep_supervision_weight),
                'edge_loss_weight': 0.0,
                'ema_decay': float(args.ema_decay),
                'checkpoint_is_ema': bool(ema is not None),
                'val_flip_tta': bool(args.val_flip_tta),
            }, os.path.join(output_dir, 'best_model_all_case.pth'))
            best_all_epoch = epoch
            improved = True

        if row['lesion_dice_mean'] > best_lesion:
            best_lesion = row['lesion_dice_mean']
            torch.save({
                'model_state_dict': eval_model.state_dict(),
                'experiment_name': args.experiment_name,
            'epoch': epoch,
                'best_metric_name': 'lesion_dice_mean',
                'best_metric_value': best_lesion,
                'best_threshold': row['val_threshold_lesion_only'],
                'selection_mode': 'lesion_only',
                'selection_threshold_metric': 'lesion_dice_mean',
                'experiment_name': args.experiment_name,
                'model_name': MODEL_ARCH,
                'model_arch': MODEL_ARCH,
                'use_cls': False,
                'cls_loss_weight': float(args.cls_loss_weight),
                'cls_normal_weight': float(args.cls_normal_weight),
                'detach_cls_gate': bool(args.detach_cls_gate),
                'use_soft_cls_gate': False,
                'cls_gate_strength': float(args.cls_gate_strength),
                'normal_penalty_weight': float(args.normal_penalty_weight),
                'presence_loss_weight': float(args.presence_loss_weight),
                'presence_normal_weight': float(args.presence_normal_weight),
                'normal_topk_weight': float(args.normal_topk_weight),
                'normal_topk_ratio': float(args.normal_topk_ratio),
                'classification_mode': 'binary_presence' if is_presence_v7(args) else 'three_class',
                'use_mixer': bool(args.use_mixer),
                'use_msag': bool(args.use_msag),
                'use_att_gate': bool(args.use_att_gate),
                'use_edge_attention': False,
                'stem_channels': int(args.stem_channels),
                'denoise_strength': float(args.denoise_strength),
                'raw_branch_bias': float(args.raw_branch_bias),
                'fusion_strength': float(args.fusion_strength),
                'fusion_mode': args.fusion_mode,
                'use_spatial_fusion': bool(args.use_spatial_fusion),
                'detail_strength': float(args.detail_strength),
                'deep_supervision_weight': float(args.deep_supervision_weight),
                'edge_loss_weight': 0.0,
                'ema_decay': float(args.ema_decay),
                'checkpoint_is_ema': bool(ema is not None),
                'val_flip_tta': bool(args.val_flip_tta),
            }, os.path.join(output_dir, 'best_model_lesion_only.pth'))
            best_lesion_epoch = epoch
            improved = True

        if improved:
            early_stop = 0
        elif epoch > 50:
            early_stop += 1
            if early_stop >= args.patience:
                print('Early Stopping!')
                break

        if epoch > 15:
            scheduler.step()

    save_best_checkpoint_summary(train_rows, best_all_epoch, best_lesion_epoch, output_dir)
    if args.export_visuals:
        ckpt_all = torch.load(os.path.join(output_dir, 'best_model_all_case.pth'), map_location=device)
        model.load_state_dict(ckpt_all['model_state_dict'])
        export_visualizations(model, vis_loader, device, float(ckpt_all.get('best_threshold', 0.5)), os.path.join(output_dir, 'visuals_all_case'), args.experiment_name + '_all_case', args, max_samples=args.max_visual_samples)
        ckpt_lesion = torch.load(os.path.join(output_dir, 'best_model_lesion_only.pth'), map_location=device)
        model.load_state_dict(ckpt_lesion['model_state_dict'])
        export_visualizations(model, vis_loader, device, float(ckpt_lesion.get('best_threshold', 0.5)), os.path.join(output_dir, 'visuals_lesion_only'), args.experiment_name + '_lesion_only', args, max_samples=args.max_visual_samples)
    print(f'实验: {args.experiment_name}')
    print(f'模型 model_arch={MODEL_ARCH} stem_channels={args.stem_channels} denoise_strength={args.denoise_strength} raw_branch_bias={args.raw_branch_bias} fusion_strength={args.fusion_strength} fusion_mode={args.fusion_mode}')
    print(f'开关 use_mixer={args.use_mixer} use_msag={args.use_msag} use_att_gate={args.use_att_gate} use_edge_attention={args.use_edge_attention} use_cls={args.use_cls}')
    print(f'策略 cls_loss_weight={args.cls_loss_weight} use_soft_cls_gate={args.use_soft_cls_gate} cls_gate_strength={args.cls_gate_strength} normal_penalty_weight={args.normal_penalty_weight}')
    print(f'困难样本 hard_sample_weight={args.hard_sample_weight} hard_normal_weight={args.hard_normal_weight} hard_lesion_dice_threshold={args.hard_lesion_dice_threshold} hard_normal_prob_threshold={args.hard_normal_prob_threshold}')
    print(f'训练结束！Best all-case dice_mean: {best_all:.4f} (epoch={best_all_epoch})')
    print(f'训练结束！Best lesion-only dice_mean: {best_lesion:.4f} (epoch={best_lesion_epoch})')
    print(f'结果保存在: {output_dir}')


if __name__ == '__main__':
    main()


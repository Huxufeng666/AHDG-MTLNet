import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.get_data import BUSI_Data

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MTL_models.MTL_DualBranch_HighFreqShallowV8_Resnet18 import DualBranchHighFreqShallowV8_Resnet18


ABLATION_MODEL_KWARGS = dict(use_hfd=True, use_detail_fusion=True, hfd_input_injection=False, fusion_mode='adf', use_msag=True)

MODEL_NAME = 'module_ablation_5_Base_HFD_DetailFusion_MSAG'

CLASS_NAMES = ["normal", "benign", "malignant"]
NORMAL_CLASS_ID = 0


class SegMetricMeter:
    def __init__(self, eps=1e-7):
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    @torch.no_grad()
    def update_pred(self, pred_bin, masks):
        if pred_bin.dim() == 3:
            pred_bin = pred_bin.unsqueeze(1)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        pred_bin = pred_bin.float()
        masks = masks.float()
        if masks.max() > 1:
            masks = masks / 255.0
        p = pred_bin.reshape(-1)
        g = masks.reshape(-1)
        self.tp += (p * g).sum().item()
        self.fp += (p * (1.0 - g)).sum().item()
        self.fn += ((1.0 - p) * g).sum().item()

    def compute(self):
        dice = (2.0 * self.tp + self.eps) / (2.0 * self.tp + self.fp + self.fn + self.eps)
        iou = (self.tp + self.eps) / (self.tp + self.fp + self.fn + self.eps)
        precision = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)
        return float(dice), float(iou), float(precision), float(recall)


def parse_args():
    parser = argparse.ArgumentParser(description="BUSI joint model evaluation.")
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--select-metric", choices=["dice_mean", "dice_global", "lesion_dice_mean", "lesion_iou_mean"], default="dice_mean")
    parser.add_argument("--threshold-start", type=float, default=0.35)
    parser.add_argument("--threshold-end", type=float, default=0.56)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-soft-cls-gate", type=int, default=None)
    parser.add_argument("--cls-gate-strength", type=float, default=None)
    parser.add_argument("--flip-tta", type=int, default=0)
    parser.add_argument("--use-checkpoint-threshold", type=int, default=1)
    return parser.parse_args()


def build_model(
    use_cls, device, mixer_depth=4, mixer_kernel=7, edge_alpha=0.08,
    use_mixer=True, use_msag=True, use_att_gate=True, use_edge_attention=True,
    stem_channels=16, raw_branch_bias=0.65, fusion_strength=0.35, detail_strength=1.0,
):
    """Build the project's only supported architecture: HighFreqShallow V8 ResNet-18."""
    return DualBranchHighFreqShallowV8_Resnet18(
        img_ch=1, output_ch=1, cls_classes=3, use_cls=use_cls,
        mixer_depth=mixer_depth, mixer_kernel=mixer_kernel, edge_alpha=edge_alpha,
        use_mixer=use_mixer, use_att_gate=use_att_gate,
        use_edge_attention=use_edge_attention, stem_channels=stem_channels,
        detail_strength=detail_strength, raw_branch_bias=raw_branch_bias,
        fusion_strength=fusion_strength,
        **ABLATION_MODEL_KWARGS,
    ).to(device)

def unpack_outputs(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 3:
            return outputs[0], outputs[1], outputs[2]
        if len(outputs) == 2:
            return outputs[0], outputs[1], None
    raise ValueError("Unexpected model output.")


def compute_metrics_per_image(pred_bin, gt_bin, smooth=1e-6):
    if pred_bin.dim() == 3:
        pred_bin = pred_bin.squeeze(0)
    if gt_bin.dim() == 3:
        gt_bin = gt_bin.squeeze(0)
    p = pred_bin.reshape(-1).float()
    g = gt_bin.reshape(-1).float()
    if g.max() > 1:
        g = g / 255.0
    tp = (p * g).sum()
    fp = (p * (1.0 - g)).sum()
    fn = ((1.0 - p) * g).sum()
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return float(dice.item()), float(iou.item()), float(precision.item()), float(recall.item())


def probs_to_logits(probs, eps=1e-4):
    probs = probs.clamp(eps, 1.0 - eps)
    return torch.logit(probs)


def apply_soft_cls_gate(seg_out, cls_out, use_soft_cls_gate=False, cls_gate_strength=0.75):
    seg_prob = torch.sigmoid(seg_out)
    if not use_soft_cls_gate or cls_out is None:
        return seg_out, seg_prob

    cls_prob = torch.softmax(cls_out, dim=1)
    lesion_prob = 1.0 - cls_prob[:, NORMAL_CLASS_ID]
    gate = (1.0 - float(cls_gate_strength)) + float(cls_gate_strength) * lesion_prob
    gate = gate.view(-1, 1, 1, 1)
    seg_prob = seg_prob * gate
    return probs_to_logits(seg_prob), seg_prob


def load_model(weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        model_name = MODEL_NAME
        use_cls = False
        mixer_depth = ckpt.get("mixer_depth", 4)
        mixer_kernel = ckpt.get("mixer_kernel", 7)
        edge_alpha = ckpt.get("edge_alpha", 0.08)
        use_mixer = ckpt.get("use_mixer", True)
        use_msag = ckpt.get("use_msag", True)
        use_att_gate = ckpt.get("use_att_gate", True)
        use_edge_attention = False
        stem_channels = ckpt.get("stem_channels", 16)
        raw_branch_bias = ckpt.get("raw_branch_bias", 0.65)
        fusion_strength = ckpt.get("fusion_strength", 0.35)
        detail_strength = ckpt.get("detail_strength", 1.0)
        use_soft_cls_gate = False
        cls_gate_strength = ckpt.get("cls_gate_strength", 0.75)
        best_threshold = ckpt.get("best_threshold")
        print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))
        print("Checkpoint best metric:", ckpt.get("best_metric_value", "unknown"))
        print("Checkpoint threshold:", best_threshold)
    else:
        state_dict = ckpt
        model_name = MODEL_NAME
        use_cls = False
        mixer_depth = 4
        mixer_kernel = 7
        edge_alpha = 0.08
        use_mixer = True
        use_msag = True
        use_att_gate = True
        use_edge_attention = False
        stem_channels = 16
        raw_branch_bias = 0.65
        fusion_strength = 0.35
        detail_strength = 1.0
        use_soft_cls_gate = False
        cls_gate_strength = 0.75
        best_threshold = None

    checkpoint_model_name = ckpt.get("model_name") if isinstance(ckpt, dict) else None
    if checkpoint_model_name and checkpoint_model_name != MODEL_NAME:
        print(f"Ignoring checkpoint model_name={checkpoint_model_name!r}; using {MODEL_NAME!r}.")
    model = build_model(
        use_cls, device, mixer_depth, mixer_kernel, edge_alpha, use_mixer, use_msag,
        use_att_gate, use_edge_attention, stem_channels, raw_branch_bias,
        fusion_strength, detail_strength,
    )
    new_state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print("Loaded weight:", weight_path)
    print("Model:", model_name)
    print("Use cls:", use_cls)
    print(
        "Ablation flags:",
        f"use_mixer={use_mixer}",
        f"use_msag={use_msag}",
        f"use_att_gate={use_att_gate}",
        f"use_edge_attention={use_edge_attention}",
    )
    eval_cfg = {
        "use_soft_cls_gate": bool(use_soft_cls_gate),
        "cls_gate_strength": float(cls_gate_strength),
    }
    print(
        "Eval gate:",
        f"use_soft_cls_gate={eval_cfg['use_soft_cls_gate']}",
        f"cls_gate_strength={eval_cfg['cls_gate_strength']}",
    )
    return model, best_threshold, eval_cfg


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    threshold=0.5,
    save_sample_csv=False,
    result_dir=None,
    use_soft_cls_gate=False,
    cls_gate_strength=0.75,
    flip_tta=False,
):
    model.eval()
    global_meter = SegMetricMeter()
    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []

    lesion_dice_list = []
    lesion_iou_list = []
    lesion_precision_list = []
    lesion_recall_list = []

    normal_total = 0
    normal_fp_images = 0
    normal_empty_correct = 0
    normal_fp_pixels = []

    sample_rows = []

    for imgs, masks, cls_gt, _, _, names in tqdm(loader):
        imgs = imgs.to(device)
        masks = masks.to(device).float()
        cls_gt = cls_gt.to(device).long()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        if masks.max() > 1:
            masks = masks / 255.0

        seg_out, _, cls_out = unpack_outputs(model(imgs))
        seg_out = F.interpolate(seg_out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        if flip_tta:
            flipped_imgs = torch.flip(imgs, dims=[-1])
            flipped_seg_out, _, flipped_cls_out = unpack_outputs(model(flipped_imgs))
            flipped_seg_out = F.interpolate(flipped_seg_out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            flipped_prob = torch.flip(torch.sigmoid(flipped_seg_out), dims=[-1])
            seg_out = probs_to_logits(0.5 * (torch.sigmoid(seg_out) + flipped_prob))
            if cls_out is not None and flipped_cls_out is not None:
                cls_out = 0.5 * (cls_out + flipped_cls_out)
        _, seg_prob = apply_soft_cls_gate(
            seg_out,
            cls_out,
            use_soft_cls_gate=use_soft_cls_gate,
            cls_gate_strength=cls_gate_strength,
        )
        pred = (seg_prob > threshold).float()

        if cls_out is not None:
            cls_pred = torch.argmax(cls_out, dim=1)
        else:
            cls_pred = torch.full_like(cls_gt, -1)

        global_meter.update_pred(pred, masks)

        for i in range(pred.size(0)):
            d, j, p, r = compute_metrics_per_image(pred[i], masks[i])
            dice_list.append(d)
            iou_list.append(j)
            precision_list.append(p)
            recall_list.append(r)

            gt_cls_i = int(cls_gt[i].detach().cpu().item())
            pred_cls_i = int(cls_pred[i].detach().cpu().item())
            pred_pixels = float(pred[i].sum().detach().cpu().item())

            if gt_cls_i == NORMAL_CLASS_ID:
                normal_total += 1
                normal_fp_pixels.append(pred_pixels)
                if pred_pixels > 0:
                    normal_fp_images += 1
                else:
                    normal_empty_correct += 1
            else:
                lesion_dice_list.append(d)
                lesion_iou_list.append(j)
                lesion_precision_list.append(p)
                lesion_recall_list.append(r)

            sample_name = names[i] if isinstance(names, (list, tuple)) else str(names)
            sample_rows.append([
                sample_name,
                d,
                j,
                p,
                r,
                gt_cls_i,
                pred_cls_i,
                pred_pixels,
                CLASS_NAMES[gt_cls_i] if gt_cls_i < len(CLASS_NAMES) else str(gt_cls_i),
                CLASS_NAMES[pred_cls_i] if 0 <= pred_cls_i < len(CLASS_NAMES) else str(pred_cls_i),
            ])

    dice_g, iou_g, precision_g, recall_g = global_meter.compute()
    normal_fp_rate = (normal_fp_images / normal_total) if normal_total > 0 else 0.0
    normal_empty_accuracy = (normal_empty_correct / normal_total) if normal_total > 0 else 0.0

    metrics = {
        "dice_global": dice_g,
        "iou_global": iou_g,
        "seg_precision_global": precision_g,
        "seg_recall_global": recall_g,
        "dice_mean": _safe_mean(dice_list),
        "iou_mean": _safe_mean(iou_list),
        "seg_precision_mean": _safe_mean(precision_list),
        "seg_recall_mean": _safe_mean(recall_list),
        "lesion_dice_mean": _safe_mean(lesion_dice_list),
        "lesion_iou_mean": _safe_mean(lesion_iou_list),
        "lesion_precision_mean": _safe_mean(lesion_precision_list),
        "lesion_recall_mean": _safe_mean(lesion_recall_list),
        "normal_fp_rate": normal_fp_rate,
        "normal_empty_accuracy": normal_empty_accuracy,
        "normal_fp_pixels_mean": _safe_mean(normal_fp_pixels),
    }

    if save_sample_csv and result_dir is not None:
        sample_csv = os.path.join(result_dir, "sample_metrics.csv")
        with open(sample_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "dice", "iou", "precision", "recall", "gt_cls", "pred_cls", "pred_pixels", "gt_cls_name", "pred_cls_name"])
            writer.writerows(sample_rows)

    return metrics


@torch.no_grad()
def sweep_threshold(model, loader, device, thresholds, select_metric, eval_cfg):
    best = {"threshold": None, "score": -1.0, "metrics": None}
    for t in thresholds:
        metrics = evaluate(
            model,
            loader,
            device,
            threshold=float(t),
            save_sample_csv=False,
            result_dir=None,
            **eval_cfg,
        )
        if metrics[select_metric] > best["score"]:
            best = {"threshold": float(t), "score": metrics[select_metric], "metrics": metrics}
    return best


def save_metrics_csv(metrics, threshold, save_path, metric_name, metric_value):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["threshold", threshold])
        writer.writerow(["selected_metric", metric_name])
        writer.writerow(["selected_metric_value", metric_value])
        for k, v in metrics.items():
            writer.writerow([k, v])


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_threshold, eval_cfg = load_model(args.weight_path, device)
    if args.use_soft_cls_gate is not None:
        eval_cfg["use_soft_cls_gate"] = bool(args.use_soft_cls_gate)
    if args.cls_gate_strength is not None:
        eval_cfg["cls_gate_strength"] = float(args.cls_gate_strength)
    eval_cfg["flip_tta"] = bool(args.flip_tta)
    print(
        "Final eval gate:",
        f"use_soft_cls_gate={eval_cfg['use_soft_cls_gate']}",
        f"cls_gate_strength={eval_cfg['cls_gate_strength']}",
    )
    result_dir = os.path.dirname(args.weight_path)
    os.makedirs(result_dir, exist_ok=True)

    dataset = BUSI_Data(root_dir=args.data_root, split="test", img_size=args.img_size, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    thresholds = np.arange(args.threshold_start, args.threshold_end + 1e-9, args.threshold_step)
    if ckpt_threshold is not None and not np.any(np.isclose(thresholds, ckpt_threshold)):
        thresholds = np.unique(np.append(thresholds, float(ckpt_threshold)))
        thresholds.sort()

    if bool(args.use_checkpoint_threshold):
        if ckpt_threshold is None:
            raise ValueError("Checkpoint does not contain best_threshold; fixed validation threshold is unavailable.")
        fixed_metrics = evaluate(model, loader, device, threshold=float(ckpt_threshold), **eval_cfg)
        best = {
            "threshold": float(ckpt_threshold),
            "score": fixed_metrics[args.select_metric],
            "metrics": fixed_metrics,
        }
    else:
        best = sweep_threshold(model, loader, device, thresholds, args.select_metric, eval_cfg)
    final_metrics = evaluate(
        model,
        loader,
        device,
        threshold=best["threshold"],
        save_sample_csv=True,
        result_dir=result_dir,
        **eval_cfg,
    )
    csv_path = os.path.join(result_dir, "test_metrics_global.csv")
    save_metrics_csv(final_metrics, best["threshold"], csv_path, args.select_metric, best["score"])
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()

'''
  python test.py \
    --weight-path /path/to/checkpoint.pth \
    --data-root /path/to/dataset_root \
    --img-size 256 \
    --batch-size 8 \
    --select-metric lesion_dice_mean


'''

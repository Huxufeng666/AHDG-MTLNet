import os
import sys
import random
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.utils as vutils






# ====================== 基础配置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)


@torch.no_grad()
def visualize_fixed(epoch, model, val_loader, log_dir, threshold=0.5):
    model.eval()

    fixed_idx = min(10, len(val_loader.dataset) - 1)
    fixed_data = val_loader.dataset[fixed_idx]

    if len(fixed_data) == 5:
        fixed_imgs, fixed_masks, fixed_cls, fixed_edge, fixed_text = fixed_data
    else:
        fixed_imgs, fixed_masks, fixed_cls, fixed_edge, *rest = fixed_data
        fixed_text = fixed_cls

    fixed_imgs = fixed_imgs.unsqueeze(0).to(device)

    if fixed_masks.dim() == 2:
        fixed_masks = fixed_masks.unsqueeze(0).unsqueeze(0)
    elif fixed_masks.dim() == 3:
        fixed_masks = fixed_masks.unsqueeze(0)
    fixed_masks = fixed_masks.float().to(device)
    fixed_masks = fixed_masks / 255.0 if fixed_masks.max() > 1 else fixed_masks

    fixed_text = torch.tensor([int(fixed_text)]).long().to(device)
    fixed_cls = torch.tensor([int(fixed_cls)]).long().to(device)

    seg_out, edge_out, cls_out = model(fixed_imgs, text_ids=fixed_text)

    seg_out = F.interpolate(seg_out, size=fixed_masks.shape[-2:], mode="bilinear", align_corners=False)
    edge_out = F.interpolate(edge_out, size=fixed_masks.shape[-2:], mode="bilinear", align_corners=False)

    pred_mask = (torch.sigmoid(seg_out) > threshold).float()
    edge_pred = (torch.sigmoid(edge_out) > threshold).float()

    imgs_vis = fixed_imgs * 0.5 + 0.5
    imgs_vis = torch.clamp(imgs_vis, 0, 1)

    min_val = imgs_vis.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    max_val = imgs_vis.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    imgs_vis = (imgs_vis - min_val) / (max_val - min_val + 1e-6)

    imgs_vis = imgs_vis.cpu()
    masks_vis = fixed_masks.cpu()
    pred_vis = pred_mask.cpu()
    edge_pred_vis = edge_pred.cpu()

    img_rgb = imgs_vis.repeat(1, 3, 1, 1)
    gt_overlay = overlay_mask(imgs_vis, masks_vis, "red")
    pred_overlay = overlay_mask(imgs_vis, pred_vis, "green")

    overlap = img_rgb.clone()
    overlap[:, 0] = torch.clamp(overlap[:, 0] + masks_vis.squeeze(1), 0, 1)
    overlap[:, 1] = torch.clamp(overlap[:, 1] + pred_vis.squeeze(1), 0, 1)

    edge_pred_rgb = edge_pred_vis.repeat(1, 3, 1, 1)

    vis = torch.cat([
        img_rgb,
        gt_overlay,
        pred_overlay,
        edge_pred_rgb,
        overlap
    ], dim=3)

    save_path = os.path.join(log_dir, "train_vis", f"vis_epoch_{epoch}.png")
    vutils.save_image(vis, save_path, nrow=1, normalize=True)

    img_pil = Image.open(save_path)
    draw = ImageDraw.Draw(img_pil)
    width, height = img_pil.size
    block_w = width // 5
    titles = ["Image", "GT", "Pred", "Edge Pred", "Overlap"]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, title in enumerate(titles):
        draw.text((block_w * i + block_w // 4, 5), title, fill=(255, 255, 0), font=font)

    dice, iou, precision, recall = calc_seg_metrics(seg_out, fixed_masks, threshold=threshold)
    cls_pred = torch.argmax(torch.softmax(cls_out, dim=1), dim=1)
    cls_acc = (cls_pred == fixed_cls).float().mean().item()

    info = f"Dice:{dice:.4f} IoU:{iou:.4f} P:{precision:.4f} R:{recall:.4f} cls_acc:{cls_acc:.2f}"
    draw.text((10, height - 20), info, fill=(255, 255, 0), font=font)
    img_pil.save(save_path)

    print(f"✅ 可视化保存: {save_path}")
    
    
@torch.no_grad()
def visualize_fixed2(
    epoch,
    model,
    val_loader,
    log_dir,
    tokenizer,
    device,
    threshold=0.5
):
    model.eval()

    fixed_idx = min(10, len(val_loader.dataset) - 1)
    fixed_data = val_loader.dataset[fixed_idx]

    if len(fixed_data) == 5:
        fixed_imgs, fixed_masks, fixed_cls, fixed_edge, fixed_text = fixed_data
    else:
        fixed_imgs, fixed_masks, fixed_cls, fixed_edge, *rest = fixed_data
        fixed_text = fixed_cls

    fixed_imgs = fixed_imgs.unsqueeze(0).to(device)

    if fixed_masks.dim() == 2:
        fixed_masks = fixed_masks.unsqueeze(0).unsqueeze(0)
    elif fixed_masks.dim() == 3:
        fixed_masks = fixed_masks.unsqueeze(0)

    fixed_masks = fixed_masks.float().to(device)
    fixed_masks = fixed_masks / 255.0 if fixed_masks.max() > 1 else fixed_masks

    fixed_cls = torch.tensor([int(fixed_cls)]).long().to(device)

    # =====================================================
    # 关键修改：BERT 文本输入，不再使用 text_ids
    # =====================================================
    if isinstance(fixed_text, str):
        texts = [fixed_text]
    else:
        fixed_text = int(fixed_text)

        if fixed_text == 0:
            texts = [
                "This is a normal breast ultrasound image without tumor."
            ]
        elif fixed_text == 1:
            texts = [
                "This is a benign breast ultrasound tumor with relatively clear boundary."
            ]
        else:
            texts = [
                "This is a malignant breast ultrasound tumor with irregular shape and blurry boundary."
            ]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    seg_out, edge_out, cls_out = model(
        fixed_imgs,
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    seg_out = F.interpolate(
        seg_out,
        size=fixed_masks.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    edge_out = F.interpolate(
        edge_out,
        size=fixed_masks.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    pred_mask = (torch.sigmoid(seg_out) > threshold).float()
    edge_pred = (torch.sigmoid(edge_out) > threshold).float()

    imgs_vis = fixed_imgs * 0.5 + 0.5
    imgs_vis = torch.clamp(imgs_vis, 0, 1)

    min_val = imgs_vis.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    max_val = imgs_vis.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    imgs_vis = (imgs_vis - min_val) / (max_val - min_val + 1e-6)

    imgs_vis = imgs_vis.cpu()
    masks_vis = fixed_masks.cpu()
    pred_vis = pred_mask.cpu()
    edge_pred_vis = edge_pred.cpu()

    img_rgb = imgs_vis.repeat(1, 3, 1, 1)
    gt_overlay = overlay_mask(imgs_vis, masks_vis, "red")
    pred_overlay = overlay_mask(imgs_vis, pred_vis, "green")

    overlap = img_rgb.clone()
    overlap[:, 0] = torch.clamp(
        overlap[:, 0] + masks_vis.squeeze(1),
        0,
        1
    )
    overlap[:, 1] = torch.clamp(
        overlap[:, 1] + pred_vis.squeeze(1),
        0,
        1
    )

    edge_pred_rgb = edge_pred_vis.repeat(1, 3, 1, 1)

    vis = torch.cat(
        [
            img_rgb,
            gt_overlay,
            pred_overlay,
            edge_pred_rgb,
            overlap
        ],
        dim=3
    )

    save_dir = os.path.join(log_dir, "train_vis")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"vis_epoch_{epoch}.png"
    )

    vutils.save_image(vis, save_path, nrow=1, normalize=True)

    img_pil = Image.open(save_path)
    draw = ImageDraw.Draw(img_pil)

    width, height = img_pil.size
    block_w = width // 5
    titles = ["Image", "GT", "Pred", "Edge Pred", "Overlap"]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            14
        )
    except Exception:
        font = ImageFont.load_default()

    for i, title in enumerate(titles):
        draw.text(
            (block_w * i + block_w // 4, 5),
            title,
            fill=(255, 255, 0),
            font=font
        )

    dice, iou, precision, recall = calc_seg_metrics(
        seg_out,
        fixed_masks,
        threshold=threshold
    )

    cls_pred = torch.argmax(torch.softmax(cls_out, dim=1), dim=1)
    cls_acc = (cls_pred == fixed_cls).float().mean().item()

    info = (
        f"Dice:{dice:.4f} "
        f"IoU:{iou:.4f} "
        f"P:{precision:.4f} "
        f"R:{recall:.4f} "
        f"cls_acc:{cls_acc:.2f}"
    )

    draw.text(
        (10, height - 20),
        info,
        fill=(255, 255, 0),
        font=font
    )

    # 可选：把 prompt 也写到图上
    draw.text(
        (10, height - 40),
        texts[0][:90],
        fill=(255, 255, 0),
        font=font
    )

    img_pil.save(save_path)

    print(f"✅ 可视化保存: {save_path}")
    
    
# ====================== 数据兼容函数 ======================
def unpack_batch(batch):
    """
    支持：
    1) img, mask, cls_gt, edge_gt, text_id
    2) img, mask, cls_gt, edge_gt, ...
       没有 text_id 时自动 text_id = cls_gt
    """
    if len(batch) == 5:
        imgs, masks, cls_gt, edge_gt, text_ids = batch
    elif len(batch) >= 4:
        imgs, masks, cls_gt, edge_gt, *rest = batch
        text_ids = cls_gt
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    return imgs, masks, cls_gt, edge_gt, text_ids

def unpack_batch2(batch):
    if len(batch) == 5:
        imgs, masks, cls_gt, edge_gt, texts = batch
    elif len(batch) >= 4:
        imgs, masks, cls_gt, edge_gt, *rest = batch

        texts = []
        for c in cls_gt:
            c = int(c)
            if c == 0:
                texts.append("This is a normal breast ultrasound image without tumor.")
            elif c == 1:
                texts.append("This is a benign breast ultrasound tumor with relatively clear boundary.")
            else:
                texts.append("This is a malignant breast ultrasound tumor with irregular shape and blurry boundary.")
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    return imgs, masks, cls_gt, edge_gt, texts



def prepare_binary_tensor(x):
    x = x.float()
    if x.dim() == 3:
        x = x.unsqueeze(1)
    x = x / 255.0 if x.max() > 1 else x
    return x


def overlay_mask(image, mask, color):
    image = image.repeat(1, 3, 1, 1)

    if color == "red":
        image[:, 0] = torch.clamp(image[:, 0] + mask.squeeze(1) * 0.8, 0, 1)
    elif color == "green":
        image[:, 1] = torch.clamp(image[:, 1] + mask.squeeze(1) * 0.8, 0, 1)

    return image


def dice_loss(logits, target, smooth=1e-6):
    prob = torch.sigmoid(logits)
    prob = prob.view(prob.size(0), -1)
    target = target.view(target.size(0), -1)

    inter = (prob * target).sum(dim=1)
    dice = (2.0 * inter + smooth) / (prob.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def calc_seg_metrics(logits, mask, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()

    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()
    fn = ((1 - pred) * mask).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return dice.item(), iou.item(), precision.item(), recall.item()



@torch.no_grad()
def find_best_threshold(model, val_loader, device):
    model.eval()
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

    best_t = 0.5
    best_dice = -1.0

    for t in thresholds:
        dice_total = 0.0
        count = 0

        for batch in val_loader:
            imgs, masks, cls_gt, edge_gt, text_ids = unpack_batch(batch)

            imgs = imgs.to(device)
            masks = prepare_binary_tensor(masks).to(device)
            text_ids = text_ids.long().to(device)

            seg_out, edge_out, cls_out = model(imgs, text_ids=text_ids)
            seg_out = F.interpolate(seg_out, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            dice, _, _, _ = calc_seg_metrics(seg_out, masks, threshold=t)
            dice_total += dice
            count += 1

        avg_dice = dice_total / max(count, 1)

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_t = t

    return best_t, best_dice


@torch.no_grad()
def find_best_threshold2(model, val_loader, device, tokenizer):
    model.eval()

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

    best_t = 0.5
    best_dice = -1.0

    for t in thresholds:
        dice_total = 0.0
        count = 0

        for batch in val_loader:
            imgs, masks, cls_gt, edge_gt, texts = unpack_batch2(batch)

            imgs = imgs.to(device)
            masks = prepare_binary_tensor(masks).to(device)

            # ===== text prompt 处理 =====
            texts = list(texts)

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # ===== BERT 版本 forward =====
            seg_out, edge_out, cls_out = model(
                imgs,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            seg_out = F.interpolate(
                seg_out,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            dice, _, _, _ = calc_seg_metrics(
                seg_out,
                masks,
                threshold=t
            )

            dice_total += dice
            count += 1

        avg_dice = dice_total / max(count, 1)

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_t = t

    return best_t, best_dice
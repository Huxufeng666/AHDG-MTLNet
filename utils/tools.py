import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# utils/tools.py
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import cv2
import numpy as np


import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont


from PIL import Image
import torchvision.transforms as T



import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont


def visualize_fixed(
    epoch,
    model,
    device,
    log_dir,
    fixed_imgs,
    fixed_masks,
    fixed_cls,
    fixed_edge,
    threshold=0.5,
    max_vis=2,
    class_names=None
):
    """
    固定样本可视化函数

    参数：
        epoch       : 当前 epoch
        model       : 训练模型
        device      : torch.device
        log_dir     : 日志保存目录
        fixed_imgs  : [B,C,H,W]
        fixed_masks : [B,1,H,W] 或 [B,H,W]
        fixed_cls   : [B]
        fixed_edge  : [B,1,H,W] 或 [B,H,W]
        threshold   : 分割/边缘二值化阈值
        max_vis     : 最多可视化前几张
        class_names : 分类名称列表，如 ["normal", "benign", "malignant"]
    """

    if class_names is None:
        class_names = ["normal", "benign", "malignant"]

    model.eval()

    with torch.no_grad():

        # =====================================================
        # 数据准备
        # =====================================================
        imgs = fixed_imgs.to(device)
        masks = fixed_masks.to(device).float()
        cls = fixed_cls.to(device)
        edge_gt = fixed_edge.to(device).float()

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        if edge_gt.dim() == 3:
            edge_gt = edge_gt.unsqueeze(1)

        masks = masks / 255.0 if masks.max() > 1 else masks
        edge_gt = edge_gt / 255.0 if edge_gt.max() > 1 else edge_gt

        masks = (masks > 0.5).float()
        edge_gt = (edge_gt > 0.5).float()

        # =====================================================
        # forward
        # =====================================================
        outputs = model(imgs)

        # 兼容不同输出形式
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 3:
                seg_out, edge_out, cls_out = outputs
            elif len(outputs) == 2:
                seg_out, edge_out = outputs
                cls_out = None
            else:
                raise ValueError(f"Unexpected model outputs length: {len(outputs)}")
        else:
            raise ValueError("Model output must be tuple/list, e.g. (seg_out, edge_out, cls_out).")

        # =====================================================
        # 对齐尺寸
        # =====================================================
        logits = F.interpolate(
            seg_out,
            size=masks.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        edge_out = F.interpolate(
            edge_out,
            size=edge_gt.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # =====================================================
        # 预测
        # =====================================================
        pred_mask = (torch.sigmoid(logits) > threshold).float()
        edge_pred = (torch.sigmoid(edge_out) > threshold).float()

        # =====================================================
        # 分类
        # =====================================================
        if cls_out is not None:
            cls_prob = torch.softmax(cls_out, dim=1)
            cls_pred = torch.argmax(cls_prob, dim=1)
        else:
            cls_prob = None
            cls_pred = None

        # =====================================================
        # 只取前 max_vis 张
        # =====================================================
        imgs_vis = imgs[:max_vis]
        masks_vis = masks[:max_vis]
        pred_vis = pred_mask[:max_vis]
        edge_gt_vis = edge_gt[:max_vis]
        edge_pred_vis = edge_pred[:max_vis]

        cls_gt_vis = cls[:max_vis].detach().cpu()
        if cls_pred is not None:
            cls_pred_vis = cls_pred[:max_vis].detach().cpu()
            cls_prob_vis = cls_prob[:max_vis].detach().cpu()
        else:
            cls_pred_vis = None
            cls_prob_vis = None

        # =====================================================
        # 反归一化 + 对比度增强
        # =====================================================
        imgs_vis = imgs_vis * 0.5 + 0.5
        imgs_vis = torch.clamp(imgs_vis, 0, 1)

        min_val = imgs_vis.amin(dim=(2, 3), keepdim=True)
        max_val = imgs_vis.amax(dim=(2, 3), keepdim=True)
        imgs_vis = (imgs_vis - min_val) / (max_val - min_val + 1e-6)

        imgs_vis = imgs_vis.cpu()
        masks_vis = masks_vis.cpu()
        pred_vis = pred_vis.cpu()
        edge_gt_vis = edge_gt_vis.cpu()
        edge_pred_vis = edge_pred_vis.cpu()

        # =====================================================
        # overlay 辅助函数
        # =====================================================
        def overlay_mask(image, mask, color="red"):
            """
            image: [B,1,H,W]
            mask : [B,1,H,W]
            return: [B,3,H,W]
            """
            image_rgb = image.repeat(1, 3, 1, 1).clone()

            if color == "red":
                image_rgb[:, 0] = torch.clamp(image_rgb[:, 0] + mask.squeeze(1) * 0.8, 0, 1)
            elif color == "green":
                image_rgb[:, 1] = torch.clamp(image_rgb[:, 1] + mask.squeeze(1) * 0.8, 0, 1)
            elif color == "blue":
                image_rgb[:, 2] = torch.clamp(image_rgb[:, 2] + mask.squeeze(1) * 0.8, 0, 1)

            return image_rgb

        # =====================================================
        # 构造可视化图
        # =====================================================
        img_rgb = imgs_vis.repeat(1, 3, 1, 1)

        gt_overlay = overlay_mask(imgs_vis, masks_vis, "red")
        pred_overlay = overlay_mask(imgs_vis, pred_vis, "green")

        overlap = img_rgb.clone()
        overlap[:, 0] = torch.clamp(overlap[:, 0] + masks_vis.squeeze(1), 0, 1)   # GT 红色
        overlap[:, 1] = torch.clamp(overlap[:, 1] + pred_vis.squeeze(1), 0, 1)    # Pred 绿色

        edge_gt_rgb = edge_gt_vis.repeat(1, 3, 1, 1)
        edge_pred_rgb = edge_pred_vis.repeat(1, 3, 1, 1)

        # 横向拼接：6个块
        vis = torch.cat([
            img_rgb,
            gt_overlay,
            pred_overlay,
            edge_gt_rgb,
            edge_pred_rgb,
            overlap
        ], dim=3)

        # =====================================================
        # 保存图片
        # =====================================================
        save_dir = os.path.join(log_dir, "train")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"vis_epoch_{epoch}.png")

        vutils.save_image(vis, save_path, nrow=1, normalize=False)

        # =====================================================
        # 添加标题 + 分类结果
        # =====================================================
        img_pil = Image.open(save_path)
        draw = ImageDraw.Draw(img_pil)

        width, height = img_pil.size
        block_w = width // 6

        titles = ["Image", "GT", "Pred", "Edge GT", "Edge Pred", "Overlap"]

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        for i, title in enumerate(titles):
            x = block_w * i + 20
            y = 8
            draw.text((x, y), title, fill=(255, 255, 0), font=font)

        # 分类信息
        if cls_pred_vis is not None:
            for i in range(len(cls_gt_vis)):
                gt_name = class_names[int(cls_gt_vis[i])]
                pd_name = class_names[int(cls_pred_vis[i])]
                conf = cls_prob_vis[i].max().item()

                text = f"GT:{gt_name} | Pred:{pd_name} | Conf:{conf:.2f}"
                color = (0, 255, 0) if gt_name == pd_name else (255, 0, 0)

                # 每个样本单独一行
                draw.text(
                    (10, height - 30 - i * 22),
                    text,
                    fill=color,
                    font=font
                )

        img_pil.save(save_path)

    print(f"✅ 可视化保存: {save_path}")


    
@torch.no_grad()
def save_recon_visualization(model, val_loader, device, log_dir, epoch, max_vis=2):

    model.eval()

    # ===== 取一个batch =====
    val_iter = iter(val_loader)
    imgs, masks, cls, edge, recon, name = next(val_iter)

    imgs = imgs.to(device)

    # =========================
    # forward
    # =========================
    recon_out = model(imgs)

    # ===== 对齐尺寸 =====
    if recon_out.shape[-2:] != imgs.shape[-2:]:
        recon_out = F.interpolate(
            recon_out,
            size=imgs.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    # ==============================
    # 取前N张
    # ==============================
    imgs_vis = imgs[:max_vis]
    recon_vis = recon_out[:max_vis]

    # ==============================
    # 反归一化（必须）
    # ==============================
    imgs_vis = imgs_vis * 0.5 + 0.5
    recon_vis = recon_vis * 0.5 + 0.5

    imgs_vis = torch.clamp(imgs_vis, 0, 1)
    recon_vis = torch.clamp(recon_vis, 0, 1)

    # ==============================
    # 对比度增强（稳定写法）
    # ==============================
    imgs_min = imgs_vis.amin(dim=(2,3), keepdim=True)
    imgs_max = imgs_vis.amax(dim=(2,3), keepdim=True)
    imgs_vis = (imgs_vis - imgs_min) / (imgs_max - imgs_min + 1e-6)

    recon_min = recon_vis.amin(dim=(2,3), keepdim=True)
    recon_max = recon_vis.amax(dim=(2,3), keepdim=True)
    recon_vis = (recon_vis - recon_min) / (recon_max - recon_min + 1e-6)

    imgs_vis = imgs_vis.cpu()
    recon_vis = recon_vis.cpu()

    # ==============================
    # 转RGB
    # ==============================
    img_rgb = imgs_vis.repeat(1,3,1,1)
    recon_rgb = recon_vis.repeat(1,3,1,1)

    # ==============================
    # 差异图（关键）
    # ==============================
    diff = torch.abs(img_rgb - recon_rgb)
    diff = torch.clamp(diff * 3.0, 0, 1)

    # ==============================
    # 拼接
    # ==============================
    vis = torch.cat([
        img_rgb,
        recon_rgb,
        diff
    ], dim=3)

    # ==============================
    # 保存
    # ==============================
    save_dir = os.path.join(log_dir, "train")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"recon_epoch_{epoch}.png")

    vutils.save_image(
        vis,
        save_path,
        nrow=1,
        normalize=False   # ⚠️ 已经处理过，不要再normalize
    )

    # ==============================
    # 标题（优化版）
    # ==============================
    img_pil = Image.open(save_path)
    draw = ImageDraw.Draw(img_pil)

    width, _ = img_pil.size
    block_w = width // 3

    titles = ["Input Image", "Reconstruction", "Difference"]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            12
        )
    except:
        font = ImageFont.load_default()

    for i, title in enumerate(titles):

        bbox = draw.textbbox((0,0), title, font=font)
        text_w = bbox[2] - bbox[0]

        x = block_w * i + (block_w - text_w) // 2
        y = 10

        draw.text(
            (x, y),
            title,
            fill=(255,255,255),
            font=font,
            stroke_width=2,
            stroke_fill=(0,0,0)
        )

    img_pil.save(save_path)

    print(f"✅ Recon Visualization Saved: {save_path}")





@torch.no_grad()
def save_cls_visualization(model, val_loader, device, log_dir, epoch, max_vis=4):

    model.eval()

    # ===== 取一个batch =====
    val_iter = iter(val_loader)
    imgs, _, cls_gt, *_ , name = next(val_iter)

    imgs = imgs.to(device)
    cls_gt = cls_gt.to(device)

    # ===== forward =====
    cls_out = model(imgs)

    probs = torch.softmax(cls_out, dim=1)
    preds = torch.argmax(probs, dim=1)
    confs = torch.max(probs, dim=1)[0]

    # ===== 取前N张 =====
    imgs_vis = imgs[:max_vis]
    preds_vis = preds[:max_vis]
    cls_gt_vis = cls_gt[:max_vis]
    confs_vis = confs[:max_vis]

    # ===== 反归一化 =====
    imgs_vis = imgs_vis * 0.5 + 0.5
    imgs_vis = torch.clamp(imgs_vis, 0, 1)

    # ===== 对比度增强（更稳写法）=====
    min_val = imgs_vis.amin(dim=(2,3), keepdim=True)
    max_val = imgs_vis.amax(dim=(2,3), keepdim=True)
    imgs_vis = (imgs_vis - min_val) / (max_val - min_val + 1e-6)

    imgs_vis = imgs_vis.cpu()

    # ===== 转3通道 =====
    imgs_vis = imgs_vis.repeat(1,3,1,1)

    # ===== 保存路径 =====
    save_dir = os.path.join(log_dir, "train")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"cls_epoch_{epoch}.png")

    vutils.save_image(
        imgs_vis,
        save_path,
        nrow=max_vis,
        normalize=False
    )

    # ======================
    # 写文字（优化版）
    # ======================
    img_pil = Image.open(save_path)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()

    # ===== 类别名称 =====
    class_names = ["Normal", "Benign", "Malignant"]

    w, h = img_pil.size
    block_w = w // max_vis

    for i in range(len(imgs_vis)):

        gt = class_names[cls_gt_vis[i].item()]
        pd = class_names[preds_vis[i].item()]
        conf = confs_vis[i].item()

                # ===== 判断对错 =====
        if preds_vis[i].item() == cls_gt_vis[i].item():
            color = (0,255,0)   # ✅ 绿色（正确）
        else:
            color = (255,0,0)   # ❌ 红色（错误）


        text = f"GT: {gt}\nPred: {pd}\n({conf:.2f})"

        # ===== 左上角稍微偏移一点（避免贴边）=====
        x = block_w * i + 10
        y = 10

        draw.text(
            (x, y),
            text,
            fill= color,
            font=font,
            stroke_width=2,
            stroke_fill=(0,0,0)
        )

    img_pil.save(save_path)

    print(f"✅ Classification Visualization Saved: {save_path}")





def load_fixed_edge_sample(img_path, edge_path, img_size=256):

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

    # ===== image =====
    img = Image.open(img_path).convert("L")
    img = transform(img)

    # ===== edge =====
    edge = Image.open(edge_path).convert("L")
    edge = T.Resize((img_size, img_size))(edge)
    edge = T.ToTensor()(edge)
    edge = (edge > 0.5).float()

    return img.unsqueeze(0), edge.unsqueeze(0)
@torch.no_grad()
def save_fixed_edge_visualization(model, img, edge_gt, device, log_dir, epoch):

    model.eval()

    img = img.to(device)
    edge_gt = edge_gt.to(device)

    # ===== forward =====
    edge_out = model(img)

    edge_out = F.interpolate(
        edge_out,
        size=edge_gt.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    edge_pred = (torch.sigmoid(edge_out) > 0.5).float()

    # ===== 反归一化 =====
    img_vis = (img * 0.5 + 0.5).clamp(0,1)[0].cpu()
    edge_gt = edge_gt[0].cpu()
    edge_pred = edge_pred[0].cpu()

    img_rgb = img_vis.repeat(3,1,1)

    # ===== overlay =====
    gt = torch.zeros_like(img_rgb)
    gt[0] = edge_gt.squeeze(0)

    pd = torch.zeros_like(img_rgb)
    pd[1] = edge_pred.squeeze(0)

    overlap = img_rgb.clone()
    overlap[0] = torch.clamp(overlap[0] + edge_gt.squeeze(0), 0, 1)
    overlap[1] = torch.clamp(overlap[1] + edge_pred.squeeze(0), 0, 1)

    vis = torch.cat([img_rgb, gt, pd, overlap], dim=2)

    # ===== 保存 =====
    save_dir = os.path.join(log_dir, "fixed")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"epoch_{epoch}.png")

    vutils.save_image(vis, save_path, normalize=False)

    print(f"✅ Fixed Image Saved: {save_path}")





@torch.no_grad()
def save_edge_visualization(model, val_loader, device, log_dir, epoch, max_vis=2):

    model.eval()

    # ===== 取一个batch =====
    val_iter = iter(val_loader)
    imgs, masks, cls, edge, recon, name = next(val_iter)

    imgs = imgs.to(device)

    # ===== mask =====
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    masks = masks.float().to(device)

    # ===== edge GT =====
    if edge is not None:
        if edge.dim() == 3:
            edge = edge.unsqueeze(1)
        edge_masks = edge.float().to(device)
    else:
        raise ValueError("❌ edge is None，数据集没有提供edge GT")

    # ===== forward =====
    edge_out = model(imgs)

    # ===== resize =====
    edge_out = F.interpolate(
        edge_out,
        size=edge_masks.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    # ===== pred =====
    edge_pred = (torch.sigmoid(edge_out) > 0.5).float()

    # ==============================
    # 取前N张
    # ==============================
    imgs_vis = imgs[:max_vis]
    edge_gt_vis = edge_masks[:max_vis]
    edge_pred_vis = edge_pred[:max_vis]

    # ==============================
    # 反归一化 + normalize
    # ==============================
    imgs_vis = imgs_vis * 0.5 + 0.5
    imgs_vis = torch.clamp(imgs_vis, 0, 1)

    min_val = imgs_vis.amin(dim=(2,3), keepdim=True)
    max_val = imgs_vis.amax(dim=(2,3), keepdim=True)
    imgs_vis = (imgs_vis - min_val) / (max_val - min_val + 1e-6)

    # ===== CPU =====
    imgs_vis = imgs_vis.cpu()
    edge_gt_vis = edge_gt_vis.cpu()
    edge_pred_vis = edge_pred_vis.cpu()

    # ==============================
    # overlay（边界推荐写法）
    # ==============================
    def overlay_mask(image, mask, color):

        image = image.repeat(1,3,1,1)

        if color == "red":
            image[:,0] = torch.clamp(image[:,0] + mask.squeeze(1), 0, 1)

        elif color == "green":
            image[:,1] = torch.clamp(image[:,1] + mask.squeeze(1), 0, 1)

        return image

    # ===== 原图 =====
    img_rgb = imgs_vis.repeat(1,3,1,1)

    # ===== overlay =====
    edge_gt_overlay = overlay_mask(imgs_vis, edge_gt_vis, "red")
    edge_pred_overlay = overlay_mask(imgs_vis, edge_pred_vis, "green")

    # ===== overlap =====
    overlap = img_rgb.clone()
    overlap[:,0] = torch.clamp(overlap[:,0] + edge_gt_vis.squeeze(1), 0, 1)
    overlap[:,1] = torch.clamp(overlap[:,1] + edge_pred_vis.squeeze(1), 0, 1)

    # ==============================
    # 拼接
    # ==============================
    vis = torch.cat([
        img_rgb,
        edge_gt_overlay,
        edge_pred_overlay,
        overlap
    ], dim=3)

    # ==============================
    # 保存
    # ==============================
    save_dir = os.path.join(log_dir, "train")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"vis_epoch_{epoch}.png")

    vutils.save_image(vis, save_path, nrow=1, normalize=False)

    # ==============================
    # 标题（优化版）
    # ==============================
    img_pil = Image.open(save_path)
    draw = ImageDraw.Draw(img_pil)

    width, _ = img_pil.size
    block_w = width // 4

    titles = ["Image", "Edge GT", "Edge Pred", "Overlap"]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()

    for i, title in enumerate(titles):

        bbox = draw.textbbox((0,0), title, font=font)
        text_w = bbox[2] - bbox[0]

        x = block_w * i + (block_w - text_w) // 2
        y = 10

        draw.text(
            (x, y),
            title,
            fill=(255,255,255),
            font=font,
            stroke_width=2,
            stroke_fill=(0,0,0)
        )

    img_pil.save(save_path)

    print(f"✅ Edge Visualization Saved: {save_path}")






def get_target_layers(model):
    """
    自动寻找 segmentation CAM 层 和 classification CAM 层
    兼容 DataParallel / 普通模型
    """
    net = model.module if hasattr(model, "module") else model

    seg_candidates = [
        "dec1", "decoder1", "Up_conv2", "up1", "final_decoder",
        "dec", "decoder", "conv_last", "seg_head"
    ]
    cls_candidates = [
        "enc4", "encoder4", "Conv5", "down4", "bottleneck",
        "e4", "cls_head", "backbone"
    ]

    seg_layer = None
    cls_layer = None

    named_modules = dict(net.named_modules())

    for name in seg_candidates:
        if name in named_modules:
            seg_layer = named_modules[name]
            print(f"[INFO] Seg CAM layer -> {name}")
            break

    for name in cls_candidates:
        if name in named_modules:
            cls_layer = named_modules[name]
            print(f"[INFO] Cls CAM layer -> {name}")
            break

    if seg_layer is None:
        print("\n[DEBUG] available modules:")
        for k in named_modules.keys():
            print(k)
        raise AttributeError("No suitable segmentation CAM layer found. Please check model.named_modules().")

    if cls_layer is None:
        print("\n[DEBUG] available modules:")
        for k in named_modules.keys():
            print(k)
        raise AttributeError("No suitable classification CAM layer found. Please check model.named_modules().")

    return seg_layer, cls_layer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # ✅ 用新版 hook（重要）
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, target_output, size):

        self.model.zero_grad()

        target_output.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=False)

        cam = cam[0,0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam



def draw_text(img, text, font_size=20):

    img = (img * 255).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # ===== 字体 =====
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    font = ImageFont.truetype(font_path, font_size)

    # ===== 计算文字尺寸 =====
    text_bbox = draw.textbbox((0, 0), text, font=font)
    w = text_bbox[2] - text_bbox[0]
    h = text_bbox[3] - text_bbox[1]

    # ===== 位置 =====
    x, y = 10, 10

    # ===== 背景框（半透明白）=====
    overlay = img_pil.copy()
    draw_overlay = ImageDraw.Draw(overlay)

    draw_overlay.rectangle(
        [x-5, y-5, x+w+5, y+h+5],
        fill=(255, 255, 255, 160)  # 白色半透明
    )

    img_pil = Image.blend(img_pil, overlay, alpha=0.6)

    draw = ImageDraw.Draw(img_pil)

    # ===== 文字颜色（深灰，论文风）=====
    draw.text((x, y), text, font=font, fill=(30, 30, 30))

    return np.array(img_pil) / 255.0
# 类别映射（🔥关键）
# ===============================
class_names = {
    0: "normal",
    1: "benign",
    2: "malignant"
}


# ===============================
# 画文字
# ===============================
def draw_text(img, text, color=(0,255,0)):

    img = (img * 255).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        img,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1,
        cv2.LINE_AA
    )

    return img / 255.0


def save_visualizations2(model, loader, device, save_dir, threshold=0.5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # ===== 自动找 CAM 层 =====
    seg_layer, cls_layer = get_target_layers(model)

    seg_cam_extractor = GradCAM(model, seg_layer)
    cls_cam_extractor = GradCAM(model, cls_layer)

    for batch in tqdm(loader):

        img, mask, cls, edge, recon, name = batch

        img = img.to(device)
        mask = mask.to(device).float()
        edge = edge.to(device).float()
        cls = cls.to(device)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if edge.dim() == 3:
            edge = edge.unsqueeze(1)

        # ===============================
        # forward（开启梯度）
        # ===============================
        with torch.enable_grad():
            outputs = model(img)

            # ===== 兼容不同输出形式 =====
            if isinstance(outputs, (list, tuple)):
                if len(outputs) == 4:
                    seg_out, edge_out, recon_out, cls_out = outputs
                elif len(outputs) == 3:
                    seg_out, edge_out, cls_out = outputs
                    recon_out = None
                elif len(outputs) == 2:
                    seg_out, cls_out = outputs
                    edge_out, recon_out = None, None
                else:
                    raise ValueError(f"Unexpected number of outputs: {len(outputs)}")
            else:
                raise ValueError("Model output must be tuple/list for this visualization function.")

            seg_out = F.interpolate(
                seg_out,
                size=mask.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            if edge_out is not None:
                edge_out = F.interpolate(
                    edge_out,
                    size=edge.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # ===== segmentation CAM =====
            seg_score = seg_out.mean()
            seg_cam = seg_cam_extractor.generate(seg_score, size=img.shape[-2:])

            # ===== classification =====
            cls_prob = torch.softmax(cls_out, dim=1)
            pred_class = torch.argmax(cls_prob, dim=1)

            # 注意：这里用 gather 更稳
            cls_score = cls_out.gather(1, pred_class.unsqueeze(1)).sum()
            cls_cam = cls_cam_extractor.generate(cls_score, size=img.shape[-2:])

        # ===============================
        # segmentation + edge 推理
        # ===============================
        with torch.no_grad():
            seg_prob = torch.sigmoid(seg_out)
            pred_mask = (seg_prob > threshold).float()

            if edge_out is not None:
                edge_prob = torch.sigmoid(edge_out)
                pred_edge = (edge_prob > threshold).float()
            else:
                pred_edge = torch.zeros_like(edge)

        # ===============================
        # 可视化
        # ===============================
        for i in range(img.size(0)):
            img_np = img[i].detach().cpu().squeeze().numpy()
            mask_np = mask[i].detach().cpu().squeeze().numpy()
            pred_np = pred_mask[i].detach().cpu().squeeze().numpy()

            edge_gt_np = edge[i].detach().cpu().squeeze().numpy()
            edge_pred_np = pred_edge[i].detach().cpu().squeeze().numpy()

            # 防止归一化范围异常
            img_np = img_np.astype(np.float32)
            if img_np.max() > 1:
                img_np = img_np / 255.0
            img_np = np.clip(img_np, 0, 1)

            img_rgb = np.stack([img_np] * 3, axis=-1)

            # ===== GT（红色）=====
            gt_rgb = img_rgb.copy()
            gt_rgb[..., 2] += mask_np
            gt_rgb = np.clip(gt_rgb, 0, 1)

            # ===== Prediction（绿色）=====
            pd_rgb = img_rgb.copy()
            pd_rgb[..., 1] += pred_np
            pd_rgb = np.clip(pd_rgb, 0, 1)

            # ===== Edge =====
            edge_gt_rgb = np.stack([edge_gt_np] * 3, axis=-1)
            edge_pred_rgb = np.stack([edge_pred_np] * 3, axis=-1)

            edge_gt_rgb = np.clip(edge_gt_rgb, 0, 1)
            edge_pred_rgb = np.clip(edge_pred_rgb, 0, 1)

            # ===== CAM =====
            seg_heat = cv2.applyColorMap((seg_cam * 255).astype(np.uint8), cv2.COLORMAP_JET) / 255.0
            cls_heat = cv2.applyColorMap((cls_cam * 255).astype(np.uint8), cv2.COLORMAP_JET) / 255.0

            seg_overlay = np.clip(0.6 * img_rgb + 0.4 * seg_heat, 0, 1)
            cls_overlay = np.clip(0.6 * img_rgb + 0.4 * cls_heat, 0, 1)

            seg_overlay = draw_text(seg_overlay, "Seg-CAM")
            cls_overlay = draw_text(cls_overlay, "Cls-CAM")

            gt_rgb = draw_text(gt_rgb, "GT")
            pd_rgb = draw_text(pd_rgb, "Seg")

            edge_gt_rgb = draw_text(edge_gt_rgb, "Edge-GT")
            edge_pred_rgb = draw_text(edge_pred_rgb, "Edge-Pred")

            # ===== 分类文字写在原图 =====
            gt_label = cls[i].item()
            pd_label = pred_class[i].item()

            gt_name = class_names.get(gt_label, str(gt_label))
            pd_name = class_names.get(pd_label, str(pd_label))

            conf = cls_prob[i, pd_label].item()

            text = f"GT:{gt_name} | Pred:{pd_name}\n({conf:.2f})"
            color = (0, 255, 0) if gt_label == pd_label else (0, 0, 255)

            img_rgb = draw_text(img_rgb, text, color)

            # ===== 拼接 =====
            panels = [
                img_rgb,
                seg_overlay,
                cls_overlay,
                gt_rgb,
                pd_rgb,
                edge_gt_rgb,
                edge_pred_rgb
            ]

            concat = np.concatenate(panels, axis=1)

            save_name = name[i] if isinstance(name, (list, tuple)) else name
            save_path = os.path.join(save_dir, f"{save_name}.png")
            cv2.imwrite(save_path, (concat * 255).astype(np.uint8))
            


# =========================
# 主可视化函数
# =========================
def save_seg_visualization(model, val_loader, device, log_dir, epoch):

    model.eval()

    with torch.no_grad():

        # ===== 取一个batch =====
        val_iter = iter(val_loader)
        imgs, masks, *_ = next(val_iter)

        imgs = imgs.to(device)

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        masks = masks.float().to(device)

        # ===== forward =====
        seg_out = model(imgs)

        logits = F.interpolate(seg_out, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        pred_mask = (torch.sigmoid(logits) > 0.5).float()

        # ===== 取前几个 =====
        max_vis=2
        imgs_vis = imgs[:max_vis]
        masks_vis = masks[:max_vis]
        pred_vis = pred_mask[:max_vis]

        # ===== 反归一化 =====
        imgs_vis = imgs_vis * 0.5 + 0.5
        imgs_vis = torch.clamp(imgs_vis, 0, 1)

        # ===== normalize每张图 =====
        min_val = imgs_vis.amin(dim=(2,3), keepdim=True)
        max_val = imgs_vis.amax(dim=(2,3), keepdim=True)
        imgs_vis = (imgs_vis - min_val) / (max_val - min_val + 1e-6)

        # ===== CPU =====
        imgs_vis = imgs_vis.cpu()
        masks_vis = masks_vis.cpu()
        pred_vis = pred_vis.cpu()

        # ===== 原图 =====
        img_rgb = imgs_vis.repeat(1, 3, 1, 1)

        # ===== overlay =====
        gt_overlay = overlay_mask(imgs_vis, masks_vis, "red")
        pred_overlay = overlay_mask(imgs_vis, pred_vis, "green")

        # ===== overlap =====
        overlap = img_rgb.clone()
        overlap[:, 0] = torch.clamp(overlap[:, 0] + masks_vis.squeeze(1), 0, 1)
        overlap[:, 1] = torch.clamp(overlap[:, 1] + pred_vis.squeeze(1), 0, 1)

        # ===== 拼接 =====
        vis = torch.cat([
            img_rgb,
            gt_overlay,
            pred_overlay,
            overlap
        ], dim=3)

        # ===== 保存路径 =====
        save_dir = os.path.join(log_dir, "train")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"vis_epoch_{epoch}.png")

        vutils.save_image(vis, save_path, nrow=1, normalize=False)

        # ======================
        # 加标题
        # ======================
        img_pil = Image.open(save_path)
        draw = ImageDraw.Draw(img_pil)

        width, _ = img_pil.size
        block_w = width // 4

        titles = ["Image", "GT contour", "Pred contour", "Overlap"]

        # ===== 字体（带fallback）=====
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()

        for i, title in enumerate(titles):
            text_w, text_h = draw.textbbox((0, 0), title, font=font)[2:]
            x = block_w * i + (block_w - text_w) // 2
            y = 10

            draw.text(
                (x, y),
                title,
                fill=(255, 255, 0),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0)
            )

        img_pil.save(save_path)

        print(f"✅ Saved visualization: {save_path}")
        


from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils
import os

import torch
import torch.nn.functional as F

@torch.no_grad()
def get_hard_samples(model, loader, device, topk=2):

    model.eval()

    results = []  # (dice, img, mask, pred)

    for imgs, masks, *_ in loader:

        imgs = imgs.to(device)
 
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        masks = (masks > 0.5).float().to(device)

        logits= model(imgs)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        # 自动适配
        if logits.shape[1] == 1:
            preds = (torch.sigmoid(logits) > 0.5).float()
        else:
            preds = torch.argmax(logits, dim=1, keepdim=True).float()

        # ===== 逐张计算 Dice =====
        for i in range(imgs.size(0)):

            p = preds[i].view(-1)
            g = masks[i].view(-1)

            inter = (p * g).sum()
            union = p.sum() + g.sum()

            dice = (2 * inter + 1e-6) / (union + 1e-6)

            results.append({
                "dice": dice.item(),
                "img": imgs[i].cpu(),
                "mask": masks[i].cpu(),
                "pred": preds[i].cpu()
            })

    # ===== 排序（最差在前）=====
    results = sorted(results, key=lambda x: x["dice"])

    return results[:topk]

def save_hardcase_visualization(hard_samples, log_dir, epoch):

    save_dir = os.path.join(log_dir, "train")
    os.makedirs(save_dir, exist_ok=True)

    imgs = []
    dices = []

    for sample in hard_samples:

        img = sample["img"]
        mask = sample["mask"]
        pred = sample["pred"]
        dice = sample["dice"]


        img_rgb = img.repeat(3,1,1)

        # ===== overlay（边界版推荐）=====
        gt = torch.zeros_like(img_rgb)
        gt[0] = mask.squeeze(0)

        pd = torch.zeros_like(img_rgb)
        pd[1] = pred.squeeze(0)

        overlap = img_rgb.clone()
        overlap[0] = torch.clamp(overlap[0] + mask.squeeze(0), 0, 1)
        overlap[1] = torch.clamp(overlap[1] + pred.squeeze(0), 0, 1)

        concat = torch.cat([img_rgb, gt, pd, overlap], dim=2)

        imgs.append(concat)
        dices.append(dice)

    vis = torch.stack(imgs)

    save_path = os.path.join(save_dir, f"epoch_{epoch}.png")
    vutils.save_image(vis, save_path, nrow=1, normalize=False)

    # ======================
    # 加文字（标题 + Dice）
    # ======================
    img_pil = Image.open(save_path)
    draw = ImageDraw.Draw(img_pil)

    width, _ = img_pil.size
    block_w = width // 4

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()

    titles = ["Image", "GT", "Pred", "Overlap"]

    for i, title in enumerate(titles):

        bbox = draw.textbbox((0,0), title, font=font)
        text_w = bbox[2] - bbox[0]

        x = block_w * i + (block_w - text_w) // 2
        y = 5

        draw.text((x, y), title, fill=(255,255,255), font=font)

    # ===== 标 Dice（每一行）=====
    h_per_img = img_pil.height // len(dices)

    for i, d in enumerate(dices):

        text = f"Dice: {d:.3f}"

        draw.text(
            (10, i * h_per_img + 10),
            text,
            fill=(255,255,0),
            font=font,
            stroke_width=2,
            stroke_fill=(0,0,0)
        )

    img_pil.save(save_path)

    print(f"🔥 Hardcase Saved: {save_path}")
    
        

        
        
def overlay_mask(image, mask, color, alpha=0.5):
    """
    image: [B,1,H,W] or [B,3,H,W]  (原图)
    mask:  [B,1,H,W]  (0/1)
    color: "red" / "green"
    alpha: 透明度
    """

    # ===== 保证3通道 =====
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)

    image = image.clone()

    # ===== 创建颜色mask =====
    color_mask = torch.zeros_like(image)

    if color == "red":
        color_mask[:, 0] = mask.squeeze(1)
    elif color == "green":
        color_mask[:, 1] = mask.squeeze(1)

    # ===== overlay（关键）=====
    overlay = image * (1 - alpha) + color_mask * alpha

    return overlay



def init_loss_log(log_path: str, header: str = "epoch,train_loss\n") -> None:
    """
    初始化或清空训练损失日志文件，并写入表头。
    """
    with open(log_path, "w") as f:
        f.write(header)
        
def append_loss_log(log_path: str, epoch: int, train_loss: float) -> None:
    """
    向日志文件末尾追加一行训练损失数据。

    Args:
        log_path (str): 日志文件路径。
        epoch (int): 当前 epoch 编号（从 1 开始）。
        train_loss (float): 这一轮的训练平均损失。
    """
    # 'a' 模式：不存在则自动创建，存在则追加
    with open(log_path, "a") as f:
        f.write(f"{epoch},{train_loss:.6f}\n")
        
        


def dice_loss(pred, target, smooth=1e-6):
    """
    pred: [B, C, H, W] logits
    target: [B, C, H, W] binary mask (0或1)
    """
    # 1) 把 logits → 概率
    pred = torch.sigmoid(pred)

    # 2) 拉平到 [B, N]，N = C*H*W
    B = pred.shape[0]
    pred_flat   = pred.view(B, -1)
    target_flat = target.view(B, -1)

    # 3) 每个样本分别求交集和并集
    intersection = (pred_flat * target_flat).sum(dim=1)      # [B]
    union        = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]

    # 4) 计算每个样本的 Dice 系数，再转为损失
    dice_score = (2 * intersection + smooth) / (union + smooth)  # [B]
    loss = 1 - dice_score                                       # [B]

    # 5) 对 batch 取平均
    return loss.mean()



def dice_loss_per_sample(logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    计算每个样本的 Dice Loss，然后返回一个 [B] 的张量。
    
    Args:
        logits: 模型原始输出，shape [B, C, H, W]
        masks: 二值化的 ground-truth，shape [B, C, H, W]
        smooth: 平滑项，防止除零
    
    Returns:
        loss_per_sample: 每个样本的 Dice Loss，shape [B]
    """
    # 1) logits -> 概率
    probs = torch.sigmoid(logits)                      # [B, C, H, W]
    B = probs.shape[0]
    # 2) 展平到 [B, N]
    probs_flat = probs.view(B, -1)
    masks_flat = masks.view(B, -1)
    # 3) 计算交集和并集
    intersection = (probs_flat * masks_flat).sum(dim=1)       # [B]
    union        = probs_flat.sum(dim=1) + masks_flat.sum(dim=1)  # [B]
    # 4) 每个样本 Dice 系数 & Dice Loss
    dice_score = (2 * intersection + smooth) / (union + smooth)  # [B]
    loss_per_sample = 1 - dice_score                             # [B]
    return loss_per_sample





def visualize_with_labels(orig, probs, mask, save_path, n=4):
    """
    在 Matplotlib Canvas 上绘制 n 个样本，每个样本有三列：
      原图 | 概率图 | 二值掩码
    并在第一行加上文字 'Image','Probability','Mask'。

    Args:
        orig (Tensor): 原图，[B,1,H,W]，取值 [0,1]
        probs (Tensor): 预测概率图，[B,1,H,W]
        mask (Tensor): 二值化掩码，[B,1,H,W]
        save_path (str): 保存路径
        n (int): 展示样本数，<= B
    """
    B, _, H, W = orig.shape
    n = min(n, B)

    # 转成 numpy [n,H,W]
    orig_np = orig[:n].cpu().squeeze(1).numpy()
    probs_np = probs[:n].cpu().squeeze(1).numpy()
    # mask_np  = mask[:n].cpu().squeeze(1).numpy()

    # 创建 n 行 3 列的画板
    fig, axes = plt.subplots(n, 3, figsize=(3*3, n*3))
    if n == 1:
        axes = axes.reshape(1, -1)

    # 在第一行加上列标题
    col_titles = ["Image", "Probability"]#, "Mask"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=14)

    # 绘制每个样本
    for i in range(n):
        for j, arr in enumerate([orig_np[i], probs_np[i]]):#, mask_np[i]]):
            ax = axes[i, j]
            ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    



def plot_loss_curve(csv_path: str,
                    output_path: str = "loss_curve.png",
                    show_head: bool = True):
    """
    从指定的 CSV 文件读取训练/验证损失日志，并绘制 loss 曲线。

    Args:
        csv_path (str): CSV 文件路径，包含 epoch, train_loss, val_loss 三列。
        output_path (str): 保存生成的曲线图像的路径。
        show_head (bool): 是否在控制台打印 CSV 前五行。
    """
    # 1) 读取 CSV
    # df = pd.read_csv(csv_path, skiprows=8,header=0)
    df = pd.read_csv(csv_path)#, skiprows=8,header=0)

    # 2) 清理列名：去除首尾空格、BOM 等
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # 3) 检查必须列是否存在
    expected = ['epoch', 'train_loss', 'val_loss']
    missing = [col for col in expected if col not in df.columns]
    if missing:
        print("CSV 中实际的列名为:", df.columns.tolist())
        raise KeyError(f"在 CSV 中找不到以下列: {missing}")

    # 4) （可选）打印前几行
    if show_head:
        print(df.head())

    # 5) 绘制曲线
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.8)
    plt.title('Training & Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 6) 保存并显示
    plt.savefig(output_path, dpi=150)
    # plt.show()
    print(f"Saved loss curve to {output_path}")




def visualize_batch(
    img, mask, cls_gt, edge, recon,
    seg_main, cls_logit, edge_out, recon_out,
    class_names=['Normal', 'Benign', 'Malignant'],
    save_path=None
):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torchvision.transforms as T

    to_pil = T.ToPILImage()

    # ------------------- 1. 安全转 CPU（关键修复！）-------------------
    img       = img.cpu()
    mask      = mask.cpu()
    seg_main  = seg_main.cpu()
    edge      = edge.cpu()
    edge_out  = edge_out.cpu()
    recon     = recon.cpu()
    recon_out = recon_out.cpu().clamp(0, 1)

    # 关键修复：cls_logit 可能是 int，也可能是 tensor
    if isinstance(cls_logit, torch.Tensor):
        cls_pred = torch.argmax(cls_logit, dim=1).cpu()           # [B]
    else:
        cls_pred = torch.tensor([int(cls_logit)]).cpu()           # 你传的是 int

    if isinstance(cls_gt, torch.Tensor):
        cls_gt_vis = cls_gt.cpu()
    else:
        cls_gt_vis = torch.tensor([int(cls_gt)]).cpu()

    # ------------------- 2. 预测结果 -------------------
    pred_seg  = (torch.sigmoid(seg_main) > 0.5).float()
    pred_edge = (torch.sigmoid(edge_out) > 0.5).float()

    img_size = img.size(2)

    # ------------------- 3. 画布 -------------------
    col_count = 7
    canvas_w = 80 + img_size * col_count
    canvas_h = 140 + img_size
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ------------------- 4. 字体（防炸）-------------------
    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_text  = ImageFont.truetype("arial.ttf", 20)
    except:
        font_title = font_text = ImageFont.load_default()

    # ------------------- 5. 标题 -------------------
    titles = ["Input", "GT Mask", "Pred Seg", "GT Edge", "Pred Edge", "GT Recon", "Pred Recon"]
    for idx, title in enumerate(titles):
        x = 40 + idx * img_size + img_size // 2
        bbox = draw.textbbox((0, 0), title, font=font_title)
        w = bbox[2] - bbox[0]
        draw.text((x - w // 2, 15), title, fill=(0, 0, 0), font=font_title)

    # ------------------- 6. 图像粘贴 -------------------
    x_base, y_base = 40, 70

    # Input + CLAHE 增强
    input_raw = (img[0] * 0.5 + 0.5).clamp(0, 1)
    input_np = (input_raw.squeeze(0).numpy() * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(input_np)
    enhanced_tensor = torch.from_numpy(enhanced / 255.0).unsqueeze(0)
    canvas.paste(to_pil(enhanced_tensor), (x_base, y_base))

    # 其他图像
    imgs_to_paste = [
        mask[0].squeeze(0).unsqueeze(0),
        pred_seg[0].squeeze(0).unsqueeze(0),
        edge[0].squeeze(0).unsqueeze(0),
        pred_edge[0].squeeze(0).unsqueeze(0),
        recon[0].squeeze(0).unsqueeze(0),
        recon_out[0].squeeze(0).unsqueeze(0),
    ]

    for i, tensor in enumerate(imgs_to_paste):
        pil_img = to_pil(tensor.clamp(0, 1).repeat(3, 1, 1) if tensor.size(0) == 1 else tensor)
        canvas.paste(pil_img, (x_base + (i + 1) * img_size, y_base))

    # ------------------- 7. 分类文字 -------------------
    pred_label = class_names[cls_pred[0].item()]
    gt_label   = class_names[cls_gt_vis[0].item()]
    correct = cls_pred[0].item() == cls_gt_vis[0].item()
    color = (0, 180, 0) if correct else (200, 0, 0)

    y_text = y_base + img_size + 15
    draw.text((x_base + 10, y_text),     f"Pred: {pred_label}", fill=color,      font=font_text)
    draw.text((x_base + 10, y_text + 28), f"GT:   {gt_label}",   fill=(0,0,0),     font=font_text)

    # ------------------- 8. 保存 -------------------
    if save_path:
        canvas.save(save_path)
        status = "Correct" if correct else "Wrong"
        print(f" [Vis] Saved: {save_path} | {status} Pred: {pred_label} ← GT: {gt_label}")

    return canvas
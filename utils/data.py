import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import transforms
from tqdm import tqdm
import datetime
import csv


# ==================== 1. Dataset（完美版） ====================
class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=384, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.img_size  = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.image_list = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if len(self.image_list) == 0:
            raise ValueError(f"No images in {image_dir}")

        # 固定 resize（必须在 ToTensor 之前）
        self.resize_img  = T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")  # (H, W)

        # === 合并双 mask ===
        base_name, _ = os.path.splitext(img_name)
        mask_paths = [
            os.path.join(self.mask_dir, f"{base_name}_mask.png"),
            os.path.join(self.mask_dir, f"{base_name}_mask_1.png")
        ]
        mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
        found = False
        for p in mask_paths:
            if os.path.exists(p):
                m = np.array(Image.open(p).convert("L"))
                mask_np = np.bitwise_or(mask_np, m)
                found = True
        if not found:
            raise FileNotFoundError(f"Mask not found: {img_name}")
        mask = Image.fromarray(mask_np)

        # === 统一 Resize（关键！）===
        image = self.resize_img(image)
        mask  = self.resize_mask(mask)

        # === 转为 Tensor（只此一次！）===
        image = T.ToTensor()(image)      # (1, 384, 384) float32 [0,1]
        mask  = T.ToTensor()(mask)       # (1, 384, 384) float32 [0,1]
        mask  = (mask > 0.5).float()     # 严格 0/1

        # === 额外增强（只对 Tensor 有效，且 image/mask 同步）===
        if self.transform is not None:
            # 固定随机种子，保证 image 和 mask 增强一致
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)  # 注意：这里不能有 Normalize！
            
        name = os.path.basename(img_path)

        return image, mask.squeeze(0),name  # image: [1,384,384], mask: [384,384]
    
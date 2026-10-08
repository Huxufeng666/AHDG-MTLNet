import os
import csv
import cv2
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import BertTokenizer


# =========================
# 数据增强
# =========================
def augment(img, mask):
    # ===== 几何 =====
    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    if random.random() > 0.5:
        img = TF.vflip(img)
        mask = TF.vflip(mask)

    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

    # ===== scale =====
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        h, w = img.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)

        img = TF.resize(img, (new_h, new_w))
        mask = TF.resize(mask, (new_h, new_w))

        img = TF.resize(img, (h, w))
        mask = TF.resize(mask, (h, w))

    # ===== 超声噪声 =====
    if random.random() > 0.5:
        noise = torch.randn_like(img) * 0.1
        img = img + img * noise

    # ===== 亮度 =====
    if random.random() > 0.5:
        brightness = random.uniform(0.7, 1.3)
        img = img * brightness

    # ===== 对比度 =====
    if random.random() > 0.5:
        mean = img.mean()
        contrast = random.uniform(0.7, 1.3)
        img = (img - mean) * contrast + mean
        img = torch.clamp(img, 0, 1)

    # ===== Gamma =====
    if random.random() > 0.5:
        gamma = random.uniform(0.7, 1.5)
        img = torch.clamp(img, 0, 1)
        img = img ** gamma

    # ===== CLAHE =====
    if random.random() > 0.5:
        img_np = img.squeeze().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)

        img = torch.tensor(img_np / 255.0, dtype=torch.float32).unsqueeze(0)

    # ===== 边界扰动 =====
    if random.random() > 0.5:
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)

        if random.random() > 0.5:
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        else:
            mask_np = cv2.erode(mask_np, kernel, iterations=1)

        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

    img = torch.clamp(img, 0, 1)
    mask = (mask > 0.5).float()

    return img, mask


# =========================
# 辅助函数
# =========================
def _load_image(path: str) -> torch.Tensor:
    try:
        img = Image.open(path).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)   # [1,H,W]
    except Exception as e:
        raise IOError(f"无法加载图像: {path} -> {e}")


def _load_mask(path: str) -> torch.Tensor:
    try:
        mask = Image.open(path).convert("L")
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]
    except Exception as e:
        raise IOError(f"无法加载掩码: {path} -> {e}")


def _load_mask_from_pil(pil_img: Image.Image) -> torch.Tensor:
    mask = np.array(pil_img)
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def _create_edge(mask_np: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    grad_x = cv2.Sobel(mask_np, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(mask_np, cv2.CV_32F, 0, 1, ksize=kernel_size)
    edge = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return (edge > 0).astype(np.float32)


# =========================
# 主 Dataset
# =========================
class BUSI_Data(Dataset):
    def __init__(
        self,
        root_dir: str = "dataset",
        split: str = "train",
        img_size: int = 256,
        seed: int = 2025,
        train: bool = False,
        use_text: bool = True,
        bert_name: str = "bert-base-uncased",
        max_text_len: int = 32
    ):
        assert split in {"train", "val", "test"}, "split 必须是 train / val / test"

        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.train = train
        self.use_text = use_text
        self.max_text_len = max_text_len

        random.seed(seed)

        # ===== tokenizer =====
        self.tokenizer = BertTokenizer.from_pretrained(bert_name) if use_text else None

        # ===== 伪文本模板 =====
        self.text_templates = {
            0: [
                "Breast ultrasound image with no obvious tumor region.",
                "This is a normal breast ultrasound image without a visible lesion."
            ],
            1: [
                "Breast ultrasound image with a benign lesion and a relatively clear boundary.",
                "This image shows a benign breast tumor with a smooth and clear margin."
            ],
            2: [
                "Breast ultrasound image with a malignant lesion and an irregular blurry boundary.",
                "This image shows a malignant breast tumor with an unclear and irregular margin."
            ]
        }

        # ===== labels.csv =====
        label_csv = self.root / "labels.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"labels.csv 不存在: {label_csv}")
        self.label_dict = self._load_labels(label_csv)

        # ===== 图像路径 =====
        img_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")

        self.samples = self._collect_samples(img_dir, mask_dir)
        print(f"[{split.upper()}] 加载 {len(self.samples)} 张图像")

    # =========================
    # 内部函数
    # =========================
    def _load_labels(self, csv_path: Path) -> Dict[str, int]:
        d = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print(f"[DEBUG] CSV 列名: {reader.fieldnames}")

            class_col = None
            for col in ["class_id", "label", "cls", "category", "class"]:
                if col in reader.fieldnames:
                    class_col = col
                    break

            if class_col is None:
                raise KeyError(f"CSV 中未找到类别列，可用列: {reader.fieldnames}")

            if "image_path" not in reader.fieldnames:
                raise KeyError(f"CSV 中未找到 image_path 列，可用列: {reader.fieldnames}")

            for i, row in enumerate(reader):
                img_path = row["image_path"].strip()
                try:
                    cls_id = int(row[class_col].strip())
                except ValueError:
                    raise ValueError(f"第 {i+2} 行类别不是整数: {row[class_col]}")

                d[img_path] = cls_id

        print(f"[DEBUG] 成功加载 {len(d)} 条标签")
        return d

    def _collect_samples(self, img_dir: Path, mask_dir: Path) -> List[Dict]:
        samples = []

        for img_path in img_dir.glob("*.png"):
            filename = img_path.name

            if filename not in self.label_dict:
                print(f"警告: labels.csv 中找不到 {filename}")
                continue

            stem = img_path.stem
            possible_masks = [
                mask_dir / f"{stem}_mask.png",
                mask_dir / f"{stem}_mask_1.png",
                mask_dir / f"{stem}.png",
                mask_dir / img_path.name
            ]

            mask_path = None
            for mp in possible_masks:
                if mp.exists():
                    mask_path = str(mp).replace("\\", "/")
                    break

            if mask_path is None:
                print(f"警告: 找不到 mask -> {img_path}")
                continue

            samples.append({
                "image_path": str(img_path).replace("\\", "/"),
                "mask_path": mask_path,
                "class_id": self.label_dict[filename]
            })

        random.shuffle(samples)
        return samples

    def _build_pseudo_text(self, class_id: int) -> str:
        if class_id not in self.text_templates:
            return "Breast ultrasound image."

        templates = self.text_templates[class_id]

        if self.train:
            return random.choice(templates)
        return templates[0]

    # =========================
    # 核心
    # =========================
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # ===== image =====
        img = _load_image(sample["image_path"])   # [1,H,W]

        # ===== mask（兼容 _mask / _mask_1）=====
        base_name = Path(sample["image_path"]).stem
        mask_dir = Path(sample["image_path"]).parent.parent / "masks"

        mask1_path = mask_dir / f"{base_name}_mask.png"
        mask2_path = mask_dir / f"{base_name}_mask_1.png"

        mask1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE) if mask1_path.exists() else None
        mask2 = cv2.imread(str(mask2_path), cv2.IMREAD_GRAYSCALE) if mask2_path.exists() else None

        if mask1 is None and mask2 is None:

            mask = _load_mask(sample["mask_path"])
        else:
            if mask1 is not None and mask2 is not None:
                final_mask = np.maximum(mask1, mask2)
            elif mask1 is not None:
                final_mask = mask1
            else:
                final_mask = mask2

            final_mask_pil = Image.fromarray(final_mask)
            mask = _load_mask_from_pil(final_mask_pil)

        # ===== resize =====
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="nearest"
        ).squeeze(0)

        mask = (mask > 0.5).float()

        # ===== augment =====
        if self.train:
            img, mask = augment(img, mask)

        # ===== edge =====
        mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        edge_np = _create_edge(mask_np, kernel_size=3)
        edge = torch.from_numpy(edge_np).unsqueeze(0).float()

        # ===== recon / cls / name =====
        recon = img.clone()
        cls = torch.tensor(sample["class_id"], dtype=torch.long)
        name = base_name

        # ===== pseudo text =====
        if self.use_text:
            text = self._build_pseudo_text(sample["class_id"])

            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].squeeze(0)           # [L]
            attention_mask = encoded["attention_mask"].squeeze(0) # [L]

            return img, mask, cls, edge, recon, name, input_ids, attention_mask, text

        else:
            return img, mask, cls, edge, recon, name
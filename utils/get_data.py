import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF  





AUGMENT_PROFILES = {"current", "fixed", "simplified", "none"}


def _clamp_image(img: torch.Tensor) -> torch.Tensor:
    return torch.clamp(img, 0.0, 1.0)


def _rotate_pair(img: torch.Tensor, mask: torch.Tensor, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
    return img, (mask > 0.5).float()


def _scale_pair(img: torch.Tensor, mask: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = img.shape[-2:]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
    img = TF.resize(img, (h, w), interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (h, w), interpolation=TF.InterpolationMode.NEAREST)
    return _clamp_image(img), (mask > 0.5).float()


def _apply_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    img = img + img * torch.randn_like(img) * std
    return _clamp_image(img)


def _apply_brightness(img: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return _clamp_image(img * random.uniform(low, high))


def _apply_contrast(img: torch.Tensor, low: float, high: float) -> torch.Tensor:
    mean = img.mean()
    img = (img - mean) * random.uniform(low, high) + mean
    return _clamp_image(img)


def _apply_gamma(img: torch.Tensor, low: float, high: float) -> torch.Tensor:
    img = _clamp_image(img)
    return _clamp_image(img ** random.uniform(low, high))


def _apply_clahe(img: torch.Tensor) -> torch.Tensor:
    img = _clamp_image(img)
    img_np = img.squeeze().cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    img_np = (img_np * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np = clahe.apply(img_np)
    return torch.tensor(img_np / 255.0, dtype=torch.float32).unsqueeze(0)


def _apply_mask_edge_perturbation(mask: torch.Tensor) -> torch.Tensor:
    mask_np = (mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    if random.random() > 0.5:
        mask_np = cv2.dilate(mask_np, kernel, 1)
    else:
        mask_np = cv2.erode(mask_np, kernel, 1)
    return torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)


def augment(img, mask, cls_id: int, profile: str = "fixed"):
    profile = profile.lower()
    if profile not in AUGMENT_PROFILES:
        raise ValueError(f"Unknown augment profile: {profile}")
    if profile == "none":
        return _clamp_image(img), (mask > 0.5).float()

    img = _clamp_image(img)
    mask = (mask > 0.5).float()
    is_normal = int(cls_id) == 0
    is_lesion = not is_normal

    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    if profile == "current":
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            img, mask = _rotate_pair(img, mask, random.uniform(-15, 15))
        if random.random() > 0.5:
            img, mask = _scale_pair(img, mask, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            img = _apply_noise(img, std=0.05)
        if random.random() > 0.5:
            img = _apply_brightness(img, 0.85, 1.15)
        if random.random() > 0.5:
            img = _apply_contrast(img, 0.85, 1.15)
        if random.random() > 0.5:
            img = _apply_gamma(img, 0.85, 1.20)
        if random.random() > 0.5:
            img = _apply_clahe(img)
        return _clamp_image(img), (mask > 0.5).float()

    if profile == "simplified":
        if random.random() > 0.7:
            img, mask = _rotate_pair(img, mask, random.uniform(-8, 8))
        if random.random() > 0.75:
            img = _apply_gamma(img, 0.92, 1.10)
        if is_lesion and random.random() > 0.88:
            mask = _apply_mask_edge_perturbation(mask)
        return _clamp_image(img), (mask > 0.5).float()

    if random.random() > 0.65:
        img, mask = _rotate_pair(img, mask, random.uniform(-10, 10))
    if random.random() > 0.75:
        img = _apply_gamma(img, 0.90, 1.15)
    if random.random() > 0.8:
        img = _apply_brightness(img, 0.92, 1.08)
    if random.random() > 0.75:
        img = _apply_contrast(img, 0.92, 1.08)

    if is_normal:
        if random.random() > 0.65:
            img = _apply_noise(img, std=0.04)
        if random.random() > 0.7:
            img = _apply_contrast(img, 0.95, 1.10)
    if is_lesion:
        if random.random() > 0.82:
            mask = _apply_mask_edge_perturbation(mask)
        if random.random() > 0.82:
            img = _apply_noise(img, std=0.03)

    return _clamp_image(img), (mask > 0.5).float()




def _load_mask_from_pil(pil_img: Image.Image) -> torch.Tensor:
    """把 PIL.Image 转成和原来 _load_mask 一样的 tensor"""
    mask = np.array(pil_img)
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)

# ---------- 辅助函数 ----------
def _load_image(path: str) -> torch.Tensor:
    try:
        img = Image.open(path).convert("L")
        return torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
    except Exception as e:
        raise IOError(f"无法加载图像: {path} → {e}")


def _load_mask(path: str) -> torch.Tensor:
    try:
        mask = Image.open(path).convert("L")
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)
    except Exception as e:
        raise IOError(f"无法加载掩码: {path} → {e}")


def _create_edge(mask_np: np.ndarray, kernel_size=3) -> np.ndarray:
    grad_x = cv2.Sobel(mask_np, cv2.CV_32F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(mask_np, cv2.CV_32F, 0, 1, ksize=kernel_size)
    edge = np.sqrt(grad_x**2 + grad_y**2)
    return (edge > 0).astype(np.float32)


# ---------- 主 Dataset ----------
class BUSI_Data(Dataset):
    def __init__(self,
                 root_dir: str = "dataset",
                 split: str = "train",
                 img_size: int = 256,
                 seed: int = 2025,
                 train=False,
                 augment_profile: str = "fixed",
                 lesion_crop_prob: float = 0.0,
                 lesion_crop_margin: float = 0.35):

        assert split in {"train", "val", "test"}, f"split 必须是 train/val/test"
        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size
        random.seed(seed)
        self.train = train
        self.lesion_crop_prob = float(lesion_crop_prob)
        self.lesion_crop_margin = float(lesion_crop_margin)
        self.augment_profile = augment_profile.lower()
        if self.augment_profile not in AUGMENT_PROFILES:
            raise ValueError(f"Unknown augment profile: {self.augment_profile}")
        # 1. 读取 labels.csv
        label_csv = self.root / "labels.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"labels.csv 不存在: {label_csv}")
        self.label_dict = self._load_labels(label_csv)

        # 2. 收集图像
        img_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")

        self.samples = self._collect_samples(img_dir, mask_dir)
        print(f"[{split.upper()}] 加载 {len(self.samples)} 张图像")

    # ------------------- 内部工具 -------------------
    def _load_labels(self, csv_path: Path) -> Dict[str, int]:
        """支持多种列名: class_id, label, cls, category"""
        d = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print(f"[DEBUG] CSV 列名: {reader.fieldnames}")

            # 自动找 class_id 列
            class_col = None
            for col in ["class_id", "label", "cls", "category", "class"]:
                if col in reader.fieldnames:
                    class_col = col
                    break
            if class_col is None:
                raise KeyError(f"CSV 中未找到类别列! 可用: {reader.fieldnames}")

            for i, row in enumerate(reader):
                img_path = row["image_path"].strip()
                try:
                    cls_id = int(row[class_col].strip())
                except ValueError:
                    raise ValueError(f"第 {i+2} 行 class_id 不是整数: {row[class_col]}")
                d[img_path] = cls_id
        print(f"[DEBUG] 成功加载 {len(d)} 条标签")
        return d

    def _collect_samples(self, img_dir: Path, mask_dir: Path) -> List[Dict]:
        samples = []
        for img_path in img_dir.glob("*.png"):
            # img_path_str = str(img_path).replace("\\", "/")
            filename = img_path.name  # 只用文件名

            if filename not in self.label_dict:
                print(f"警告: labels.csv 中找不到: {filename}")
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
                print(f"警告: 找不到 mask: {img_path}")
                continue

            samples.append({
                "image_path": str(img_path),
                "mask_path": mask_path,
                "class_id": self.label_dict[filename]
            })

        random.shuffle(samples)
        return samples



    # ------------------- 核心 -------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _load_combined_mask(self, sample: Dict) -> np.ndarray:
        image_path = Path(sample["image_path"])
        base_name = image_path.stem
        mask_dir = image_path.parent.parent / "masks"
        candidates = [
            Path(sample["mask_path"]),
            mask_dir / f"{base_name}_mask.png",
            mask_dir / f"{base_name}_mask_1.png",
            mask_dir / image_path.name,
        ]
        loaded_masks = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                candidate_mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if candidate_mask is not None:
                    loaded_masks.append(candidate_mask)
        if not loaded_masks:
            raise FileNotFoundError(f"No mask found for {image_path.name}")
        combined = loaded_masks[0]
        for extra_mask in loaded_masks[1:]:
            combined = np.maximum(combined, extra_mask)
        return combined

    def get_mask_area_ratios(self) -> List[float]:
        """Original-resolution lesion ratios used by the balanced sampler."""
        return [float((self._load_combined_mask(s) > 0).mean()) for s in self.samples]

    def _random_lesion_crop(self, img: torch.Tensor, mask: torch.Tensor):
        """Crop around the lesion with context; called only for training cases."""
        foreground = torch.nonzero(mask[0] > 0.5, as_tuple=False)
        if foreground.numel() == 0:
            return img, mask
        y0, x0 = foreground.min(dim=0).values.tolist()
        y1, x1 = foreground.max(dim=0).values.tolist()
        lesion_h, lesion_w = y1 - y0 + 1, x1 - x0 + 1
        margin_y = max(8, int(round(lesion_h * self.lesion_crop_margin)))
        margin_x = max(8, int(round(lesion_w * self.lesion_crop_margin)))
        height, width = mask.shape[-2:]
        y0, y1 = max(0, y0 - margin_y), min(height, y1 + margin_y + 1)
        x0, x1 = max(0, x0 - margin_x), min(width, x1 + margin_x + 1)
        return img[:, y0:y1, x0:x1], mask[:, y0:y1, x0:x1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[idx]

        img = _load_image(sample["image_path"])

        image_path = Path(sample["image_path"])
        base_name = image_path.stem
        final_mask = self._load_combined_mask(sample)
        final_mask_pil = Image.fromarray(final_mask)
        mask = _load_mask_from_pil(final_mask_pil) 

        # Dual-scale training: each lesion is seen either as the full image or
        # as a contextual lesion crop. Validation/test always remain full-image.
        if self.train and self.lesion_crop_prob > 0 and random.random() < self.lesion_crop_prob:
            img, mask = self._random_lesion_crop(img, mask)

        # Resize
        resize = torch.nn.functional.interpolate
        img = resize(img.unsqueeze(0), size=(self.img_size, self.img_size),
                     mode="bilinear", align_corners=False).squeeze(0)
        mask = resize(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                      mode="nearest").squeeze(0)

        if self.train:
            img, mask = augment(img, mask, sample["class_id"], profile=self.augment_profile)

        mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        edge_np = _create_edge(mask_np,kernel_size=3)
        edge = torch.from_numpy(edge_np).unsqueeze(0).float()# / 255.0


        recon = img.clone()
        cls = torch.tensor(sample["class_id"], dtype=torch.long)
        
        name = os.path.basename(base_name)

        return img, mask, cls, edge, recon,name




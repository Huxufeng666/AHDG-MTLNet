import os
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================
# 你只需要改这里
# =========================================================
BUSI_ROOT = os.environ.get("BUSI_ROOT", "./data/BUSI")
NNUNET_RAW = os.environ.get("NNUNET_RAW", "./data/nnUNet_raw")

DATASET_ID = 1
DATASET_NAME = "BUSI"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

SEED = 2025

RESIZE = True
IMAGE_SIZE = 256

KEEP_NORMAL = True
CLEAR_OLD_DATASET = True


# =========================================================
# 基础函数
# =========================================================
def mkdir(path):
    os.makedirs(path, exist_ok=True)


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def save_png(path, array):
    ok = cv2.imwrite(str(path), array)
    if not ok:
        raise RuntimeError(f"Cannot save image: {path}")


def normalize_image_to_uint8(img):
    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


def binarize_mask(mask):
    return (mask > 0).astype(np.uint8)


def is_image_file(path):
    return path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def is_mask_file(path):
    name = path.stem.lower()
    return (
        "_mask" in name
        or "mask" in name
        or "_seg" in name
        or "segmentation" in name
    )


def clear_folder(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


# =========================================================
# 多 mask 查找与合并
# =========================================================
def find_all_masks_for_image(img_path):
    folder = img_path.parent
    stem = img_path.stem
    suffix = img_path.suffix

    masks = []

    candidate_patterns = [
        f"{stem}_mask{suffix}",
        f"{stem}_mask_1{suffix}",
        f"{stem}_mask_2{suffix}",
        f"{stem}_mask_3{suffix}",
        f"{stem}_mask_4{suffix}",
        f"{stem}_mask_5{suffix}",
        f"{stem}_seg{suffix}",
        f"{stem}_segmentation{suffix}",
        f"{stem} mask{suffix}",
    ]

    for pattern in candidate_patterns:
        p = folder / pattern
        if p.exists():
            masks.append(p)

    for p in folder.iterdir():
        if not is_image_file(p):
            continue

        p_stem = p.stem.lower()
        stem_lower = stem.lower()

        if stem_lower in p_stem and "mask" in p_stem:
            if p not in masks:
                masks.append(p)

    return sorted(masks, key=lambda x: x.name)


def merge_masks(mask_paths, target_size=None):
    if len(mask_paths) == 0:
        raise ValueError("mask_paths is empty")

    merged = None

    for mask_path in mask_paths:
        mask = read_gray(mask_path)
        mask = binarize_mask(mask)

        if target_size is not None:
            mask = cv2.resize(
                mask,
                target_size,
                interpolation=cv2.INTER_NEAREST
            )
            mask = binarize_mask(mask)

        if merged is None:
            merged = mask
        else:
            merged = np.logical_or(merged > 0, mask > 0).astype(np.uint8)

    return binarize_mask(merged)


# =========================================================
# 扫描 BUSI 数据
# =========================================================
def find_busi_cases(busi_root):
    busi_root = Path(busi_root)

    class_folders = ["benign", "malignant", "normal"]
    cases = []

    for cls_name in class_folders:
        cls_dir = busi_root / cls_name

        if not cls_dir.exists():
            print(f"[Warning] Missing folder: {cls_dir}")
            continue

        files = sorted([p for p in cls_dir.iterdir() if is_image_file(p)])

        for img_path in files:
            if is_mask_file(img_path):
                continue

            if cls_name == "normal" and not KEEP_NORMAL:
                continue

            mask_paths = find_all_masks_for_image(img_path)

            if len(mask_paths) == 0:
                if cls_name == "normal":
                    cases.append({
                        "image": img_path,
                        "masks": [],
                        "class": cls_name
                    })
                else:
                    print(f"[Skip] Missing mask for: {img_path}")
                    continue
            else:
                cases.append({
                    "image": img_path,
                    "masks": mask_paths,
                    "class": cls_name
                })

    return cases


# =========================================================
# 分层划分 train / val / test
# =========================================================
def stratified_split(cases):
    random.seed(SEED)

    by_class = {}
    for item in cases:
        cls_name = item["class"]
        by_class.setdefault(cls_name, []).append(item)

    train_cases = []
    val_cases = []
    test_cases = []

    for cls_name, cls_cases in by_class.items():
        random.shuffle(cls_cases)

        n = len(cls_cases)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        cls_train = cls_cases[:n_train]
        cls_val = cls_cases[n_train:n_train + n_val]
        cls_test = cls_cases[n_train + n_val:]

        train_cases.extend(cls_train)
        val_cases.extend(cls_val)
        test_cases.extend(cls_test)

        print(
            f"{cls_name}: total={n}, "
            f"train={len(cls_train)}, "
            f"val={len(cls_val)}, "
            f"test={len(cls_test)}"
        )

    random.shuffle(train_cases)
    random.shuffle(val_cases)
    random.shuffle(test_cases)

    return train_cases, val_cases, test_cases


# =========================================================
# 转换单个 case
# =========================================================
def convert_case(img_path, mask_paths, out_img_path, out_mask_path=None):
    img = read_gray(img_path)
    img = normalize_image_to_uint8(img)

    if RESIZE:
        img = cv2.resize(
            img,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

    save_png(out_img_path, img)

    if out_mask_path is not None:
        if len(mask_paths) == 0:
            mask = np.zeros_like(img, dtype=np.uint8)
        else:
            if RESIZE:
                mask = merge_masks(
                    mask_paths,
                    target_size=(IMAGE_SIZE, IMAGE_SIZE)
                )
            else:
                mask = merge_masks(mask_paths)

        mask = binarize_mask(mask)
        save_png(out_mask_path, mask)


# =========================================================
# dataset.json
# =========================================================
def create_dataset_json(dataset_dir, num_training):
    dataset_json = {
        "channel_names": {
            "0": "Ultrasound"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": int(num_training),
        "file_ending": ".png"
    }

    json_path = os.path.join(dataset_dir, "dataset.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Saved dataset.json: {json_path}")


# =========================================================
# splits_final.json
# =========================================================
def create_splits_final(dataset_dir, train_ids, val_ids):
    """
    nnU-Net 会读取：
    nnUNet_preprocessed/Dataset001_BUSI/splits_final.json

    但是我们先保存在 raw 目录下，预处理后再复制过去。
    """
    splits = [
        {
            "train": train_ids,
            "val": val_ids
        }
    ]

    split_path = os.path.join(dataset_dir, "splits_final.json")

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    print(f"Saved split file: {split_path}")


# =========================================================
# 保存 test 对照表
# =========================================================
def save_case_mapping(dataset_dir, mapping):
    mapping_path = os.path.join(dataset_dir, "case_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    print(f"Saved case mapping: {mapping_path}")


# =========================================================
# 主函数
# =========================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    dataset_dir = Path(NNUNET_RAW) / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    imagesTs = dataset_dir / "imagesTs"
    labelsTs = dataset_dir / "labelsTs"

    if dataset_dir.exists() and CLEAR_OLD_DATASET:
        print(f"[Clear] Old dataset folder: {dataset_dir}")
        clear_folder(dataset_dir)

    mkdir(imagesTr)
    mkdir(labelsTr)
    mkdir(imagesTs)
    mkdir(labelsTs)

    cases = find_busi_cases(BUSI_ROOT)

    if len(cases) == 0:
        raise RuntimeError("No valid BUSI cases found.")

    print("=" * 80)
    print(f"Total cases: {len(cases)}")

    train_cases, val_cases, test_cases = stratified_split(cases)

    print("=" * 80)
    print(f"Final split:")
    print(f"Train: {len(train_cases)}")
    print(f"Val  : {len(val_cases)}")
    print(f"Test : {len(test_cases)}")
    print("=" * 80)

    train_ids = []
    val_ids = []
    mapping = {}

    # =========================================================
    # Train cases -> imagesTr / labelsTr
    # =========================================================
    all_tr_val_cases = []

    for idx, item in enumerate(train_cases):
        case_id = f"BUSI_TR_{idx:04d}"
        train_ids.append(case_id)
        all_tr_val_cases.append((case_id, item, "train"))

    for idx, item in enumerate(val_cases):
        case_id = f"BUSI_VAL_{idx:04d}"
        val_ids.append(case_id)
        all_tr_val_cases.append((case_id, item, "val"))

    for case_id, item, split_name in tqdm(all_tr_val_cases, desc="Converting Train+Val"):
        out_img_path = imagesTr / f"{case_id}_0000.png"
        out_mask_path = labelsTr / f"{case_id}.png"

        convert_case(
            img_path=item["image"],
            mask_paths=item["masks"],
            out_img_path=out_img_path,
            out_mask_path=out_mask_path
        )

        mapping[case_id] = {
            "split": split_name,
            "class": item["class"],
            "source_image": str(item["image"]),
            "source_masks": [str(m) for m in item["masks"]]
        }

    # =========================================================
    # Test cases -> imagesTs / labelsTs
    # =========================================================
    for idx, item in enumerate(tqdm(test_cases, desc="Converting Test")):
        case_id = f"BUSI_TS_{idx:04d}"

        out_img_path = imagesTs / f"{case_id}_0000.png"
        out_mask_path = labelsTs / f"{case_id}.png"

        convert_case(
            img_path=item["image"],
            mask_paths=item["masks"],
            out_img_path=out_img_path,
            out_mask_path=out_mask_path
        )

        mapping[case_id] = {
            "split": "test",
            "class": item["class"],
            "source_image": str(item["image"]),
            "source_masks": [str(m) for m in item["masks"]]
        }

    create_dataset_json(
        dataset_dir=str(dataset_dir),
        num_training=len(train_ids) + len(val_ids)
    )

    create_splits_final(
        dataset_dir=str(dataset_dir),
        train_ids=train_ids,
        val_ids=val_ids
    )

    save_case_mapping(
        dataset_dir=str(dataset_dir),
        mapping=mapping
    )

    print("\nDone.")
    print(f"Dataset saved to: {dataset_dir}")

    print("\nImportant:")
    print("imagesTr contains train + val cases")
    print("labelsTr contains train + val masks")
    print("imagesTs contains test images")
    print("labelsTs contains test masks for your own evaluation")
    print("splits_final.json defines fixed train/val split")


if __name__ == "__main__":
    main()
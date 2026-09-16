import sys
import os
import time
import json
import random
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tatar_ocr_augmentation import (
    TATAR_UPPERCASE,
    get_available_fonts,
    render_base_char,
    generate_augmented_cell_image,
    apply_elastic_wobble,
)
from tatar_ocr_dataset import TatarOCRNet


# ---------------------------------------------------------------------------
# Advanced Real Handwriting Augmentation Pipeline
# ---------------------------------------------------------------------------

def advanced_augment_real_cell(img_arr, is_train=True):
    """
    Applies real-world multi-stage handwriting perturbations:
    1. Perspective shear & affine rotation (-8° to +8°, scale 0.90 to 1.08, shift)
    2. Morphological dilation / erosion (thin gel pen vs thick ballpoint / marker)
    3. Cell margin noise / border line injection
    4. Elastic muscle tremor / wobble
    5. Cutout / ink-skip starvation (partial dry stroke)
    6. Camera defocus / blur
    7. Dynamic contrast & lighting stretch
    """
    if not is_train:
        p_lo = np.percentile(img_arr, 2)
        p_hi = np.percentile(img_arr, 98)
        norm = np.clip((img_arr.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)
        return norm

    h, w = img_arr.shape
    center = (w / 2.0, h / 2.0)

    # 1. Perspective shear, rotation, scale, shift
    angle = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.92, 1.08)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    shear = random.uniform(-6.0, 6.0)
    rot_mat[0, 1] += np.tan(np.deg2rad(shear)) * rot_mat[0, 0]
    rot_mat[0, 2] += random.uniform(-3.0, 3.0)
    rot_mat[1, 2] += random.uniform(-3.0, 3.0)

    aug = cv2.warpAffine(img_arr, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # 2. Morphological dilation or erosion
    r_morph = random.random()
    if r_morph < 0.30:
        # Darken / thicken strokes (simulate marker / heavy pressure)
        k = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT if k == 2 else cv2.MORPH_ELLIPSE, (k, k))
        aug = cv2.erode(aug, kernel)
    elif r_morph < 0.50:
        # Thin out strokes (simulate thin ballpoint / fast writing)
        k = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        aug = cv2.dilate(aug, kernel)

    # 3. Cell margin noise / border line injection
    if random.random() < 0.40:
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        line_col = random.randint(55, 175)
        thick = random.choice([1, 2])
        if edge == 'top':
            y = random.randint(0, 3)
            cv2.line(aug, (0, y), (w - 1, y), line_col, thick)
        elif edge == 'bottom':
            y = random.randint(h - 4, h - 1)
            cv2.line(aug, (0, y), (w - 1, y), line_col, thick)
        elif edge == 'left':
            x = random.randint(0, 3)
            cv2.line(aug, (x, 0), (x, h - 1), line_col, thick)
        elif edge == 'right':
            x = random.randint(w - 4, w - 1)
            cv2.line(aug, (x, 0), (x, h - 1), line_col, thick)

    # 4. Elastic muscle tremor / wobble
    if random.random() < 0.35:
        aug = apply_elastic_wobble(aug, alpha=random.uniform(2.5, 5.5), sigma=random.uniform(2.5, 4.0))

    # 5. Cutout / ink-skip starvation (partial dry pen stroke)
    if random.random() < 0.25:
        cx = random.randint(8, w - 8)
        cy = random.randint(8, h - 8)
        rx = random.randint(3, 7)
        ry = random.randint(3, 7)
        cv2.circle(aug, (cx, cy), rx, 245, -1)

    # 6. Defocus blur
    if random.random() < 0.25:
        aug = cv2.GaussianBlur(aug, (3, 3), random.uniform(0.4, 0.8))

    # 7. Contrast & brightness stretch
    p_lo = np.percentile(aug, 2)
    p_hi = np.percentile(aug, 98)
    norm = np.clip((aug.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + random.uniform(5.0, 15.0), 0, 255).astype(np.uint8)
    return norm


class AdvancedRealDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_arr, char, cls_idx = self.samples[idx]
        aug_arr = advanced_augment_real_cell(img_arr, is_train=self.is_train)
        tensor = torch.from_numpy(aug_arr.astype(np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, cls_idx


def load_real_uppercase_dataset(metadata_csv="realdataset/metadata.csv", val_ratio=0.15, seed=42):
    base_dir = Path(metadata_csv).parent
    with open(metadata_csv, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f if line.strip()]
    header, rows = lines[0], lines[1:]

    by_char = {}
    for r in rows:
        ch = r[1]
        norm_path = base_dir / r[6]
        if ch not in by_char:
            by_char[ch] = []
        by_char[ch].append(norm_path)

    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for ch in TATAR_UPPERCASE:
        paths = by_char.get(ch, [])
        cls_idx = TATAR_UPPERCASE.index(ch)
        rng.shuffle(paths)

        n_val = max(1, int(len(paths) * val_ratio))
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]

        for p in train_paths:
            im = np.array(Image.open(p).convert("L"))
            train_samples.append((im, ch, cls_idx))

        for p in val_paths:
            im = np.array(Image.open(p).convert("L"))
            val_samples.append((im, ch, cls_idx))

    return train_samples, val_samples


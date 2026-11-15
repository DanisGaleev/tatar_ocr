"""
v3.dataset
High-throughput domain-aligned dataset loaders and confusion-aware loss for v3.
Strictly zero data leakage: benchmark/images/blank_*.jpg are NEVER touched in training.
"""

import os
import sys
import copy
import random
from pathlib import Path
from typing import Tuple, List, Dict

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tatar_ocr_augmentation import (
    TATAR_UPPERCASE,
    get_available_fonts,
    render_base_char,
    generate_augmented_cell_image,
    normalize_real_cell_like_scanner,
    apply_elastic_wobble,
)
from finetune_advanced_pipeline import advanced_augment_real_cell
from train_tatar_ocr_v2 import CONFUSION_GROUPS, HARD_CHARS


class ConfusionAwareLoss(nn.Module):
    """
    Label-smoothed cross-entropy combined with rival suppression loss
    on known Tatar handwritten confusion pairs.
    """
    def __init__(self, chars=TATAR_UPPERCASE, smoothing: float = 0.03, rival_weight: float = 0.45):
        super().__init__()
        self.chars = chars
        self.num_classes = len(chars)
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.rival_weight = rival_weight

        rival_mask = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        for ch, rivals in CONFUSION_GROUPS.items():
            if ch in chars:
                y_idx = chars.index(ch)
                for r_ch in rivals:
                    if r_ch in chars:
                        r_idx = chars.index(r_ch)
                        rival_mask[y_idx, r_idx] = 1.0
        self.register_buffer("rival_mask", rival_mask)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        batch_mask = self.rival_mask[targets]
        rival_probs = (probs * batch_mask).sum(dim=1).clamp(max=0.999)
        rival_loss = -torch.log(1.0 - rival_probs).mean()
        return ce_loss + self.rival_weight * rival_loss


def augment_scanner_cell(norm_64: np.ndarray, is_train: bool = True) -> np.ndarray:
    """Applies realistic variations while preserving 48px occupancy in 64x64 canvas."""
    if not is_train:
        return norm_64.copy()

    h, w = norm_64.shape
    aug = norm_64.copy()

    angle = random.uniform(-6.0, 6.0)
    shear = random.uniform(-6.0, 6.0)
    scale = random.uniform(0.93, 1.05)
    aspect = random.uniform(0.88, 1.14)
    sx = scale * np.sqrt(aspect)
    sy = scale / np.sqrt(aspect)

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 0] *= sx
    rot_mat[0, 1] *= sy
    rot_mat[1, 0] *= sx
    rot_mat[1, 1] *= sy
    rot_mat[0, 1] += np.tan(np.deg2rad(shear)) * rot_mat[0, 0]
    rot_mat[0, 2] = center[0] - (rot_mat[0, 0] * center[0] + rot_mat[0, 1] * center[1]) + random.uniform(-1.5, 1.5)
    rot_mat[1, 2] = center[1] - (rot_mat[1, 0] * center[0] + rot_mat[1, 1] * center[1]) + random.uniform(-1.5, 1.5)

    aug = cv2.warpAffine(aug, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=250)

    # Thickness variation
    r_m = random.random()
    if r_m < 0.22:
        aug = cv2.erode(aug, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    elif r_m < 0.40:
        aug = cv2.dilate(aug, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # Mild wobble
    if random.random() < 0.25:
        aug = apply_elastic_wobble(aug, alpha=random.uniform(1.8, 3.8), sigma=3.2)

    # Re-normalize contrast
    p_lo, p_hi = np.percentile(aug, 2), np.percentile(aug, 98)
    if p_hi - p_lo > 10:
        aug = np.clip((aug.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 250).astype(np.uint8)

    return aug


class RealDatasetV3(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, str, int]], is_train: bool = True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scanner_64, raw_64, ch, cls_idx = self.samples[idx]
        if self.is_train:
            if random.random() < 0.75:
                img = augment_scanner_cell(scanner_64, is_train=True)
            else:
                img = advanced_augment_real_cell(raw_64, is_train=True)
        else:
            img = scanner_64.copy()

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, cls_idx


class SyntheticDatasetV3(Dataset):
    def __init__(self, chars=TATAR_UPPERCASE, epoch_size: int = 5000, is_train: bool = True, seed: int = 42):
        self.chars = chars
        self.num_classes = len(chars)
        self.epoch_size = epoch_size
        self.is_train = is_train

        fonts_dict, font_weights = get_available_fonts()
        self.fonts = fonts_dict
        self.font_names = list(fonts_dict.keys())
        w = np.array([font_weights[k] for k in self.font_names], dtype=np.float32)
        self.font_probs = w / w.sum()

        c_weights = np.array([2.5 if ch in HARD_CHARS else 1.0 for ch in self.chars], dtype=np.float32)
        self.class_probs = c_weights / c_weights.sum()

        self.glyph_cache = {}
        for ch in self.chars:
            for fname, fpath in self.fonts.items():
                try:
                    self.glyph_cache[(ch, fname)] = render_base_char(ch, fpath, canvas_size=128)
                except Exception:
                    self.glyph_cache[(ch, fname)] = np.full((128, 128), 255, dtype=np.uint8)

        if not self.is_train:
            rng = np.random.RandomState(seed)
            self.fixed_val = []
            for i in range(epoch_size):
                cls_idx = i % self.num_classes
                ch = self.chars[cls_idx]
                fname = rng.choice(self.font_names, p=self.font_probs)
                fpath = self.fonts[fname]
                arr = generate_augmented_cell_image(
                    char=ch,
                    font_path=fpath,
                    canvas_size=64,
                    elastic_strength=10.0,
                    pressure_variation=1.3,
                    enable_overwriting=False,
                    enable_flourishes=True,
                    enable_cell_borders=True,
                    enable_paper_texture=True,
                    enable_camera_blur=True,
                    base_img=self.glyph_cache.get((ch, fname)),
                    enable_handwritten_variants=True,
                    align_to_scanner=True,
                )
                self.fixed_val.append((arr, cls_idx))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if not self.is_train:
            arr, cls_idx = self.fixed_val[idx]
        else:
            cls_idx = int(np.random.choice(self.num_classes, p=self.class_probs))
            ch = self.chars[cls_idx]
            fname = np.random.choice(self.font_names, p=self.font_probs)
            fpath = self.fonts[fname]
            base_img = self.glyph_cache.get((ch, fname))

            arr = generate_augmented_cell_image(
                char=ch,
                font_path=fpath,
                canvas_size=64,
                elastic_strength=random.uniform(6.0, 18.0),
                pressure_variation=random.uniform(1.0, 2.0),
                enable_overwriting=random.random() < 0.15,
                enable_flourishes=random.random() < 0.28,
                enable_cell_borders=random.random() < 0.50,
                enable_paper_texture=True,
                enable_camera_blur=True,
                base_img=base_img,
                enable_handwritten_variants=True,
                align_to_scanner=True,
            )

        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, cls_idx


def load_real_samples(metadata_csv: str = "realdataset/metadata.csv", val_ratio: float = 0.15, seed: int = 42):
    base_dir = Path(metadata_csv).parent
    with open(metadata_csv, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f if line.strip()][1:]

    by_char = {}
    for r in lines:
        ch = r[1]
        raw_100 = base_dir / r[5]
        norm_64 = base_dir / r[6]
        by_char.setdefault(ch, []).append((raw_100, norm_64))

    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for ch in TATAR_UPPERCASE:
        paths = by_char.get(ch, [])
        cls_idx = TATAR_UPPERCASE.index(ch)
        rng.shuffle(paths)

        n_val = max(1, int(len(paths) * val_ratio))
        val_list = paths[:n_val]
        train_list = paths[n_val:]

        for p100, p64 in train_list:
            im64 = np.array(Image.open(p64).convert("L"))
            im100 = np.array(Image.open(p100).convert("L")) if p100.exists() else im64
            scanner_64 = normalize_real_cell_like_scanner(im100, canvas_size=64)
            train_samples.append((scanner_64, im64, ch, cls_idx))
            if ch in HARD_CHARS:
                train_samples.append((scanner_64, im64, ch, cls_idx))

        for p100, p64 in val_list:
            im64 = np.array(Image.open(p64).convert("L"))
            im100 = np.array(Image.open(p100).convert("L")) if p100.exists() else im64
            scanner_64 = normalize_real_cell_like_scanner(im100, canvas_size=64)
            val_samples.append((scanner_64, im64, ch, cls_idx))

    return train_samples, val_samples

"""
Tatar OCR Dataset & Online Augmentation Pipeline
Supports both:
1. Online On-The-Fly Augmentation (Zero disk footprint, infinite diversity, optimal for PyTorch training)
2. High-speed Multi-threaded Batch Dataset Generator (Exports 15,000 - 20,000 PNG images with CSV/JSON manifests)
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tatar_ocr_augmentation import (
    TATAR_UPPERCASE,
    TATAR_LOWERCASE,
    PUNCTUATION_CHARS,
    DIGIT_CHARS,
    ALL_CHARS,
    CHAR_TO_ID,
    ID_TO_CHAR,
    get_available_fonts,
    generate_augmented_cell_image,
    render_base_char
)


class TatarOnlineAugmentedDataset(Dataset):
    """
    High-Performance PyTorch Dataset performing on-the-fly realistic handwriting augmentation.
    Key Advantages over static disk datasets:
    1. Infinite Variety: Every epoch generates completely new perturbations (wobbles, pressure, overwriting, borders).
    2. Zero Disk Footprint: Avoids writing tens of thousands of tiny PNG files to disk.
    3. Memory Cached: Base clean character glyphs for all fonts are pre-cached in RAM.
    """

    def __init__(
        self,
        chars=None,
        canvas_size=64,
        epoch_size=18360,
        font_paths=None,
    ):
        super().__init__()
        self.chars = chars if chars is not None else ALL_CHARS
        self.canvas_size = canvas_size
        self.epoch_size = epoch_size
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}

        all_fonts, font_weights = get_available_fonts()
        if font_paths:
            self.font_paths = font_paths
            self.font_weights = [1.0] * len(font_paths)
        else:
            self.font_paths = list(all_fonts.values())
            self.font_weights = [font_weights.get(Path(p).stem, 1.0) for p in self.font_paths]

        # Pre-cache clean base glyphs in memory at 2x resolution: dict of (char_idx, font_idx) -> np.uint8 array
        self.base_glyph_cache = {}
        self._warmup_glyph_cache()

    def _warmup_glyph_cache(self):
        """Renders raw clean glyphs for each char and font once into memory."""
        internal_size = max(128, self.canvas_size * 2)
        for c_idx, ch in enumerate(self.chars):
            for f_idx, fp in enumerate(self.font_paths):
                try:
                    arr = render_base_char(ch, fp, canvas_size=internal_size)
                except Exception:
                    arr = np.full((internal_size, internal_size), 255, dtype=np.uint8)
                self.base_glyph_cache[(c_idx, f_idx)] = arr

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Balanced class indexing: cycles evenly through all target classes
        char_idx = idx % len(self.chars)
        target_char = self.chars[char_idx]

        # Weighted font selection
        font_idx = random.choices(range(len(self.font_paths)), weights=self.font_weights, k=1)[0]
        font_path = self.font_paths[font_idx]

        # Retrieve pre-cached clean glyph to eliminate repeated font rasterization overhead
        base_img = self.base_glyph_cache.get((char_idx, font_idx))

        # Generate realistic augmented cell image
        img_arr = generate_augmented_cell_image(
            char=target_char,
            font_path=font_path,
            canvas_size=self.canvas_size,
            elastic_strength=random.uniform(8.0, 24.0),
            pressure_variation=random.uniform(1.0, 2.5),
            enable_overwriting=random.random() < 0.20,
            enable_flourishes=random.random() < 0.40,
            enable_cell_borders=random.random() < 0.70,
            enable_paper_texture=True,
            enable_camera_blur=random.random() < 0.40,
            base_img=base_img,
        )

        # Normalize to PyTorch Tensor: [1, H, W] in [-1.0, 1.0]
        norm_arr = img_arr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(norm_arr).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, char_idx


class TatarDiskDataset(Dataset):
    """PyTorch Dataset loader for saved synthetic or real Tatar character images."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        meta_file = self.root_dir / "metadata.csv"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")[1:]
                for line in lines:
                    if line.strip():
                        parts = line.strip().split(",")
                        self.samples.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img_arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_arr).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, label


class TatarOCRNet(nn.Module):
    """
    Optimized Convolutional Neural Network for Tatar handwritten character recognition in cells.
    Designed for 64x64 grayscale cell images, fast training, and low-latency mobile inference.
    """

    def __init__(self, num_classes=102):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3: 16x16 -> 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def generate_batch_dataset_to_disk(
    out_dir="dataset/tatar_hw_dataset",
    samples_per_class=180,
    img_size=64,
    include_upper=True,
    include_lower=True,
    include_punct=True,
    include_digits=True,
    max_workers=4,
):
    """
    High-speed multi-threaded batch generator exporting 15k - 20k balanced samples to disk.
    Writes class_index.json and metadata.csv.
    """
    selected_chars = []
    if include_upper:
        selected_chars.extend(TATAR_UPPERCASE)
    if include_lower:
        selected_chars.extend(TATAR_LOWERCASE)
    if include_punct:
        selected_chars.extend(PUNCTUATION_CHARS)
    if include_digits:
        selected_chars.extend(DIGIT_CHARS)

    total_expected = len(selected_chars) * samples_per_class
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_fonts, font_weights = get_available_fonts()
    font_keys = list(all_fonts.keys())
    weights_list = [font_weights[k] for k in font_keys]
    font_paths = [all_fonts[k] for k in font_keys]

    class_map = {ch: idx for idx, ch in enumerate(selected_chars)}
    with open(out_path / "class_index.json", "w", encoding="utf-8") as f:
        json.dump({str(idx): ch for ch, idx in class_map.items()}, f, ensure_ascii=False, indent=2)

    print(f"Starting batch generation: {len(selected_chars)} classes * {samples_per_class} = {total_expected} images")
    start_time = time.time()

    # Pre-render base clean glyphs once into memory for massive speedup
    internal_size = max(128, img_size * 2)
    glyph_cache = {}
    for ch in selected_chars:
        for fp in font_paths:
            try:
                glyph_cache[(ch, fp)] = render_base_char(ch, fp, canvas_size=internal_size)
            except Exception:
                glyph_cache[(ch, fp)] = np.full((internal_size, internal_size), 255, dtype=np.uint8)

    def generate_single_sample(args):
        ch, cls_id, s_idx, char_folder = args
        chosen_font = random.choices(font_paths, weights=weights_list, k=1)[0]
        base_img = glyph_cache.get((ch, chosen_font))
        arr = generate_augmented_cell_image(
            char=ch,
            font_path=chosen_font,
            canvas_size=img_size,
            elastic_strength=random.uniform(8.0, 24.0),
            pressure_variation=random.uniform(1.0, 2.5),
            enable_overwriting=random.random() < 0.20,
            enable_flourishes=random.random() < 0.40,
            enable_cell_borders=random.random() < 0.70,
            enable_paper_texture=True,
            enable_camera_blur=random.random() < 0.40,
            base_img=base_img,
        )
        img_filename = f"char_{cls_id:03d}_{s_idx:04d}.png"
        full_img_path = char_folder / img_filename
        Image.fromarray(arr).save(full_img_path)
        return f"{full_img_path.as_posix()},{cls_id},{ord(ch)}"

    tasks = []
    for ch in selected_chars:
        cls_id = class_map[ch]
        char_folder = out_path / f"cls_{cls_id:03d}"
        char_folder.mkdir(exist_ok=True)
        for s_idx in range(samples_per_class):
            tasks.append((ch, cls_id, s_idx, char_folder))

    metadata_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for record in executor.map(generate_single_sample, tasks):
            metadata_records.append(record)

    with open(out_path / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("image_path,class_id,unicode_ord\n" + "\n".join(metadata_records) + "\n")

    elapsed = time.time() - start_time
    print(f"Generation Complete! Saved {len(metadata_records)} images to {out_path.resolve().as_posix()} in {elapsed:.2f}s ({len(metadata_records)/elapsed:.1f} img/s)")
    return len(metadata_records), elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tatar OCR Dataset & Generator")
    parser.add_argument("--generate", action="store_true", help="Generate dataset to disk")
    parser.add_argument("--samples", type=int, default=180, help="Samples per class (180 * 102 = 18,360)")
    parser.add_argument("--out-dir", type=str, default="dataset/tatar_hw_dataset", help="Output directory")
    parser.add_argument("--img-size", type=int, default=64, help="Image canvas size")
    args = parser.parse_args()

    if args.generate:
        generate_batch_dataset_to_disk(
            out_dir=args.out_dir,
            samples_per_class=args.samples,
            img_size=args.img_size,
        )
    else:
        print("Tatar OCR Dataset module loaded.")
        print(f"Total defined classes: {len(ALL_CHARS)}")
        all_fonts, _ = get_available_fonts()
        print(f"Available fonts: {len(all_fonts)}")

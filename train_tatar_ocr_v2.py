"""
Tatar OCR v2 Domain-Aligned & Confusion-Aware Training Pipeline
- Starts from `models/finetuned_uppercase39.pth` (preserving v1 checkpoints untouched)
- Combines scanner-normalized `realdataset/` cells + v2 procedural/camera-aligned synthetic data
- Uses Confusion-Aware Margin Loss targeting real-world handwritten confusion pairs
- Applies WiSE-FT (Weight-Space Ensembling with BN recalibration) for zero catastrophic forgetting
- Exports `models/tatar_ocr_v2_handwritten39.pth`, `_fp32.onnx`, and `_int8.onnx`
- Strictly zero data leakage: `benchmark/images/blank_01..12.jpg` are NEVER used in training or validation.
"""

import os
import sys
import time
import copy
import json
import random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

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
from tatar_ocr_dataset import TatarOCRNet
from finetune_advanced_pipeline import advanced_augment_real_cell


# ---------------------------------------------------------------------------
# Confusion Matrix Pairs & Priority Sampling Weights
# ---------------------------------------------------------------------------

CONFUSION_GROUPS = {
    "П": ["И", "Л", "Һ"],
    "И": ["П", "Л", "Ч", "Ц", "М"],
    "Л": ["П", "И", "Ц", "Х", "А"],
    "Һ": ["П", "Ь", "Б", "Г"],
    "Н": ["М", "Ң", "Ч", "И"],
    "Ң": ["Н", "Җ", "М", "Щ"],
    "М": ["Н", "И", "Ы", "Л"],
    "Ә": ["Э", "Я", "Ө", "Ъ"],
    "Э": ["Ә", "Я", "С", "Е"],
    "Я": ["Ә", "Э", "Ц", "Р"],
    "У": ["Ч", "Ү"],
    "Ү": ["У", "Ч", "Ж", "Һ"],
    "Ч": ["У", "Ү", "И", "Ц"],
    "В": ["Ь", "Е", "Ю", "Р", "Ө", "Б"],
    "Ь": ["В", "Ъ", "Һ", "Б"],
    "Е": ["В", "Э", "С", "Ё"],
    "Ц": ["Щ", "И", "Ч", "Л", "Я"],
    "Щ": ["Ц", "Ш", "Җ"],
    "Ж": ["Җ", "Х", "Ү"],
    "Җ": ["Ж", "Ң", "Щ"],
    "Х": ["Л", "Ж", "К"],
    "Ф": ["Ө", "В", "Р"],
    "Р": ["В", "Ф", "Ь"],
    "А": ["Л", "Д", "Н"],
    "К": ["И", "Ү", "Х"],
    "Ю": ["О", "Ө", "В", "Ы"],
}

HARD_CHARS = set(CONFUSION_GROUPS.keys())


class ConfusionAwareLoss(nn.Module):
    """
    Combines label-smoothed Cross-Entropy with an explicit rival-suppression penalty
    on known handwritten confusion pairs (e.g. П<->Һ/Л/И, Н<->М/Ң, Ә<->Э/Я, У<->Ч, Ц<->Щ).
    """
    def __init__(self, chars, smoothing=0.03, rival_weight=0.45):
        super().__init__()
        self.chars = chars
        self.num_classes = len(chars)
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.rival_weight = rival_weight

        # Build binary mask [num_classes, num_classes] where mask[y, r] = 1 if r is a rival of y
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
        batch_mask = self.rival_mask[targets]  # [B, C]
        # Sum of probabilities assigned to known rival classes for each sample
        rival_probs = (probs * batch_mask).sum(dim=1).clamp(max=0.999)
        rival_loss = -torch.log(1.0 - rival_probs).mean()
        return ce_loss + self.rival_weight * rival_loss


# ---------------------------------------------------------------------------
# Real Dataset Augmentation (Scanner-Normalized + Distant Camera Simulation)
# ---------------------------------------------------------------------------

def augment_scanner_normalized_real_cell(norm_64, is_train=True):
    """
    Applies realistic perturbations to a scanner-normalized 64x64 real handwritten cell
    while preserving the canonical 48px tight framing and 250 outer canvas.
    """
    if not is_train:
        return norm_64.copy()

    h, w = norm_64.shape
    aug = norm_64.copy()

    # 1. Slight aspect-ratio squeeze/stretch, slant, rotation, and 1-2px centering jitter
    angle = random.uniform(-7.5, 7.5)
    shear = random.uniform(-7.5, 7.5)
    scale = random.uniform(0.92, 1.06)
    aspect = random.uniform(0.84, 1.18)
    sx = scale * np.sqrt(aspect)
    sy = scale / np.sqrt(aspect)

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 0] *= sx
    rot_mat[0, 1] *= sy
    rot_mat[1, 0] *= sx
    rot_mat[1, 1] *= sy
    rot_mat[0, 1] += np.tan(np.deg2rad(shear)) * rot_mat[0, 0]
    rot_mat[0, 2] = center[0] - (rot_mat[0, 0] * center[0] + rot_mat[0, 1] * center[1]) + random.uniform(-2.0, 2.0)
    rot_mat[1, 2] = center[1] - (rot_mat[1, 0] * center[0] + rot_mat[1, 1] * center[1]) + random.uniform(-2.0, 2.0)

    aug = cv2.warpAffine(aug, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=250)

    # 2. Pen thickness variation (fine ballpoint vs thick marker)
    r_m = random.random()
    if r_m < 0.25:
        k = 2
        aug = cv2.erode(aug, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    elif r_m < 0.45:
        k = 2
        aug = cv2.dilate(aug, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # 3. Mild hand tremor wobble
    if random.random() < 0.30:
        aug = apply_elastic_wobble(aug, alpha=random.uniform(2.0, 4.5), sigma=3.5)

    # 4. Distant camera low-res bottleneck + INTER_LANCZOS4 resampling (45% prob)
    if random.random() < 0.45:
        low_px = random.randint(20, 36)
        low = cv2.resize(aug, (low_px, low_px), interpolation=cv2.INTER_AREA)
        if random.random() < 0.5:
            low = cv2.GaussianBlur(low, (3, 3), random.uniform(0.35, 0.75))
        aug = cv2.resize(low, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 5. Re-normalize contrast to [10..250]
    p_lo, p_hi = np.percentile(aug, 2), np.percentile(aug, 98)
    if p_hi - p_lo > 10:
        aug = np.clip((aug.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 250).astype(np.uint8)
    return aug


def load_v2_real_dataset(metadata_csv="realdataset/metadata.csv", val_ratio=0.15, seed=42):
    """
    Loads `realdataset/` and prepares both:
    - `norm_scanner`: 64x64 tight-cropped & 48px-scaled image via `normalize_real_cell_like_scanner()`
    - `raw_64`: original 64x64 cell image
    """
    base_dir = Path(metadata_csv).parent
    with open(metadata_csv, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f if line.strip()]
    rows = lines[1:]

    by_char = {}
    for r in rows:
        ch = r[1]
        raw_100_path = base_dir / r[5]
        norm_64_path = base_dir / r[6]
        by_char.setdefault(ch, []).append((raw_100_path, norm_64_path))

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
            # Oversample hard/confused classes in training set (including А, К, Ю, М)
            if ch in HARD_CHARS:
                train_samples.append((scanner_64, im64, ch, cls_idx))

        for p100, p64 in val_list:
            im64 = np.array(Image.open(p64).convert("L"))
            im100 = np.array(Image.open(p100).convert("L")) if p100.exists() else im64
            scanner_64 = normalize_real_cell_like_scanner(im100, canvas_size=64)
            val_samples.append((scanner_64, im64, ch, cls_idx))

    return train_samples, val_samples


class V2RealHandwritingDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scanner_64, raw_64, ch, cls_idx = self.samples[idx]
        if self.is_train:
            # 70% scanner-aligned tight 48px crop, 30% legacy full-cell crop for stability
            if random.random() < 0.70:
                img = augment_scanner_normalized_real_cell(scanner_64, is_train=True)
            else:
                img = advanced_augment_real_cell(raw_64, is_train=True)
        else:
            img = scanner_64.copy()

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, cls_idx


class V2SyntheticHandwritingDataset(Dataset):
    """
    Generates online v2 synthetic handwritten cells aligned with `BlankOCRScanner.classify_cell`,
    with higher sampling weight on the 23 confusion-prone classes.
    """
    def __init__(self, chars, fonts_dict, font_weights, epoch_size=4500, is_train=True, seed=123):
        self.chars = chars
        self.num_classes = len(chars)
        self.epoch_size = epoch_size
        self.is_train = is_train
        self.fonts = fonts_dict
        self.font_names = list(fonts_dict.keys())
        w = np.array([font_weights[k] for k in self.font_names], dtype=np.float32)
        self.font_probs = w / w.sum()

        # Class sampling weights: 2.2x on confused classes (including А, К, Ю, М), 1.0x on all other classes
        c_weights = np.array([2.2 if ch in HARD_CHARS else 1.0 for ch in self.chars], dtype=np.float32)
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
            self.val_items = []
            for i in range(epoch_size):
                cls_idx = i % self.num_classes
                ch = self.chars[cls_idx]
                fname = rng.choice(self.font_names, p=self.font_probs)
                fpath = self.fonts[fname]
                arr = generate_augmented_cell_image(
                    char=ch,
                    font_path=fpath,
                    canvas_size=64,
                    elastic_strength=12.0,
                    pressure_variation=1.4,
                    enable_overwriting=False,
                    enable_flourishes=True,
                    enable_cell_borders=True,
                    enable_paper_texture=True,
                    enable_camera_blur=True,
                    base_img=self.glyph_cache.get((ch, fname)),
                    enable_handwritten_variants=True,
                    align_to_scanner=True,
                )
                self.val_items.append((arr, cls_idx))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if not self.is_train:
            arr, cls_idx = self.val_items[idx]
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
                elastic_strength=random.uniform(7.0, 19.0),
                pressure_variation=random.uniform(1.0, 2.1),
                enable_overwriting=random.random() < 0.16,
                enable_flourishes=random.random() < 0.30,
                enable_cell_borders=random.random() < 0.55,
                enable_paper_texture=True,
                enable_camera_blur=True,
                base_img=base_img,
                enable_handwritten_variants=True,
                align_to_scanner=True,
            )

        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, cls_idx


def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            preds = model(x_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
    return (correct / max(1, total)) * 100.0


def recalibrate_bn(model, real_loader, synth_loader, device, num_batches=35):
    """Recalibrates BatchNorm running statistics for WiSE-FT interpolated weights."""
    model.train()
    synth_iter = iter(synth_loader)
    with torch.no_grad():
        for idx, (x_r, _) in enumerate(real_loader):
            if idx >= num_batches:
                break
            try:
                x_s, _ = next(synth_iter)
            except StopIteration:
                synth_iter = iter(synth_loader)
                x_s, _ = next(synth_iter)
            x_b = torch.cat([x_r, x_s], dim=0).to(device)
            _ = model(x_b)
    model.eval()


def export_v2_onnx_and_int8(model, save_fp32_path, save_int8_path):
    """Exports FP32 ONNX model and dynamically quantizes to INT8 ONNX."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        str(save_fp32_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"✓ Exported FP32 ONNX model: {save_fp32_path} ({os.path.getsize(save_fp32_path)/1024:.1f} KB)")

    # Fuse Conv+BN before dynamic quantization by running ONNX Runtime pre-processing / quantization
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        prep_path = str(save_fp32_path).replace("_fp32.onnx", "_prep.onnx")
        quant_pre_process(str(save_fp32_path), prep_path, skip_symbolic_shape=False)
        src_for_quant = prep_path
    except Exception:
        src_for_quant = str(save_fp32_path)
        prep_path = None

    quantize_dynamic(
        model_input=src_for_quant,
        model_output=str(save_int8_path),
        weight_type=QuantType.QUInt8,
    )
    if prep_path and os.path.exists(prep_path):
        os.remove(prep_path)
    print(f"✓ Exported INT8 ONNX model: {save_int8_path} ({os.path.getsize(save_int8_path)/1024:.1f} KB)")


def run_v2_training(
    base_weights_path="models/finetuned_uppercase39.pth",
    save_pth="models/tatar_ocr_v2_handwritten39.pth",
    save_fp32="models/tatar_ocr_v2_handwritten39_fp32.onnx",
    save_int8="models/tatar_ocr_v2_handwritten39_int8.onnx",
    epochs=14,
    batch_size=48,
):
    print("=" * 75)
    print("TATAR OCR v2 TRAINING: SCANNER-ALIGNED & CONFUSION-AWARE FINE-TUNING")
    print("=" * 75)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(12)
    device = torch.device("cpu")

    chars = TATAR_UPPERCASE
    num_classes = len(chars)

    # 1. Load Real Dataset (with scanner tight-crop normalization + hard-class oversampling)
    train_real_samples, val_real_samples = load_v2_real_dataset()
    # Also load original uncropped validation set to verify zero regression on legacy validation
    orig_val_samples = [(raw64, raw64, ch, cidx) for (_, raw64, ch, cidx) in val_real_samples]

    train_real_ds = V2RealHandwritingDataset(train_real_samples, is_train=True)
    val_real_scanner_ds = V2RealHandwritingDataset(val_real_samples, is_train=False)
    val_real_orig_ds = V2RealHandwritingDataset(orig_val_samples, is_train=False)

    # 2. Load v2 Synthetic Dataset (aligned to BlankOCRScanner.classify_cell + handwritten allographs)
    all_fonts, font_weights = get_available_fonts()
    train_synth_ds = V2SyntheticHandwritingDataset(
        chars, all_fonts, font_weights, epoch_size=len(train_real_samples), is_train=True
    )
    val_synth_ds = V2SyntheticHandwritingDataset(
        chars, all_fonts, font_weights, epoch_size=39 * 12, is_train=False, seed=777
    )

    real_loader = DataLoader(train_real_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    synth_loader = DataLoader(train_synth_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_scanner_loader = DataLoader(val_real_scanner_ds, batch_size=64, shuffle=False)
    val_orig_loader = DataLoader(val_real_orig_ds, batch_size=64, shuffle=False)
    val_synth_loader = DataLoader(val_synth_ds, batch_size=64, shuffle=False)

    print(f"Loaded Training & Validation Sets (39 Uppercase Classes):")
    print(f"  • Real Train Samples (with hard-class oversampling): {len(train_real_ds)}")
    print(f"  • Synthetic v2 Samples per Epoch:                    {len(train_synth_ds)}")
    print(f"  • Held-out Real Val (Scanner-Normalized):            {len(val_real_scanner_ds)}")
    print(f"  • Held-out Synthetic v2 Allograph Val:               {len(val_synth_ds)}")

    # 3. Initialize Model from `models/finetuned_uppercase39.pth`
    model = TatarOCRNet(num_classes=num_classes).to(device)
    base_state = torch.load(base_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(copy.deepcopy(base_state))

    base_scanner_acc = evaluate_model(model, val_scanner_loader, device)
    base_orig_acc = evaluate_model(model, val_orig_loader, device)
    base_synth_acc = evaluate_model(model, val_synth_loader, device)
    print(f"\n--- Pre-Fine-Tuning v1 Baseline Validation ---")
    print(f"  • v1 on Scanner-Normalized Real Val: {base_scanner_acc:.2f}%")
    print(f"  • v1 on Legacy Real Val:             {base_orig_acc:.2f}%")
    print(f"  • v1 on v2 Handwritten Allograph Val:{base_synth_acc:.2f}%\n")

    # 4. Differential Learning Rate + Confusion-Aware Loss
    criterion = ConfusionAwareLoss(chars, smoothing=0.03, rival_weight=0.45).to(device)
    early_params = list(model.features[:8].parameters())
    late_params = list(model.features[8:].parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": early_params, "lr": 4e-5},
            {"params": late_params, "lr": 1.8e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_combined_score = -1.0
    best_finetuned_state = copy.deepcopy(model.state_dict())

    print("=" * 82)
    print(f"{'Epoch':^6} | {'Loss':^8} | {'Train Acc':^9} | {'RealScan Val':^12} | {'Allograph Val':^13} | {'Combined':^9} | {'Time':^6}")
    print("=" * 82)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, tot_corr, tot_cnt = 0.0, 0, 0

        synth_iter = iter(synth_loader)
        for x_real, y_real in real_loader:
            try:
                x_synth, y_synth = next(synth_iter)
            except StopIteration:
                synth_iter = iter(synth_loader)
                x_synth, y_synth = next(synth_iter)

            # 50% Real + 50% v2 Synthetic (procedural allographs + camera bottleneck + scanner tight-crop)
            x_b = torch.cat([x_real, x_synth], dim=0).to(device)
            y_b = torch.cat([y_real, y_synth], dim=0).to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            tot_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=1)
            tot_corr += (preds == y_b).sum().item()
            tot_cnt += len(y_b)

        scheduler.step()
        tr_loss = tot_loss / tot_cnt
        tr_acc = (tot_corr / tot_cnt) * 100.0

        v_scan = evaluate_model(model, val_scanner_loader, device)
        v_synth = evaluate_model(model, val_synth_loader, device)
        combined = 0.55 * v_scan + 0.45 * v_synth
        dt = time.time() - t0

        is_best = combined >= best_combined_score
        if is_best:
            best_combined_score = combined
            best_finetuned_state = copy.deepcopy(model.state_dict())

        star = " ★" if is_best else ""
        print(f"{ep:^6d} | {tr_loss:^8.4f} | {tr_acc:^8.2f}% | {v_scan:^11.2f}% | {v_synth:^12.2f}% | {combined:^8.2f}% | {dt:^5.1f}s{star}")

    # 5. WiSE-FT Weight-Space Ensemble Sweep on Held-Out Validation
    print("\n" + "=" * 82)
    print("RUNNING WiSE-FT WEIGHT-SPACE ENSEMBLE SWEEP (v1 theta_0 <-> v2 theta_1)")
    print("=" * 82)

    best_wise_score = -1.0
    best_wise_alpha = 0.70
    best_wise_state = copy.deepcopy(best_finetuned_state)

    for alpha in [0.50, 0.60, 0.68, 0.75, 0.82, 0.90, 1.00]:
        interp_state = {}
        for k in base_state.keys():
            if base_state[k].is_floating_point():
                interp_state[k] = (1.0 - alpha) * base_state[k] + alpha * best_finetuned_state[k]
            else:
                interp_state[k] = best_finetuned_state[k]

        model.load_state_dict(interp_state)
        recalibrate_bn(model, real_loader, synth_loader, device, num_batches=30)

        v_scan = evaluate_model(model, val_scanner_loader, device)
        v_orig = evaluate_model(model, val_orig_loader, device)
        v_synth = evaluate_model(model, val_synth_loader, device)
        wise_score = 0.45 * v_scan + 0.15 * v_orig + 0.40 * v_synth
        print(f"  • alpha={alpha:.2f} -> RealScan Val: {v_scan:6.2f}% | Legacy Val: {v_orig:6.2f}% | Allograph Val: {v_synth:6.2f}% | Score: {wise_score:6.2f}%")

        if wise_score >= best_wise_score:
            best_wise_score = wise_score
            best_wise_alpha = alpha
            best_wise_state = copy.deepcopy(model.state_dict())

    print(f"\n✓ Selected optimal WiSE-FT alpha = {best_wise_alpha:.2f} (Score: {best_wise_score:.2f}%)")
    model.load_state_dict(best_wise_state)
    Path(save_pth).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_pth)
    print(f"✓ Saved PyTorch checkpoint: {save_pth}")

    export_v2_onnx_and_int8(model, save_fp32, save_int8)
    return model


if __name__ == "__main__":
    run_v2_training()

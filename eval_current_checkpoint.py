"""
Live Standalone Quality Verification for Current Checkpoint
Evaluates models/best_tatar_ocr_net.pth without stopping or interfering with background training.
"""

import sys
import time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tatar_ocr_augmentation import (
    ALL_CHARS,
    TATAR_UPPERCASE,
    TATAR_LOWERCASE,
    DIGIT_CHARS,
    PUNCTUATION_CHARS,
    get_available_fonts,
    render_base_char,
    generate_augmented_cell_image,
)
from tatar_ocr_dataset import TatarOCRNet
from train_tatar_ocr import CachedAugmentedDataset


def evaluate_checkpoint(weights_path="models/best_tatar_ocr_net.pth", num_val_samples=1500):
    print("=" * 70, flush=True)
    print("LIVE EVALUATION OF CURRENT CHECKPOINT (NON-INTRUSIVE)", flush=True)
    print("=" * 70, flush=True)

    weights_file = Path(weights_path)
    if not weights_file.exists():
        print(f"Error: Weights file {weights_path} does not exist!", flush=True)
        return

    # 1. Load weights
    device = torch.device("cpu")
    torch.set_num_threads(4)  # Use 4 threads to avoid starving the main training process
    
    print(f"Loading checkpoint: {weights_file.resolve().as_posix()}", flush=True)
    file_stat = weights_file.stat()
    print(f"Checkpoint last updated: {time.ctime(file_stat.st_mtime)} (Size: {file_stat.st_size / (1024*1024):.2f} MB)", flush=True)

    model = TatarOCRNet(num_classes=len(ALL_CHARS)).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.eval()
    print("✓ Model successfully loaded into evaluation harness.", flush=True)

    # 2. Build representative validation dataset
    all_fonts, font_weights = get_available_fonts()
    print(f"Generating {num_val_samples} challenging realistic notebook test samples across 102 classes...", flush=True)
    t0_gen = time.time()
    val_ds = CachedAugmentedDataset(
        chars=ALL_CHARS,
        fonts_dict=all_fonts,
        font_weights=font_weights,
        epoch_size=num_val_samples,
        is_train=False,
        seed=123,  # Fixed test seed
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    print(f"✓ Test dataset ready in {time.time() - t0_gen:.2f}s!", flush=True)

    # 3. Evaluation inference
    print("\nRunning model inference across test set...", flush=True)
    t0_inf = time.time()
    all_preds = []
    all_targets = []
    all_confidences = []
    top3_correct = 0

    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            probs = torch.softmax(logits, dim=1)

            top3 = torch.topk(probs, k=3, dim=1).indices
            for i in range(len(y_b)):
                if y_b[i] in top3[i]:
                    top3_correct += 1

            max_probs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(y_b.numpy())
            all_confidences.extend(max_probs.numpy())

    inference_time = time.time() - t0_inf
    print(f"✓ Inference finished in {inference_time:.2f}s ({num_val_samples/inference_time:.1f} samples/sec)!", flush=True)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)

    # 4. Metrics
    overall_top1 = (all_preds == all_targets).mean() * 100.0
    overall_top3 = (top3_correct / len(all_targets)) * 100.0

    print("\n" + "=" * 70, flush=True)
    print(f"OVERALL PERFORMANCE ON REALISTIC NOTEBOOK TEST SAMPLES", flush=True)
    print("=" * 70, flush=True)
    print(f"  ★ Top-1 Accuracy: {overall_top1:6.2f}%", flush=True)
    print(f"  ★ Top-3 Accuracy: {overall_top3:6.2f}%", flush=True)

    # 5. Category breakdown
    categories = {
        "Tatar Specific (Ә, Ө, Ү, Җ, Ң, Һ + lower)": [
            "Ә", "Ө", "Ү", "Җ", "Ң", "Һ", "ә", "ө", "ү", "җ", "ң", "һ"
        ],
        "Standard Tatar Uppercase (А-Я)": TATAR_UPPERCASE,
        "Standard Tatar Lowercase (а-я)": TATAR_LOWERCASE,
        "Digits (0-9)": DIGIT_CHARS,
        "Punctuation Chars": PUNCTUATION_CHARS,
    }

    print("\n--- Accuracy by Character Category ---", flush=True)
    for cat_name, cat_chars in categories.items():
        cat_indices = [ALL_CHARS.index(c) for c in cat_chars if c in ALL_CHARS]
        mask = np.isin(all_targets, cat_indices)
        if mask.sum() > 0:
            cat_acc = (all_preds[mask] == all_targets[mask]).mean() * 100.0
            print(f"  • {cat_name:<44}: {cat_acc:6.2f}% ({mask.sum():3d} samples)", flush=True)

    # 6. Specific Tatar Glyphs
    special_chars = ["Ә", "Ө", "Ү", "Җ", "Ң", "Һ", "ә", "ө", "ү", "җ", "ң", "һ"]
    print("\n--- Individual Accuracy for Special Tatar Letters ---", flush=True)
    for sc in special_chars:
        sc_idx = ALL_CHARS.index(sc)
        mask = (all_targets == sc_idx)
        if mask.sum() > 0:
            sc_acc = (all_preds[mask] == all_targets[mask]).mean() * 100.0
            print(f"  • Letter '{sc}' (U+{ord(sc):04X}): {sc_acc:6.2f}% ({mask.sum():2d} samples)", flush=True)

    # 7. Confidence Calibration (Decision vs Flagging for Teacher)
    print("\n--- Mobile HITL Confidence Calibration ('Flag vs Decision') ---", flush=True)
    print(f"{'Threshold':^12} | {'Accuracy on Accepted':^22} | {'Flagged for Teacher':^22}", flush=True)
    print("-" * 62, flush=True)
    for th in [0.50, 0.70, 0.85, 0.90, 0.95]:
        accepted_mask = all_confidences >= th
        acc_at_th = (all_preds[accepted_mask] == all_targets[accepted_mask]).mean() * 100.0 if accepted_mask.sum() > 0 else 100.0
        flagged_pct = (1.0 - accepted_mask.mean()) * 100.0
        print(f"  Confidence >= {th:.2f} | {acc_at_th:18.2f}% | {flagged_pct:18.1f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    evaluate_checkpoint()

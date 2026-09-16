"""
Tatar OCR Training & Quality Evaluation Script
Option C: 102 classes, 20 epochs, online realistic augmentation, 
validation tracking, comprehensive metrics, and ONNX export.
"""

import sys
import os
import time
import json
import random
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

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


# ---------------------------------------------------------------------------
# Dataset with Train / Validation Split & Efficient In-Memory Glyph Cache
# ---------------------------------------------------------------------------

class CachedAugmentedDataset(Dataset):
    """
    High-throughput dataset using in-memory pre-rendered base glyphs
    with on-the-fly realistic augmentation pipelines.
    """
    def __init__(self, chars, fonts_dict, font_weights, epoch_size=15606, is_train=True, seed=42):
        self.chars = chars
        self.num_classes = len(chars)
        self.epoch_size = epoch_size
        self.is_train = is_train
        self.fonts = fonts_dict
        self.font_names = list(fonts_dict.keys())
        self.weights = np.array([font_weights[k] for k in self.font_names], dtype=np.float32)
        self.weights /= self.weights.sum()
        
        # Pre-render clean base glyphs in memory (128x128)
        self.glyph_cache = {}
        for ch in self.chars:
            for fname, fpath in self.fonts.items():
                try:
                    self.glyph_cache[(ch, fname)] = render_base_char(ch, fpath, canvas_size=128)
                except Exception:
                    self.glyph_cache[(ch, fname)] = np.full((128, 128), 255, dtype=np.uint8)
                    
        # Fixed random seed sequence for validation reproducibility
        if not self.is_train:
            rng = np.random.RandomState(seed)
            self.fixed_tasks = []
            for _ in range(epoch_size):
                cls_idx = rng.randint(0, self.num_classes)
                ch = self.chars[cls_idx]
                fname = rng.choice(self.font_names, p=self.weights)
                self.fixed_tasks.append((cls_idx, ch, fname))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.is_train:
            cls_idx = random.randint(0, self.num_classes - 1)
            ch = self.chars[cls_idx]
            fname = np.random.choice(self.font_names, p=self.weights)
            fpath = self.fonts[fname]
            base_img = self.glyph_cache.get((ch, fname))
            
            # Full realistic augmentation training pipeline
            arr = generate_augmented_cell_image(
                char=ch,
                font_path=fpath,
                canvas_size=64,
                elastic_strength=random.uniform(8.0, 22.0),
                pressure_variation=random.uniform(1.0, 2.2),
                enable_overwriting=random.random() < 0.22,
                enable_flourishes=random.random() < 0.35,
                enable_cell_borders=random.random() < 0.65,
                enable_paper_texture=True,
                enable_camera_blur=random.random() < 0.35,
                base_img=base_img,
            )
        else:
            cls_idx, ch, fname = self.fixed_tasks[idx]
            fpath = self.fonts[fname]
            base_img = self.glyph_cache.get((ch, fname))
            
            # Validation pipeline (representative realistic corruption)
            arr = generate_augmented_cell_image(
                char=ch,
                font_path=fpath,
                canvas_size=64,
                elastic_strength=14.0,
                pressure_variation=1.4,
                enable_overwriting=True,
                enable_flourishes=True,
                enable_cell_borders=True,
                enable_paper_texture=True,
                enable_camera_blur=True,
                base_img=base_img,
            )

        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        return tensor, cls_idx


# ---------------------------------------------------------------------------
# Training & Verification Runner
# ---------------------------------------------------------------------------

def run_training(
    epochs=20,
    batch_size=64,
    samples_per_class=180,
    learning_rate=1e-3,
    save_dir="models",
    eval_only=False,
):
    print("=" * 70)
    print("TATAR OCR TRAINING: OPTION C (102 CLASSES, 20 EPOCHS)")
    print("=" * 70)
    
    # 1. Hardware & Thread Optimization
    torch.set_num_threads(12)
    device = torch.device("cpu")
    print(f"Device: {device} | PyTorch CPU Threads: {torch.get_num_threads()}")
    
    # 2. Fonts & Classes
    all_fonts, font_weights = get_available_fonts()
    chars = ALL_CHARS
    num_classes = len(chars)
    print(f"Total Target Classes: {num_classes}")
    print(f"Loaded Fonts: {len(all_fonts)}")
    
    # 3. Dataset Sizes (85% Train / 15% Validation)
    total_samples = num_classes * samples_per_class  # 102 * 180 = 18,360
    train_size = int(total_samples * 0.85)           # 15,606
    val_size = total_samples - train_size             # 2,754
    print(f"Total dataset budget: {total_samples} samples per epoch")
    print(f"  - Train Set:      {train_size} samples / epoch")
    print(f"  - Validation Set: {val_size} samples (held-out)")

    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pre-caching in-memory datasets
    print("\nPre-rasterizing clean glyphs into memory cache...")
    t0_cache = time.time()
    train_ds = CachedAugmentedDataset(chars, all_fonts, font_weights, epoch_size=train_size, is_train=True)
    val_ds = CachedAugmentedDataset(chars, all_fonts, font_weights, epoch_size=val_size, is_train=False, seed=42)
    print(f"Cache complete in {time.time() - t0_cache:.2f}s! ({len(train_ds.glyph_cache)} glyph templates cached)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Model Architecture & Optimizer
    model = TatarOCRNet(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Weights size: ~{total_params * 4 / (1024*1024):.2f} MB)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    history = []
    start_total_time = time.time()

    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Elapsed':^9}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t0_epoch = time.time()
        
        # --- TRAIN STEP ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_b).sum().item()
            train_total += len(y_b)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # --- VALIDATION STEP ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        elapsed_epoch = time.time() - t0_epoch

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "elapsed_s": elapsed_epoch,
        })

        is_best = (epoch == 1) or (epoch_val_acc >= best_val_acc)
        if is_best:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), out_path / "best_tatar_ocr_net.pth")

        best_marker = " 🌟 BEST" if is_best else ""
        print(f"{epoch:^7d} | {epoch_train_loss:^10.4f} | {epoch_train_acc:^9.2f}% | {epoch_val_loss:^10.4f} | {epoch_val_acc:^9.2f}% | {elapsed_epoch:^7.1f}s{best_marker}")

    total_training_time = time.time() - start_total_time
    print("=" * 70)
    print(f"Training Complete in {total_training_time/60:.2f} minutes!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Load best checkpoint for final verification
    best_weights_path = out_path / "best_tatar_ocr_net.pth"
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    
    # -----------------------------------------------------------------------
    # 5. Comprehensive Quality Verification Suite
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING FINAL QUALITY VERIFICATION SUITE")
    print("=" * 70)
    
    model.eval()
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
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            all_confidences.extend(max_probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)

    overall_top1 = (all_preds == all_targets).mean() * 100.0
    overall_top3 = (top3_correct / len(all_targets)) * 100.0
    print(f"Overall Top-1 Accuracy: {overall_top1:.2f}%")
    print(f"Overall Top-3 Accuracy: {overall_top3:.2f}%")

    # Category breakdowns
    categories = {
        "Tatar Specific (Ә, Ө, Ү, Җ, Ң, Һ + lower)": [
            "Ә", "Ө", "Ү", "Җ", "Ң", "Һ", "ә", "ө", "ү", "җ", "ң", "һ"
        ],
        "Standard Tatar Uppercase": TATAR_UPPERCASE,
        "Standard Tatar Lowercase": TATAR_LOWERCASE,
        "Digits (0-9)": DIGIT_CHARS,
        "Punctuation": PUNCTUATION_CHARS,
    }

    category_results = {}
    print("\n--- Accuracy Breakdown by Character Category ---")
    for cat_name, cat_chars in categories.items():
        cat_indices = [chars.index(c) for c in cat_chars if c in chars]
        mask = np.isin(all_targets, cat_indices)
        if mask.sum() > 0:
            cat_acc = (all_preds[mask] == all_targets[mask]).mean() * 100.0
            category_results[cat_name] = round(float(cat_acc), 2)
            print(f"  • {cat_name:<42}: {cat_acc:6.2f}% ({mask.sum()} samples)")

    # Per-letter specific Tatar accuracy
    special_tatar_acc = {}
    special_chars = ["Ә", "Ө", "Ү", "Җ", "Ң", "Һ", "ә", "ө", "ү", "җ", "ң", "һ"]
    print("\n--- Individual Accuracy for Special Tatar Letters ---")
    for sc in special_chars:
        sc_idx = chars.index(sc)
        mask = (all_targets == sc_idx)
        if mask.sum() > 0:
            sc_acc = (all_preds[mask] == all_targets[mask]).mean() * 100.0
            special_tatar_acc[sc] = round(float(sc_acc), 2)
            print(f"  • Glyph '{sc}' (U+{ord(sc):04X}): {sc_acc:6.2f}% ({mask.sum()} test samples)")

    # Confidence vs Error Analysis (Calibration for the 'Flag vs Decision' feature)
    print("\n--- Model Calibration & Uncertainty (Flag Thresholds) ---")
    thresholds = [0.50, 0.70, 0.85, 0.95]
    flag_stats = {}
    for th in thresholds:
        accepted_mask = all_confidences >= th
        acc_at_th = (all_preds[accepted_mask] == all_targets[accepted_mask]).mean() * 100.0 if accepted_mask.sum() > 0 else 100.0
        flagged_ratio = (1.0 - accepted_mask.mean()) * 100.0
        flag_stats[f"Threshold_{th}"] = {
            "Accuracy_On_Accepted": round(float(acc_at_th), 2),
            "Flagged_For_Teacher_Review": round(float(flagged_ratio), 2),
        }
        print(f"  • Threshold >= {th:.2f} -> Accuracy: {acc_at_th:6.2f}% | Flagged for Review: {flagged_ratio:5.1f}%")

    # 6. Save Model Checkpoint & ONNX Export
    print("\n--- Exporting Deployable Weights & Mobile ONNX Model ---")
    weights_path = out_path / "best_tatar_ocr_net.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"  ✓ Saved PyTorch weights: {weights_path} ({weights_path.stat().st_size / (1024*1024):.2f} MB)")

    try:
        onnx_path = out_path / "tatar_ocr_net.onnx"
        dummy_input = torch.randn(1, 1, 64, 64)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["cell_image"],
            output_names=["class_logits"],
            dynamic_axes={"cell_image": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
            opset_version=14,
        )
        print(f"  ✓ Exported mobile ONNX model: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"  ℹ Note: ONNX export skipped ({e}). PyTorch model weights are fully saved and ready.")

    # 7. Write Final Quality Evaluation Report JSON
    report = {
        "config": {
            "epochs": epochs,
            "classes": num_classes,
            "total_budget_per_epoch": total_samples,
            "train_samples": train_size,
            "val_samples": val_size,
            "batch_size": batch_size,
            "training_time_minutes": round(total_training_time / 60.0, 2),
        },
        "overall_metrics": {
            "best_val_accuracy_top1": round(best_val_acc, 2),
            "val_accuracy_top3": round(overall_top3, 2),
        },
        "category_accuracy": category_results,
        "special_tatar_accuracy": special_tatar_acc,
        "confidence_calibration": flag_stats,
        "history": history,
    }
    
    with open(out_path / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Evaluation report written to: {out_path / 'evaluation_report.json'}")

    print("\n🎉 ALL TRAINING AND VERIFICATION STEPS COMPLETED SUCCESSFULLY!")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-dry-run", action="store_true", help="Quick 1-epoch dry run for verification")
    args = parser.parse_args()

    if args.test_dry_run:
        print("Executing quick dry run...")
        run_training(epochs=1, batch_size=32, samples_per_class=2)
    else:
        run_training(epochs=args.epochs, batch_size=args.batch_size, samples_per_class=180)

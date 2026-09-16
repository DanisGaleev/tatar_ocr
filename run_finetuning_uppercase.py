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
from torch.utils.data import DataLoader

from tatar_ocr_augmentation import (
    TATAR_UPPERCASE,
    get_available_fonts,
)
from tatar_ocr_dataset import TatarOCRNet
from train_tatar_ocr import CachedAugmentedDataset
from finetune_advanced_pipeline import (
    AdvancedRealDataset,
    load_real_uppercase_dataset,
)


def run_finetuning_uppercase(
    base_weights_path="models/baseline_synthetic_uppercase39.pth",
    epochs=12,
    batch_size=48,
    lr=2e-4,
    save_path="models/finetuned_uppercase39.pth",
):
    print("=" * 70, flush=True)
    print("STAGE 2: DOMAIN ADAPTATION ON REAL HANDWRITING WITH ADVANCED AUGMENTATION", flush=True)
    print("=" * 70, flush=True)

    torch.set_num_threads(12)
    device = torch.device("cpu")
    chars = TATAR_UPPERCASE
    num_classes = len(chars)

    # 1. Load Real Dataset
    train_samples, val_samples = load_real_uppercase_dataset()
    print(f"Loaded Real Handwriting Dataset (39 classes):")
    print(f"  • Real Train Samples:      {len(train_samples)} (advanced dynamic augmentation)")
    print(f"  • Real Validation Samples: {len(val_samples)} (held-out test set)")

    train_real_ds = AdvancedRealDataset(train_samples, is_train=True)
    val_real_ds = AdvancedRealDataset(val_samples, is_train=False)

    real_loader = DataLoader(train_real_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_real_ds, batch_size=64, shuffle=False)

    # 2. Synthetic Regularization Dataset (Joint Regularization to prevent catastrophic forgetting)
    all_fonts, font_weights = get_available_fonts()
    synth_train_ds = CachedAugmentedDataset(chars, all_fonts, font_weights, epoch_size=len(train_samples), is_train=True)
    synth_loader = DataLoader(synth_train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # 3. Model Architecture & Load Base Checkpoint
    model = TatarOCRNet(num_classes=num_classes).to(device)
    print(f"\nLoading baseline synthetic weights: {base_weights_path}...", flush=True)
    model.load_state_dict(torch.load(base_weights_path, map_location=device, weights_only=True))

    # Evaluate Baseline Model on Real Val Set BEFORE Fine-Tuning
    print("\n--- Zero-Shot Baseline Performance on Real Handwriting (Pre-Fine-Tuning) ---", flush=True)
    model.eval()
    base_preds, base_targets, base_confs = [], [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            probs = torch.softmax(model(x_b), dim=1)
            confs, preds = torch.max(probs, dim=1)
            base_preds.extend(preds.numpy())
            base_targets.extend(y_b.numpy())
            base_confs.extend(confs.numpy())

    base_preds = np.array(base_preds)
    base_targets = np.array(base_targets)
    baseline_top1 = (base_preds == base_targets).mean() * 100.0
    print(f"★ Baseline Real Handwriting Top-1 Accuracy: {baseline_top1:.2f}%\n", flush=True)

    # 4. Fine-Tuning with Joint Regularization & Cosine Annealing
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("=" * 70, flush=True)
    hdr = f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Elapsed':^9}"
    print(hdr, flush=True)
    print("=" * 70, flush=True)

    best_val_acc = 0.0
    history = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0

        synth_iter = iter(synth_loader)
        for x_real, y_real in real_loader:
            try:
                x_synth, y_synth = next(synth_iter)
            except StopIteration:
                synth_iter = iter(synth_loader)
                x_synth, y_synth = next(synth_iter)

            # Combine 75% real + 25% synthetic regularization
            n_synth = min(len(y_synth), max(4, len(y_real) // 3))
            x_batch = torch.cat([x_real, x_synth[:n_synth]], dim=0).to(device)
            y_batch = torch.cat([y_real, y_synth[:n_synth]], dim=0).to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * len(y_real)
            preds = logits[:len(y_real)].argmax(dim=1)
            t_corr += (preds == y_real.to(device)).sum().item()
            t_tot += len(y_real)

        scheduler.step()
        ep_t_loss = t_loss / t_tot
        ep_t_acc = (t_corr / t_tot) * 100.0

        # Validate strictly on held-out real handwriting test set
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                v_loss += loss.item() * len(y_b)
                preds = logits.argmax(dim=1)
                v_corr += (preds == y_b).sum().item()
                v_tot += len(y_b)

        ep_v_loss = v_loss / v_tot
        ep_v_acc = (v_corr / v_tot) * 100.0
        elapsed = time.time() - t0

        is_best = (ep == 1) or (ep_v_acc >= best_val_acc)
        if is_best:
            best_val_acc = ep_v_acc
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

        best_star = " [BEST]" if is_best else ""
        row = f"{ep:^7d} | {ep_t_loss:^10.4f} | {ep_t_acc:^9.2f}% | {ep_v_loss:^10.4f} | {ep_v_acc:^9.2f}% | {elapsed:^7.1f}s{best_star}"
        print(row, flush=True)
        history.append({
            "epoch": ep, "train_loss": ep_t_loss, "train_acc": ep_t_acc,
            "val_loss": ep_v_loss, "val_acc": ep_v_acc, "elapsed_s": elapsed
        })

    print("=" * 70, flush=True)
    print(f"Fine-tuning complete in {(time.time() - t_start)/60:.2f} mins! Best Real Val Acc: {best_val_acc:.2f}%\n", flush=True)

    # Step C: Deep verification on real validation set
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    fin_preds, fin_targets, fin_confs = [], [], []
    top3_count = 0
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            probs = torch.softmax(model(x_b), dim=1)
            top3 = torch.topk(probs, k=3, dim=1).indices
            for i in range(len(y_b)):
                if y_b[i] in top3[i]:
                    top3_count += 1
            confs, preds = torch.max(probs, dim=1)
            fin_preds.extend(preds.numpy())
            fin_targets.extend(y_b.numpy())
            fin_confs.extend(confs.numpy())

    fin_preds = np.array(fin_preds)
    fin_targets = np.array(fin_targets)
    fin_confs = np.array(fin_confs)

    post_top1 = (fin_preds == fin_targets).mean() * 100.0
    post_top3 = (top3_count / len(fin_targets)) * 100.0

    print(f"  ★ Baseline Real Accuracy:    {baseline_top1:6.2f}%")
    print(f"  ★ Fine-Tuned Real Accuracy:  {post_top1:6.2f}% (Gain: +{post_top1 - baseline_top1:+.2f}%)")
    print(f"  ★ Fine-Tuned Top-3 Accuracy: {post_top3:6.2f}%\n", flush=True)

    # Detailed Tatar specific letters
    priority_letters = ["Ә", "Җ", "Ң", "Ө", "Ү", "Һ", "Щ", "Ц", "Ч", "Ш", "Д", "Ю", "Б", "Э"]
    print("--- Detailed Accuracy for Key & Difficult Tatar Letters ---", flush=True)
    hdr_p = f"{'Letter':^8} | {'Unicode':^8} | {'Baseline Acc':^14} | {'Fine-Tuned Acc':^16} | {'Gain':^8} | {'Count':^8}"
    print(hdr_p, flush=True)
    print("-" * 75, flush=True)

    per_char_metrics = {}
    for ch in priority_letters:
        cls_idx = chars.index(ch)
        mask = (fin_targets == cls_idx)
        if mask.sum() > 0:
            b_acc = (base_preds[mask] == base_targets[mask]).mean() * 100.0
            f_acc = (fin_preds[mask] == fin_targets[mask]).mean() * 100.0
            gain = f_acc - b_acc
            per_char_metrics[ch] = {
                "baseline_acc": round(float(b_acc), 2),
                "finetuned_acc": round(float(f_acc), 2),
                "gain": round(float(gain), 2),
                "count": int(mask.sum())
            }
            row_p = f"   '{ch}'   | U+{ord(ch):04X} | {b_acc:12.2f}% | {f_acc:14.2f}% | {gain:+6.2f}% | {mask.sum():6d}"
            print(row_p, flush=True)

    # Step D: Test on 6-cell strip
    print("\n--- Evaluating on 6-Cell Photo Strip (Ә, А, Б, Җ, Щ, Ч) ---", flush=True)
    ascii_names = {1: ('Ae', 'Ә'), 2: ('A', 'А'), 3: ('B', 'Б'), 4: ('Zh', 'Җ'), 5: ('Shch', 'Щ'), 6: ('Ch', 'Ч')}
    strip2_results = []
    strip2_correct = 0
    for cid in range(1, 7):
        fname, exp = ascii_names[cid]
        cpath = f"dataset/six_cells_test/cell_{cid}_{fname}.png"
        if os.path.exists(cpath):
            img = cv2.imread(cpath, cv2.IMREAD_GRAYSCALE)
            img_64 = cv2.resize(img, (64, 64))
            t = (torch.from_numpy(img_64.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0) - 0.5) / 0.5
            with torch.no_grad():
                probs = torch.softmax(model(t), dim=1)[0]
            top_idx = probs.argmax().item()
            top_char = chars[top_idx]
            conf = probs.max().item() * 100.0
            exp_conf = probs[chars.index(exp)].item() * 100.0
            is_match = (top_char == exp)
            if is_match:
                strip2_correct += 1
            status = "[MATCH]" if is_match else "[FLAG]"
            print(f"  Cell {cid} (Expected '{exp}'): Pred -> '{top_char}' ({conf:5.1f}%) | Confidence on '{exp}': {exp_conf:5.1f}% {status}", flush=True)
            strip2_results.append({"cell": cid, "expected": exp, "pred": top_char, "conf": round(conf, 2), "exp_conf": round(exp_conf, 2), "match": is_match})

    report = {
        "summary": {
            "task": "39-Class Tatar Uppercase OCR Model with Advanced Augmentation Fine-Tuning",
            "real_dataset_cells": len(train_samples) + len(val_samples),
            "epochs": epochs,
            "baseline_real_accuracy": round(baseline_top1, 2),
            "finetuned_real_accuracy": round(post_top1, 2),
            "accuracy_gain": round(post_top1 - baseline_top1, 2),
            "top3_accuracy": round(post_top3, 2),
            "strip2_score": f"{strip2_correct}/6"
        },
        "priority_letters": per_char_metrics,
        "strip2_predictions": strip2_results,
        "history": history
    }

    report_path = "models/finetuned_uppercase39_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved fine-tuned 39-class model to: {save_path}")
    print(f"✓ Saved evaluation report to: {report_path}")
    return report

if __name__ == "__main__":
    run_finetuning_uppercase()


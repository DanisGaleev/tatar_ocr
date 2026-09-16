import sys
import os
import time
import json
import random
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tatar_ocr_augmentation import (
    TATAR_UPPERCASE,
    get_available_fonts,
)
from tatar_ocr_dataset import TatarOCRNet
from train_tatar_ocr import CachedAugmentedDataset


def train_uppercase_synthetic(
    epochs=15,
    batch_size=64,
    samples_per_class=180,
    learning_rate=1e-3,
    save_path="models/baseline_synthetic_uppercase39.pth",
):
    print("=" * 70, flush=True)
    print("STAGE 1: TRAINING DEDICATED 39-CLASS TATAR UPPERCASE MODEL", flush=True)
    print("=" * 70, flush=True)

    torch.set_num_threads(12)
    device = torch.device("cpu")
    chars = TATAR_UPPERCASE
    num_classes = len(chars)
    print(f"Target classes: {num_classes} Tatar uppercase letters")

    all_fonts, font_weights = get_available_fonts()
    print(f"Loaded {len(all_fonts)} verified fonts")

    total_budget = num_classes * samples_per_class  # 39 * 180 = 7020
    train_size = int(total_budget * 0.85)           # 5967
    val_size = total_budget - train_size             # 1053

    print(f"Dataset Budget per Epoch: {total_budget} (Train: {train_size}, Val: {val_size})")

    print("Pre-rasterizing glyphs into RAM cache...", flush=True)
    train_ds = CachedAugmentedDataset(chars, all_fonts, font_weights, epoch_size=train_size, is_train=True)
    val_ds = CachedAugmentedDataset(chars, all_fonts, font_weights, epoch_size=val_size, is_train=False, seed=42)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TatarOCRNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    history = []
    t_start = time.time()

    print("\n" + "=" * 70, flush=True)
    hdr = f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Elapsed':^9}"
    print(hdr, flush=True)
    print("=" * 70, flush=True)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=1)
            t_corr += (preds == y_b).sum().item()
            t_tot += len(y_b)

        scheduler.step()
        ep_t_loss = t_loss / t_tot
        ep_t_acc = (t_corr / t_tot) * 100.0

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
    print(f"Synthetic base training complete in {(time.time() - t_start)/60:.2f} mins! Best Val Acc: {best_val_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            preds = model(x_b.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    top1 = (all_preds == all_targets).mean() * 100.0

    report = {
        "stage": "synthetic_uppercase_base",
        "classes_count": num_classes,
        "epochs": epochs,
        "best_val_accuracy": round(best_val_acc, 2),
        "final_top1_accuracy": round(float(top1), 2),
        "weights_path": save_path,
        "history": history
    }
    with open("models/synthetic_uppercase_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved synthetic uppercase model to: {save_path}")
    return model, report

if __name__ == "__main__":
    train_uppercase_synthetic(epochs=15, batch_size=64, samples_per_class=180)

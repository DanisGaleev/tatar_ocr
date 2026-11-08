"""
v3.screen_architectures
Rapid empirical shootout of v3 architectural candidates (ResSE, Mobile, MultiScale)
Trained under identical mini-budgets (5 epochs) on real + synthetic data
Evaluated directly on the 12 canonical blanks benchmark.
"""

import os
import sys
import time
import json
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from v3.architecture import (
    TatarOCRNetV3_ResSE,
    TatarOCRNetV3_Mobile,
    TatarOCRNetV3_MultiScale,
    count_parameters,
)
from v3.dataset import (
    RealDatasetV3,
    SyntheticDatasetV3,
    load_real_samples,
    ConfusionAwareLoss,
)
from v3.evaluate_blanks import evaluate_model_on_blanks
from tatar_ocr_augmentation import TATAR_UPPERCASE


def run_screening(epochs: int = 5, batch_size: int = 64, device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device_str)
    print("=" * 75)
    print(f"TATAR OCR v3 ARCHITECTURAL SCREENING SHOOTOUT (Device: {device})")
    print(f"Running {epochs} epochs per candidate on combined Real + Synthetic dataset")
    print("=" * 75)

    # 1. Prepare Datasets
    print("\nLoading real and synthetic datasets...")
    train_real_samples, val_real_samples = load_real_samples("realdataset/metadata.csv", val_ratio=0.15, seed=42)
    real_train_ds = RealDatasetV3(train_real_samples, is_train=True)
    real_val_ds = RealDatasetV3(val_real_samples, is_train=False)

    synth_train_ds = SyntheticDatasetV3(TATAR_UPPERCASE, epoch_size=3000, is_train=True, seed=42)
    synth_val_ds = SyntheticDatasetV3(TATAR_UPPERCASE, epoch_size=800, is_train=False, seed=42)

    real_loader = DataLoader(real_train_ds, batch_size=batch_size // 2, shuffle=True, drop_last=True)
    synth_loader = DataLoader(synth_train_ds, batch_size=batch_size // 2, shuffle=True, drop_last=True)
    val_loader = DataLoader(real_val_ds, batch_size=batch_size, shuffle=False)

    candidates = {
        "ResSE": TatarOCRNetV3_ResSE(num_classes=len(TATAR_UPPERCASE)),
        "Mobile": TatarOCRNetV3_Mobile(num_classes=len(TATAR_UPPERCASE)),
        "MultiScale": TatarOCRNetV3_MultiScale(num_classes=len(TATAR_UPPERCASE)),
    }

    results = {}

    for name, model in candidates.items():
        print(f"\n>>> [Testing Candidate: {name}] <<<")
        params = count_parameters(model)
        print(f"  Parameters: {params:,} ({params * 4 / (1024*1024):.2f} MB float32)")

        model = model.to(device)
        criterion = ConfusionAwareLoss(TATAR_UPPERCASE, smoothing=0.03, rival_weight=0.40).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        t0_train = time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_corr = 0
            total_samples = 0

            synth_iter = iter(synth_loader)
            for x_r, y_r in real_loader:
                try:
                    x_s, y_s = next(synth_iter)
                except StopIteration:
                    synth_iter = iter(synth_loader)
                    x_s, y_s = next(synth_iter)

                x_b = torch.cat([x_r, x_s], dim=0).to(device)
                y_b = torch.cat([y_r, y_s], dim=0).to(device)

                optimizer.zero_grad()
                logits = model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_b)
                preds = logits.argmax(dim=1)
                total_corr += (preds == y_b).sum().item()
                total_samples += len(y_b)

            scheduler.step()
            train_acc = (total_corr / max(1, total_samples)) * 100.0

            # Real validation accuracy
            model.eval()
            val_corr, val_tot = 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_preds = model(vx).argmax(dim=1)
                    val_corr += (v_preds == vy).sum().item()
                    val_tot += len(vy)
            val_acc = (val_corr / max(1, val_tot)) * 100.0

            print(f"    Epoch {epoch}/{epochs} | Train Loss: {total_loss/total_samples:.4f} | Train Acc: {train_acc:.1f}% | Real Val Acc: {val_acc:.1f}%")

        train_time = time.perf_counter() - t0_train

        # Latency benchmark
        model.eval()
        dummy = torch.randn(1, 1, 64, 64).to(device)
        for _ in range(30):
            _ = model(dummy)
        t_lat_0 = time.perf_counter()
        n_lat = 500
        with torch.no_grad():
            for _ in range(n_lat):
                _ = model(dummy)
        lat_ms = (time.perf_counter() - t_lat_0) / n_lat * 1000.0

        # Benchmark on 12 Blanks
        print(f"  Evaluating {name} on 12 test blanks...")
        model_cpu = model.to("cpu")
        blank_metrics = evaluate_model_on_blanks(model_cpu, device="cpu", verbose=False)

        results[name] = {
            "params": params,
            "train_time_sec": round(train_time, 2),
            "latency_ms": round(lat_ms, 3),
            "real_val_acc": round(val_acc, 2),
            "Acc_char": blank_metrics["Acc_char"],
            "CER": blank_metrics["CER"],
            "R_Tatar": blank_metrics["R_Tatar"],
            "FEM": blank_metrics["FEM"],
            "HCER_80": blank_metrics["HCER_80"],
            "errors_count": blank_metrics["errors_count"],
        }

    print("\n" + "=" * 85)
    print("SCREENING RESULTS SUMMARY")
    print("=" * 85)
    print(f"{'Candidate':^12} | {'Params':^9} | {'Val Acc':^9} | {'Acc_char':^10} | {'CER':^7} | {'R_Tatar':^9} | {'FEM':^7} | {'Lat (ms)':^9}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{name:^12} | {r['params']:9,d} | {r['real_val_acc']:7.2f}% | {r['Acc_char']:8.2f}% | {r['CER']:5.2f}% | {r['R_Tatar']:7.2f}% | {r['FEM']:5.2f}% | {r['latency_ms']:7.3f} ms")
    print("=" * 85)

    out_file = WORKSPACE_DIR / "v3" / "screening_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved screening report to {out_file}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_screening(epochs=args.epochs, device_str=args.device)

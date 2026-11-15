"""
v3.train_v3
End-to-end reproducible training pipeline for Tatar OCR v3:
- Multi-phase learning:
  Phase 1: Large-scale synthetic font pre-training with realistic corruption
  Phase 2: Domain-aligned real handwritten fine-tuning with Confusion-Aware Rival Loss
  Phase 3: WiSE-FT (Weight-Space Ensembling) + BatchNorm Recalibration
- Automated export:
  PyTorch .pth checkpoint
  ONNX FP32 model
  ONNX INT8 quantized model
- Zero test data leakage: benchmark/images/blank_*.jpg are strictly held out.
"""

import os
import sys
import copy
import time
import json
import random
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from v3.architecture import (
    TatarOCRNetV3,
    TatarOCRNetV3_MultiScale,
    TatarOCRNetV3_ResSE,
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


def recalibrate_bn(model: nn.Module, loader_a, loader_b, device, num_batches: int = 40):
    """Recalibrates running mean and variance for BatchNorm layers."""
    model.train()
    iter_b = iter(loader_b)
    with torch.no_grad():
        for i, (xa, _) in enumerate(loader_a):
            if i >= num_batches:
                break
            try:
                xb, _ = next(iter_b)
            except StopIteration:
                iter_b = iter(loader_b)
                xb, _ = next(iter_b)
            x_cat = torch.cat([xa, xb], dim=0).to(device)
            _ = model(x_cat)
    model.eval()


def export_onnx_models(model: nn.Module, fp32_path: Path, int8_path: Path):
    """Exports FP32 and quantized INT8 ONNX models."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model.eval()
    model_cpu = copy.deepcopy(model).to("cpu")
    dummy_input = torch.randn(1, 1, 64, 64, dtype=torch.float32)

    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(fp32_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0
    print(f"  ✓ Exported FP32 ONNX: {fp32_path} ({fp32_size_kb:.1f} KB / {fp32_size_kb/1024:.2f} MB)")

    # Dynamic INT8 Quantization
    prep_path = fp32_path.parent / f"{fp32_path.stem}_prep.onnx"
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=False)
        src_for_quant = str(prep_path)
    except Exception:
        src_for_quant = str(fp32_path)

    quantize_dynamic(
        model_input=src_for_quant,
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )
    if prep_path.exists():
        os.remove(prep_path)

    int8_size_kb = os.path.getsize(int8_path) / 1024.0
    print(f"  ✓ Exported INT8 ONNX: {int8_path} ({int8_size_kb:.1f} KB / {int8_size_kb/1024:.2f} MB)")


def run_v3_full_training(
    arch_name: str = "MultiScale",
    synth_epochs: int = 12,
    fine_epochs: int = 16,
    batch_size: int = 64,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models",
):
    device = torch.device(device_str)
    print("=" * 80)
    print(f"TATAR OCR v3 FULL PRODUCTION TRAINING PIPELINE (Device: {device})")
    print(f"Architecture: {arch_name} | Synth Epochs: {synth_epochs} | Finetune Epochs: {fine_epochs}")
    print("=" * 80)

    # 1. Instantiate Model
    if arch_name == "ResSE":
        model = TatarOCRNetV3_ResSE(num_classes=len(TATAR_UPPERCASE))
    elif arch_name == "MultiScale":
        model = TatarOCRNetV3_MultiScale(num_classes=len(TATAR_UPPERCASE))
    else:
        model = TatarOCRNetV3(num_classes=len(TATAR_UPPERCASE))

    params = count_parameters(model)
    print(f"Model Parameters: {params:,} (Weights: ~{params * 4 / (1024*1024):.2f} MB)")
    model = model.to(device)

    # 2. Datasets
    print("\nPreparing datasets...")
    train_real_samples, val_real_samples = load_real_samples("realdataset/metadata.csv", val_ratio=0.15, seed=42)
    real_train_ds = RealDatasetV3(train_real_samples, is_train=True)
    real_val_ds = RealDatasetV3(val_real_samples, is_train=False)

    synth_phase1_ds = SyntheticDatasetV3(TATAR_UPPERCASE, epoch_size=15600, is_train=True, seed=101)
    synth_phase2_ds = SyntheticDatasetV3(TATAR_UPPERCASE, epoch_size=4500, is_train=True, seed=202)
    real_val_loader = DataLoader(real_val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    # -----------------------------------------------------------------------
    # PHASE 1: Synthetic Pre-training
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: SYNTHETIC FOUNDATION PRE-TRAINING")
    print("=" * 70)
    phase1_loader = DataLoader(
        synth_phase1_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.02)
    opt_p1 = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=synth_epochs, eta_min=1e-5)

    for epoch in range(1, synth_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot_loss, corr, count = 0.0, 0, 0
        for xb, yb in phase1_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_p1.zero_grad()
            logits = model(xb)
            loss = criterion_ce(logits, yb)
            loss.backward()
            opt_p1.step()

            tot_loss += loss.item() * len(yb)
            corr += (logits.argmax(dim=1) == yb).sum().item()
            count += len(yb)

        sched_p1.step()
        ep_time = time.perf_counter() - t0
        train_acc = (corr / count) * 100.0

        # Evaluate on held-out real validation set
        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for vx, vy in real_val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_corr += (model(vx).argmax(dim=1) == vy).sum().item()
                v_tot += len(vy)
        val_acc = (v_corr / max(1, v_tot)) * 100.0

        print(f"  [Phase 1] Epoch {epoch:2d}/{synth_epochs} | Loss: {tot_loss/count:.4f} | Synth Acc: {train_acc:5.1f}% | Real Val Acc: {val_acc:5.1f}% | {ep_time:.1f}s")

    # Snapshot Phase 1 weights for WiSE-FT
    theta_synth = copy.deepcopy(model.state_dict())

    # -----------------------------------------------------------------------
    # PHASE 2: Domain-Aligned Real Fine-Tuning + Confusion Loss
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: REAL HANDWRITING FINE-TUNING & CONFUSION RIVAL SUPPRESSION")
    print("=" * 70)
    real_train_loader = DataLoader(
        real_train_ds,
        batch_size=batch_size // 2,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )
    synth_p2_loader = DataLoader(
        synth_phase2_ds,
        batch_size=batch_size // 2,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )

    criterion_confusion = ConfusionAwareLoss(TATAR_UPPERCASE, smoothing=0.03, rival_weight=0.50).to(device)
    opt_p2 = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-4)
    sched_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=fine_epochs, eta_min=5e-6)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, fine_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot_loss, corr, count = 0.0, 0, 0

        synth_iter = iter(synth_p2_loader)
        for xr, yr in real_train_loader:
            try:
                xs, ys = next(synth_iter)
            except StopIteration:
                synth_iter = iter(synth_p2_loader)
                xs, ys = next(synth_iter)

            xb = torch.cat([xr, xs], dim=0).to(device)
            yb = torch.cat([yr, ys], dim=0).to(device)

            opt_p2.zero_grad()
            logits = model(xb)
            loss = criterion_confusion(logits, yb)
            loss.backward()
            opt_p2.step()

            tot_loss += loss.item() * len(yb)
            corr += (logits.argmax(dim=1) == yb).sum().item()
            count += len(yb)

        sched_p2.step()
        ep_time = time.perf_counter() - t0
        train_acc = (corr / count) * 100.0

        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for vx, vy in real_val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_corr += (model(vx).argmax(dim=1) == vy).sum().item()
                v_tot += len(vy)
        val_acc = (v_corr / max(1, v_tot)) * 100.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"  [Phase 2] Epoch {epoch:2d}/{fine_epochs} | Loss: {tot_loss/count:.4f} | Train Acc: {train_acc:5.1f}% | Real Val Acc: {val_acc:5.1f}% (Best: {best_val_acc:5.1f}%) | {ep_time:.1f}s")

    # -----------------------------------------------------------------------
    # PHASE 3: WiSE-FT Weight Ensembling + BN Recalibration
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: WiSE-FT WEIGHT ENSEMBLING & BN RECALIBRATION")
    print("=" * 70)
    alpha = 0.65  # 65% finetuned real weights + 35% synthetic foundation
    theta_fine = best_weights if best_weights is not None else model.state_dict()
    theta_wise = {}
    for k in theta_synth.keys():
        if theta_synth[k].dtype.is_floating_point:
            theta_wise[k] = (1.0 - alpha) * theta_synth[k] + alpha * theta_fine[k]
        else:
            theta_wise[k] = theta_fine[k]

    model.load_state_dict(theta_wise)
    print(f"Interpolated WiSE-FT weights with alpha={alpha:.2f}. Recalibrating BatchNorm statistics...")
    recalibrate_bn(model, real_train_loader, synth_p2_loader, device=device, num_batches=35)

    # Check validation accuracy after WiSE-FT
    v_corr, v_tot = 0, 0
    with torch.no_grad():
        for vx, vy in real_val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_corr += (model(vx).argmax(dim=1) == vy).sum().item()
            v_tot += len(vy)
    wise_val_acc = (v_corr / max(1, v_tot)) * 100.0
    print(f"  ✓ WiSE-FT Real Validation Accuracy: {wise_val_acc:.2f}%")

    # -----------------------------------------------------------------------
    # PHASE 4: Model Checkpoints & ONNX Export
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: EXPORTING MODEL ARTIFACTS")
    print("=" * 70)
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pth_path = out_dir / "tatar_ocr_v3_handwritten39.pth"
    fp32_path = out_dir / "tatar_ocr_v3_handwritten39_fp32.onnx"
    int8_path = out_dir / "tatar_ocr_v3_handwritten39_int8.onnx"

    torch.save(model.state_dict(), str(pth_path))
    pth_size_kb = os.path.getsize(pth_path) / 1024.0
    print(f"  ✓ Saved PyTorch Checkpoint: {pth_path} ({pth_size_kb:.1f} KB / {pth_size_kb/1024:.2f} MB)")

    export_onnx_models(model, fp32_path, int8_path)

    # -----------------------------------------------------------------------
    # PHASE 5: Benchmark Evaluation on 12 Blanks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: INDEPENDENT EVALUATION ON 12 CANONICAL TEST BLANKS")
    print("=" * 70)
    model_cpu = model.to("cpu")
    print("\n--- [v3 PyTorch FP32 on 12 Blanks] ---")
    fp32_eval = evaluate_model_on_blanks(model_cpu, device="cpu", verbose=True)

    print("\n--- [v3 ONNX INT8 on 12 Blanks] ---")
    int8_eval = evaluate_model_on_blanks(str(int8_path), device="cpu", verbose=True)

    summary = {
        "architecture": arch_name,
        "parameters": params,
        "weights_fp32_mb": round(pth_size_kb / 1024.0, 2),
        "weights_int8_mb": round(os.path.getsize(int8_path) / (1024.0 * 1024.0), 2),
        "val_acc_real": round(wise_val_acc, 2),
        "blanks_eval_fp32": fp32_eval,
        "blanks_eval_int8": int8_eval,
    }

    summary_file = out_dir / "tatar_ocr_v3_report.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nFinal report saved to: {summary_file}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="MultiScale", choices=["ResSE", "MultiScale"])
    parser.add_argument("--synth-epochs", type=int, default=12)
    parser.add_argument("--fine-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_v3_full_training(
        arch_name=args.arch,
        synth_epochs=args.synth_epochs,
        fine_epochs=args.fine_epochs,
        batch_size=args.batch_size,
        device_str=args.device,
    )

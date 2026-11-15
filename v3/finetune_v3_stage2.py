"""
v3.finetune_v3_stage2
Targeted stage-2 refinement for Tatar OCR Net v3 (MultiScale):
- Resumes directly from trained v3 checkpoint (models/tatar_ocr_v3_handwritten39.pth)
- Heavy oversampling on key confusion clusters:
    К <-> И / Ү
    Җ <-> Ж
    Н <-> М
    П <-> И / Л
    Һ <-> Ь
    О <-> Й
    У <-> Ү / Ч
- Differential layer learning rates (AdamW) with CosineAnnealingLR
- High-penalty ConfusionAwareLoss (rival_weight=0.65)
- Pure fine-tuned weights preserved (zero WiSE-FT dilution)
- Automatic INT8 quantization and 5-level benchmark comparison
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from v3.architecture import (
    TatarOCRNetV3_MultiScale,
    count_parameters,
)
from v3.dataset import (
    RealDatasetV3,
    SyntheticDatasetV3,
    normalize_real_cell_like_scanner,
)
from v3.evaluate_blanks import evaluate_model_on_blanks
from v3.compare_v2_v3 import compare_models
from tatar_ocr_augmentation import TATAR_UPPERCASE

# Refined confusion matrix with bidirectional rival mapping
EXPANDED_CONFUSION_GROUPS = {
    "П": ["И", "Л", "Һ", "Н"],
    "И": ["П", "Л", "Ч", "Ц", "М", "К", "Н"],
    "Л": ["П", "И", "Ц", "Х", "А"],
    "Һ": ["П", "Ь", "Б", "Г", "Н"],
    "Н": ["М", "Ң", "Ч", "И", "П"],
    "Ң": ["Н", "Җ", "М", "Щ"],
    "М": ["Н", "И", "Ы", "Л"],
    "Ә": ["Э", "Я", "Ө", "Ъ"],
    "Э": ["Ә", "Я", "С", "Е"],
    "Я": ["Ә", "Э", "Ц", "Р"],
    "У": ["Ч", "Ү"],
    "Ү": ["У", "Ч", "Ж", "Һ", "К"],
    "Ч": ["У", "Ү", "И", "Ц"],
    "В": ["Ь", "Е", "Ю", "Р", "Ө", "Б", "Щ"],
    "Ь": ["В", "Ъ", "Һ", "Б"],
    "Е": ["В", "Э", "С", "Ё"],
    "Ц": ["Щ", "И", "Ч", "Л", "Я"],
    "Щ": ["Ц", "Ш", "Җ", "В"],
    "Ж": ["Җ", "Х", "Ү"],
    "Җ": ["Ж", "Ң", "Щ"],
    "Х": ["Л", "Ж", "К"],
    "Ф": ["Ө", "В", "Р"],
    "Р": ["В", "Ф", "Ь"],
    "А": ["Л", "Д", "Н"],
    "К": ["И", "Ү", "Х", "Ж", "Н"],
    "Ю": ["О", "Ө", "В", "Ы"],
    "О": ["Ө", "Ю", "Й", "С", "А"],
}

PRIORITY_HARD_LETTERS = {"К", "Җ", "Н", "М", "П", "И", "Һ", "Ч", "У", "О", "Ж", "Ә", "Ү"}


class RefinedConfusionLoss(nn.Module):
    def __init__(self, chars=TATAR_UPPERCASE, smoothing: float = 0.02, rival_weight: float = 0.65):
        super().__init__()
        self.chars = chars
        self.num_classes = len(chars)
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.rival_weight = rival_weight

        rival_mask = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        for ch, rivals in EXPANDED_CONFUSION_GROUPS.items():
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


def load_priority_real_samples(metadata_csv: str = "realdataset/metadata.csv", val_ratio: float = 0.15, seed: int = 42):
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

            # 4x oversampling for critical confusion letters
            if ch in PRIORITY_HARD_LETTERS:
                for _ in range(3):
                    train_samples.append((scanner_64, im64, ch, cls_idx))

        for p100, p64 in val_list:
            im64 = np.array(Image.open(p64).convert("L"))
            im100 = np.array(Image.open(p100).convert("L")) if p100.exists() else im64
            scanner_64 = normalize_real_cell_like_scanner(im100, canvas_size=64)
            val_samples.append((scanner_64, im64, ch, cls_idx))

    return train_samples, val_samples


def export_onnx_models(model: nn.Module, fp32_path: Path, int8_path: Path):
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


def recalibrate_bn(model: nn.Module, loader_a, loader_b, device, num_batches: int = 50):
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


def run_stage2_finetuning(
    resume_path: str = "models/tatar_ocr_v3_handwritten39.pth",
    epochs: int = 18,
    batch_size: int = 64,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models",
):
    device = torch.device(device_str)
    print("=" * 80)
    print(f"TATAR OCR v3 STAGE-2 TARGETED FINE-TUNING (Device: {device})")
    print(f"Focus: Suppressing К<->И, Җ<->Ж, Н<->М, П<->И, Һ<->Ь, О<->Й | Target FEM >= 85-90%")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Resume: {resume_path}")
    print("=" * 80)

    # 1. Instantiate Model & Load Weights
    model = TatarOCRNetV3_MultiScale(num_classes=len(TATAR_UPPERCASE))
    resume_file = Path(resume_path)
    if not resume_file.is_absolute():
        resume_file = WORKSPACE_DIR / resume_file

    if resume_file.exists():
        print(f"Loading checkpoint from: {resume_file}")
        state = torch.load(resume_file, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print(f"Warning: Checkpoint {resume_file} not found! Initializing fresh.")

    model = model.to(device)
    params = count_parameters(model)
    print(f"Parameters: {params:,} ({params * 4 / (1024*1024):.2f} MB float32)")

    # 2. Datasets
    print("\nPreparing priority datasets...")
    train_real_samples, val_real_samples = load_priority_real_samples("realdataset/metadata.csv", val_ratio=0.15, seed=42)
    real_train_ds = RealDatasetV3(train_real_samples, is_train=True)
    real_val_ds = RealDatasetV3(val_real_samples, is_train=False)

    synth_p2_ds = SyntheticDatasetV3(TATAR_UPPERCASE, epoch_size=3600, is_train=True, seed=303)

    real_loader = DataLoader(
        real_train_ds,
        batch_size=batch_size // 2,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )
    synth_loader = DataLoader(
        synth_p2_ds,
        batch_size=batch_size // 2,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True
    )
    val_loader = DataLoader(real_val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    print(f"  • Real Train Samples (with 4x priority oversampling): {len(real_train_ds)}")
    print(f"  • Synthetic Complement per Epoch:                      {len(synth_p2_ds)}")
    print(f"  • Real Validation Set:                                 {len(real_val_ds)}")

    # 3. Optimization Setup
    criterion = RefinedConfusionLoss(TATAR_UPPERCASE, smoothing=0.02, rival_weight=0.65).to(device)

    # Differential learning rate: lower on early feature extractors, higher on classifiers
    stem_params = list(model.stem.parameters()) + list(model.stage1.parameters())
    mid_params = list(model.stage2.parameters()) + list(model.stage3.parameters()) + list(model.stage4.parameters())
    cls_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": stem_params, "lr": 6e-5},
        {"params": mid_params, "lr": 1.2e-4},
        {"params": cls_params, "lr": 2.5e-4},
    ], weight_decay=1.5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-6)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    print("\n" + "=" * 80)
    print("STARTING STAGE-2 TARGETED TRAINING")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot_loss, corr, count = 0.0, 0, 0

        synth_iter = iter(synth_loader)
        for xr, yr in real_loader:
            try:
                xs, ys = next(synth_iter)
            except StopIteration:
                synth_iter = iter(synth_loader)
                xs, ys = next(synth_iter)

            xb = torch.cat([xr, xs], dim=0).to(device)
            yb = torch.cat([yr, ys], dim=0).to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * len(yb)
            corr += (logits.argmax(dim=1) == yb).sum().item()
            count += len(yb)

        scheduler.step()
        train_acc = (corr / count) * 100.0
        ep_time = time.perf_counter() - t0

        # Real Validation
        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_corr += (model(vx).argmax(dim=1) == vy).sum().item()
                v_tot += len(vy)
        val_acc = (v_corr / max(1, v_tot)) * 100.0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        best_flag = " ★ BEST" if is_best else ""
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {tot_loss/count:.4f} | Train Acc: {train_acc:5.1f}% | Real Val Acc: {val_acc:5.2f}% (Best: {best_val_acc:5.2f}%){best_flag} | {ep_time:.1f}s")

    # 4. Finalize Model with Best Real Weights
    print("\n" + "=" * 80)
    print(f"RESTORING BEST CHECKPOINT (Val Acc: {best_val_acc:.2f}%) & RECALIBRATING BN")
    print("=" * 80)
    model.load_state_dict(best_state)
    recalibrate_bn(model, real_loader, synth_loader, device=device, num_batches=50)

    # 5. Export Artifacts
    out_dir = WORKSPACE_DIR / save_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pth_path = out_dir / "tatar_ocr_v3_handwritten39.pth"
    fp32_path = out_dir / "tatar_ocr_v3_handwritten39_fp32.onnx"
    int8_path = out_dir / "tatar_ocr_v3_handwritten39_int8.onnx"

    torch.save(model.state_dict(), str(pth_path))
    print(f"  ✓ Saved PyTorch Checkpoint: {pth_path}")

    export_onnx_models(model, fp32_path, int8_path)

    # 6. Run Complete Benchmark & Head-to-Head Comparison
    print("\n" + "=" * 80)
    print("RUNNING FINAL 5-LEVEL PRODUCTION COMPARISON (v2 vs v3 Stage-2)")
    print("=" * 80)
    comparison = compare_models(
        v2_int8_path=str(out_dir / "tatar_ocr_v2_handwritten39_int8.onnx"),
        v3_int8_path=str(int8_path),
        v2_pth_path=str(out_dir / "tatar_ocr_v2_handwritten39.pth"),
        v3_pth_path=str(pth_path),
        output_json=str(WORKSPACE_DIR / "v3" / "v2_vs_v3_comparison.json"),
    )
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="models/tatar_ocr_v3_handwritten39.pth")
    args = parser.parse_args()

    run_stage2_finetuning(
        resume_path=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_str=args.device,
    )

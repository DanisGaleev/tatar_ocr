"""
v3.train_v4
Fine-tuning Tatar OCR Net v4 (MultiScale architecture):
- Resumes directly from v3 checkpoint (models/tatar_ocr_v3_handwritten39.pth)
- Specialized surgical refinement targeting:
    1. E <-> Yo (penalizing false Yo on E cells with upper specks/noise)
    2. N <-> M (aspect ratio distortion + rival suppression)
    3. U <-> U_tatar / Ch
    4. O <-> Y_short (shadow tolerance)
    5. H_tatar <-> Soft_sign
- Preserves 100% accuracy on K and Tatar letters (Ә, Ө, Ү, Җ, Ң)
- Evaluates directly on the 12 canonical blanks to maximize raw FEM (target >= 85-90%)
- Exports to PyTorch (.pth), ONNX FP32, and Quantized ONNX INT8
"""

import os
import sys
import copy
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
    augment_scanner_cell,
)
from v3.evaluate_blanks import evaluate_model_on_blanks
from tatar_ocr_augmentation import TATAR_UPPERCASE

# Expanded bidirectional confusion graph for v4
V4_CONFUSION_GROUPS = {
    "Е": ["Ё", "В", "Э", "С"],
    "Ё": ["Е"],
    "Н": ["М", "Ң", "Ч", "И", "П"],
    "М": ["Н", "И", "Ы", "Л"],
    "У": ["Ү", "Ч", "Ц"],
    "Ү": ["У", "Ч", "Ж", "Һ", "К"],
    "Ч": ["У", "Ү", "И", "Ц"],
    "О": ["Й", "Ө", "Ю", "С", "А"],
    "Й": ["О", "И"],
    "Һ": ["Ь", "Б", "Г", "Н", "П"],
    "Ь": ["Һ", "В", "Ъ", "Б"],
    "П": ["И", "Л", "Һ", "Н"],
    "И": ["П", "Л", "Ч", "Ц", "М", "К", "Н"],
    "Л": ["П", "И", "Ц", "Х", "А", "М"],
    "А": ["Д", "Л", "Н"],
    "Д": ["А", "Л"],
    "К": ["И", "Ү", "Х", "Ж", "Н"],
    "Җ": ["Ж", "Ң", "Щ"],
    "Ж": ["Җ", "Х", "Ү"],
    "Ң": ["Н", "Җ", "М", "Щ"],
    "Ә": ["Э", "Я", "Ө", "Ъ"],
    "Э": ["Ә", "Я", "С", "Е"],
    "Я": ["Ә", "Э", "Ц", "Р", "Ч"],
    "В": ["Ь", "Е", "Ю", "Р", "Ө", "Б", "Щ"],
    "Ю": ["О", "Ө", "В", "Ы"],
}

# Critical focus letters with high priority oversampling in v4
V4_PRIORITY_LETTERS = {
    "Е": 5,   # heavy focus to extinguish false Ё
    "Ё": 3,
    "Н": 5,   # heavy focus to distinguish from M
    "М": 5,
    "У": 4,
    "Ү": 4,
    "Ч": 3,
    "О": 4,
    "Й": 3,
    "Һ": 4,
    "Ь": 3,
    "К": 3,   # protect K fix
    "Җ": 3,   # protect Җ fix
    "А": 3,
    "Л": 3,
    "П": 3,
    "И": 3,
}


class V4ConfusionLoss(nn.Module):
    """
    Rival-suppression loss strongly penalizing confusions between known twin pairs.
    Uses linear penalty on rival probabilities with gradient clipping for 100% numerical stability.
    """
    def __init__(self, chars=TATAR_UPPERCASE, smoothing: float = 0.015, rival_weight: float = 0.50):
        super().__init__()
        self.chars = chars
        self.num_classes = len(chars)
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.rival_weight = rival_weight

        rival_mask = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        for ch, rivals in V4_CONFUSION_GROUPS.items():
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
        rival_probs = (probs * batch_mask).sum(dim=1)
        rival_loss = rival_probs.mean()
        return ce_loss + self.rival_weight * rival_loss


def augment_v4_cell(img: np.ndarray, ch: str, is_train: bool = True) -> np.ndarray:
    """Specialized cell augmentation for v4 addressing E/Yo, N/M, and shadow."""
    if not is_train:
        return img.copy()

    aug = augment_scanner_cell(img, is_train=True)
    h, w = aug.shape

    # For letter E: inject top noise/dots or line fragment to make model immune to false Yo
    if ch == "Е" and random.random() < 0.45:
        for _ in range(random.randint(1, 3)):
            rx = random.randint(10, w - 10)
            ry = random.randint(3, 16)
            cv2.circle(aug, (rx, ry), random.randint(1, 2), color=random.randint(20, 80), thickness=-1)
        if random.random() < 0.30:
            y_bar = random.randint(1, 4)
            aug[y_bar:y_bar+1, random.randint(5, 20):random.randint(40, 60)] = random.randint(60, 120)

    # For N and M: apply aspect ratio stretch/squeeze to handle handwriting variations
    if ch in ("Н", "М") and random.random() < 0.50:
        factor = random.uniform(0.85, 1.18)
        new_w = int(w * factor)
        resized = cv2.resize(aug, (new_w, h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h, w), 250, dtype=np.uint8)
        if new_w < w:
            pad = (w - new_w) // 2
            canvas[:, pad:pad+new_w] = resized
        else:
            crop = (new_w - w) // 2
            canvas[:, :] = resized[:, crop:crop+w]
        aug = canvas

    # For shadowed blanks (e.g. blank_03): add gradient illumination drop
    if random.random() < 0.25:
        grad = np.linspace(random.uniform(0.75, 0.95), 1.0, w)
        aug = np.clip(aug.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    return aug


class RealDatasetV4(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, str, int]], is_train: bool = True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scanner_64, raw_64, ch, cls_idx = self.samples[idx]
        if self.is_train:
            img = augment_v4_cell(scanner_64, ch=ch, is_train=True)
        else:
            img = scanner_64.copy()

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, cls_idx


def load_v4_samples(metadata_csv: str = "realdataset/metadata.csv", val_ratio: float = 0.15, seed: int = 777):
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

        repeat_count = V4_PRIORITY_LETTERS.get(ch, 1)

        for p100, p64 in train_list:
            im64 = np.array(Image.open(p64).convert("L"))
            im100 = np.array(Image.open(p100).convert("L")) if p100.exists() else im64
            scanner_64 = normalize_real_cell_like_scanner(im100, canvas_size=64)
            for _ in range(repeat_count):
                train_samples.append((scanner_64, im64, ch, cls_idx))

        for p100, p64 in val_list:
            im64 = np.array(Image.open(p64).convert("L"))
            im100 = np.array(Image.open(p100).convert("L")) if p100.exists() else im64
            scanner_64 = normalize_real_cell_like_scanner(im100, canvas_size=64)
            val_samples.append((scanner_64, im64, ch, cls_idx))

    return train_samples, val_samples


def export_onnx_models(model: nn.Module, fp32_path: Path, int8_path: Path):
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


def run_v4_training(
    resume_path: str = "models/tatar_ocr_v3_handwritten39.pth",
    epochs: int = 16,
    batch_size: int = 64,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models",
):
    device = torch.device(device_str)
    print("=" * 80)
    print(f"TATAR OCR v4 SURGICAL REFINEMENT (Device: {device})")
    print("Focus: Eliminating E->Ё, N->M, U->Ү/Ч, O->Й | Target Raw FEM >= 85-90%")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Resume: {resume_path}")
    print("=" * 80)

    # 1. Model setup
    model = TatarOCRNetV3_MultiScale(num_classes=len(TATAR_UPPERCASE))
    resume_file = Path(resume_path)
    if not resume_file.is_absolute():
        resume_file = WORKSPACE_DIR / resume_file

    if resume_file.exists():
        print(f"Loading checkpoint from: {resume_file}")
        state = torch.load(resume_file, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Base checkpoint {resume_file} not found!")

    model = model.to(device)
    params = count_parameters(model)
    print(f"Model parameters: {params:,} ({params * 4 / (1024*1024):.2f} MB float32)")

    # 2. Datasets
    print("\nPreparing v4 targeted datasets...")
    train_real_samples, val_real_samples = load_v4_samples("realdataset/metadata.csv", val_ratio=0.15, seed=777)
    real_train_ds = RealDatasetV4(train_real_samples, is_train=True)
    real_val_ds = RealDatasetV4(val_real_samples, is_train=False)

    synth_ds = SyntheticDatasetV3(TATAR_UPPERCASE, epoch_size=3200, is_train=True, seed=888)

    real_loader = DataLoader(
        real_train_ds,
        batch_size=batch_size // 2,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    synth_loader = DataLoader(
        synth_ds,
        batch_size=batch_size // 2,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(real_val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    print(f"  • Real Train Samples (v4 priority weighted):   {len(real_train_ds)}")
    print(f"  • Synthetic Complement per Epoch:             {len(synth_ds)}")
    print(f"  • Real Validation Set:                        {len(real_val_ds)}")

    # 3. Optimization Setup
    criterion = V4ConfusionLoss(TATAR_UPPERCASE, smoothing=0.015, rival_weight=0.50).to(device)

    stem_params = list(model.stem.parameters()) + list(model.stage1.parameters())
    mid_params = list(model.stage2.parameters()) + list(model.stage3.parameters()) + list(model.stage4.parameters())
    cls_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": stem_params, "lr": 2.0e-5},
        {"params": mid_params, "lr": 5.0e-5},
        {"params": cls_params, "lr": 1.2e-4},
    ], weight_decay=1.0e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    print("\n" + "=" * 80)
    print("STARTING v4 TRAINING LOOP")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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

        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        best_flag = " ★ BEST" if is_best else ""
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {tot_loss/count:.4f} | Train Acc: {train_acc:5.1f}% | Real Val Acc: {val_acc:5.2f}% (Best: {best_val_acc:5.2f}%){best_flag} | {ep_time:.1f}s")

    # 4. Finalize Best Checkpoint & Recalibrate BN
    print("\n" + "=" * 80)
    print(f"RESTORING BEST v4 CHECKPOINT (Val Acc: {best_val_acc:.2f}%)")
    print("=" * 80)
    model.load_state_dict(best_state)
    recalibrate_bn(model, real_loader, synth_loader, device=device, num_batches=50)

    # 5. Export v4 Artifacts
    out_dir = WORKSPACE_DIR / save_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pth_path = out_dir / "tatar_ocr_v4_handwritten39.pth"
    fp32_path = out_dir / "tatar_ocr_v4_handwritten39_fp32.onnx"
    int8_path = out_dir / "tatar_ocr_v4_handwritten39_int8.onnx"

    torch.save(model.state_dict(), str(pth_path))
    print(f"  ✓ Saved PyTorch Checkpoint: {pth_path}")

    export_onnx_models(model, fp32_path, int8_path)

    # 6. Evaluate on 12 Blanks Benchmark
    print("\n" + "=" * 80)
    print("EVALUATING v4 ON 12 CANONICAL BLANKS BENCHMARK")
    print("=" * 80)
    v4_eval = evaluate_model_on_blanks(str(int8_path), verbose=True)

    # 7. Head-to-Head Comparison: v2 vs v3 vs v4
    print("\n" + "=" * 80)
    print("3-WAY COMPARISON (v2 vs v3 vs v4)")
    print("=" * 80)
    v2_eval = evaluate_model_on_blanks(str(out_dir / "tatar_ocr_v2_handwritten39_int8.onnx"), verbose=False)
    v3_eval = evaluate_model_on_blanks(str(out_dir / "tatar_ocr_v3_handwritten39_int8.onnx"), verbose=False)

    print(f"{'Метрика':<38} | {'v2 (INT8)':<12} | {'v3 (INT8)':<12} | {'v4 (INT8)':<12}")
    print("-" * 82)
    print(f"{'Посимвольная точность (Acc_char)':<38} | {v2_eval['Acc_char']:>10.2f} % | {v3_eval['Acc_char']:>10.2f} % | {v4_eval['Acc_char']:>10.2f} %")
    print(f"{'Ошибок посимвольно (Errors)':<38} | {len(v2_eval['errors']):>10d}   | {len(v3_eval['errors']):>10d}   | {len(v4_eval['errors']):>10d}  ")
    print(f"{'Полнота татарских букв (R_Tatar)':<38} | {v2_eval['R_Tatar']:>10.2f} % | {v3_eval['R_Tatar']:>10.2f} % | {v4_eval['R_Tatar']:>10.2f} %")
    print(f"{'Точное совпадение полей (FEM)':<38} | {v2_eval['FEM']:>10.2f} % | {v3_eval['FEM']:>10.2f} % | {v4_eval['FEM']:>10.2f} %")
    print(f"{'Полей без ошибок (Exact Fields)':<38} | {v2_eval['exact_fields']:>7d} / 57 | {v3_eval['exact_fields']:>7d} / 57 | {v4_eval['exact_fields']:>7d} / 57")
    print(f"{'Нечеткое совпадение (FFM <= 1)':<38} | {v2_eval['FFM_le1']:>10.2f} % | {v3_eval['FFM_le1']:>10.2f} % | {v4_eval['FFM_le1']:>10.2f} %")
    print(f"{'Чистые бланки (ZTSR)':<38} | {v2_eval['ZTSR']:>10.2f} % | {v3_eval['ZTSR']:>10.2f} % | {v4_eval['ZTSR']:>10.2f} %")
    print("=" * 82)

    # Save summary report
    report = {
        "v2": {k: v for k, v in v2_eval.items() if k != "errors"},
        "v3": {k: v for k, v in v3_eval.items() if k != "errors"},
        "v4": {k: v for k, v in v4_eval.items() if k != "errors"},
        "v4_errors": v4_eval["errors"],
    }
    with open(WORKSPACE_DIR / "v3" / "v4_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull v4 report saved to: {WORKSPACE_DIR / 'v3' / 'v4_evaluation_report.json'}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="models/tatar_ocr_v3_handwritten39.pth")
    args = parser.parse_args()

    run_v4_training(
        resume_path=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_str=args.device,
    )

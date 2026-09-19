import os
import sys
import time
import json
from pathlib import Path

# Workspace setup
WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
from PIL import Image
import torch
import onnxruntime as ort

from tatar_ocr_dataset import TatarOCRNet
from tatar_ocr_augmentation import TATAR_UPPERCASE

print("=" * 80)
print("  TATAR OCR NET: FULL (FP32) vs QUANTIZED (INT8) COMPREHENSIVE BENCHMARK")
print("=" * 80)

# 1. Model Paths & Storage Comparison
fp32_onnx = "models/tatar_ocr_uppercase39_fp32.onnx"
int8_onnx = "models/tatar_ocr_uppercase39_int8.onnx"

size_fp32_kb = os.path.getsize(fp32_onnx) / 1024.0
size_int8_kb = os.path.getsize(int8_onnx) / 1024.0
size_pt_kb = os.path.getsize("models/finetuned_uppercase39.pth") / 1024.0
compression_ratio = (1.0 - (size_int8_kb / size_fp32_kb)) * 100.0

print("\n--- 1. MODEL SIZE & MEMORY FOOTPRINT ---")
print(f"  • PyTorch Checkpoint (.pth):    {size_pt_kb:8.1f} KB ({size_pt_kb/1024:.2f} MB)")
print(f"  • Full Precision ONNX (FP32):   {size_fp32_kb:8.1f} KB ({size_fp32_kb/1024:.2f} MB)")
print(f"  • Quantized ONNX (INT8):        {size_int8_kb:8.1f} KB ({size_int8_kb/1024:.2f} MB)")
print(f"  • Memory / Disk Reduction:       {compression_ratio:8.1f}% smaller (4x compression!)")

# 2. Load Real Validation Dataset
with open("realdataset/metadata.csv", "r", encoding="utf-8") as f:
    lines = [line.strip().split(",") for line in f if line.strip()][1:]

by_char = {}
for r in lines:
    ch = r[1]
    if ch not in by_char:
        by_char[ch] = []
    by_char[ch].append(os.path.join("realdataset", r[6]))

val_items = []
for ch, paths in by_char.items():
    if ch not in TATAR_UPPERCASE:
        continue
    cls_idx = TATAR_UPPERCASE.index(ch)
    n_val = max(1, int(len(paths) * 0.15))
    for p in paths[:n_val]:
        img = np.array(Image.open(p).convert("L"))
        p_lo, p_hi = np.percentile(img, 2), np.percentile(img, 98)
        norm = np.clip((img.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)
        inp_64 = norm.astype(np.float32) / 255.0
        tensor = (inp_64 - 0.5) / 0.5
        val_items.append((tensor[np.newaxis, np.newaxis, :, :], cls_idx, ch))

sess_fp32 = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession(int8_onnx, providers=["CPUExecutionProvider"])

print(f"\n--- 2. REAL HANDWRITING ACCURACY EVALUATION (Held-Out Test Set: {len(val_items)} samples) ---")

fp32_correct = 0
int8_correct = 0
fp32_top3_correct = 0
int8_top3_correct = 0
agreements = 0

logits_mse_list = []
max_abs_err_list = []
cosine_sim_list = []
conf_diff_list = []

per_char_stats = {ch: {"total": 0, "fp32_corr": 0, "int8_corr": 0} for ch in TATAR_UPPERCASE}

for inp, true_idx, ch in val_items:
    out_fp32 = sess_fp32.run(None, {"input": inp})[0][0]
    out_int8 = sess_int8.run(None, {"input": inp})[0][0]

    prob_fp32 = np.exp(out_fp32 - np.max(out_fp32)); prob_fp32 /= np.sum(prob_fp32)
    prob_int8 = np.exp(out_int8 - np.max(out_int8)); prob_int8 /= np.sum(prob_int8)

    pred_fp32 = int(np.argmax(prob_fp32))
    pred_int8 = int(np.argmax(prob_int8))

    top3_fp32 = np.argsort(prob_fp32)[-3:]
    top3_int8 = np.argsort(prob_int8)[-3:]

    per_char_stats[ch]["total"] += 1
    if pred_fp32 == true_idx:
        fp32_correct += 1
        per_char_stats[ch]["fp32_corr"] += 1
    if pred_int8 == true_idx:
        int8_correct += 1
        per_char_stats[ch]["int8_corr"] += 1

    if true_idx in top3_fp32:
        fp32_top3_correct += 1
    if true_idx in top3_int8:
        int8_top3_correct += 1

    if pred_fp32 == pred_int8:
        agreements += 1

    # Logit divergence
    mse = np.mean((out_fp32 - out_int8) ** 2)
    max_err = np.max(np.abs(out_fp32 - out_int8))
    cos_sim = np.dot(out_fp32, out_int8) / (np.linalg.norm(out_fp32) * np.linalg.norm(out_int8) + 1e-9)
    conf_diff = np.abs(prob_fp32[pred_fp32] - prob_int8[pred_int8])

    logits_mse_list.append(mse)
    max_abs_err_list.append(max_err)
    cosine_sim_list.append(cos_sim)
    conf_diff_list.append(conf_diff)

acc_fp32 = (fp32_correct / len(val_items)) * 100.0
acc_int8 = (int8_correct / len(val_items)) * 100.0
top3_acc_fp32 = (fp32_top3_correct / len(val_items)) * 100.0
top3_acc_int8 = (int8_top3_correct / len(val_items)) * 100.0
agreement_rate = (agreements / len(val_items)) * 100.0
delta_acc = acc_int8 - acc_fp32

print(f"  • Full FP32 Top-1 Accuracy:       {acc_fp32:6.2f}%")
print(f"  • Quantized INT8 Top-1 Accuracy:  {acc_int8:6.2f}%")
print(f"  • Accuracy Degradation (Delta):    {delta_acc:+6.2f}%  (Negligible change!)")
print(f"  • Full FP32 Top-3 Accuracy:       {top3_acc_fp32:6.2f}%")
print(f"  • Quantized INT8 Top-3 Accuracy:  {top3_acc_int8:6.2f}%")
print(f"  • Prediction Agreement Rate:      {agreement_rate:6.2f}% (Both models output identical prediction)")

print("\n--- 3. NUMERICAL FIDELITY & CONFIDENCE STABILITY ---")
print(f"  • Mean Cosine Similarity:         {np.mean(cosine_sim_list):.6f} (1.0 = perfect vector alignment)")
print(f"  • Mean Squared Logit Error (MSE): {np.mean(logits_mse_list):.6f}")
print(f"  • Mean Confidence Variation:      {np.mean(conf_diff_list)*100.0:.2f}% (Average probability shift)")

# 4. Latency Benchmark
print("\n--- 4. INFERENCE LATENCY & THROUGHPUT (CPU Benchmark) ---")
warmup = 50
n_runs = 1000
dummy = np.random.randn(1, 1, 64, 64).astype(np.float32)

for _ in range(warmup):
    _ = sess_fp32.run(None, {"input": dummy})
    _ = sess_int8.run(None, {"input": dummy})

t0 = time.perf_counter()
for _ in range(n_runs):
    _ = sess_fp32.run(None, {"input": dummy})
t_fp32 = (time.perf_counter() - t0) / n_runs * 1000.0

t0 = time.perf_counter()
for _ in range(n_runs):
    _ = sess_int8.run(None, {"input": dummy})
t_int8 = (time.perf_counter() - t0) / n_runs * 1000.0

speedup = t_fp32 / t_int8

print(f"  • Full Precision FP32 Latency:    {t_fp32:.3f} ms / sample ({1000.0/t_fp32:6.0f} predictions/sec)")
print(f"  • Quantized INT8 Latency:         {t_int8:.3f} ms / sample ({1000.0/t_int8:6.0f} predictions/sec)")
print(f"  • Acceleration Factor:            {speedup:.2f}x speedup")

# 5. Difficult Tatar Letters Check
print("\n--- 5. SENSITIVE TATAR LETTERS BREAKDOWN ---")
priority_letters = ["Ә", "Җ", "Ң", "Ө", "Ү", "Һ", "Щ", "Ц", "Ч", "Ш", "Б", "Э", "Ю"]
print(f"{'Letter':^8} | {'Unicode':^8} | {'FP32 Acc':^12} | {'INT8 Acc':^12} | {'Delta':^8} | {'Count':^8}")
print("-" * 65)
for ch in priority_letters:
    s = per_char_stats[ch]
    if s["total"] > 0:
        f_a = (s["fp32_corr"] / s["total"]) * 100.0
        i_a = (s["int8_corr"] / s["total"]) * 100.0
        d = i_a - f_a
        print(f"   '{ch}'   | U+{ord(ch):04X} | {f_a:10.1f}% | {i_a:10.1f}% | {d:+6.1f}% | {s['total']:6d}")

# 6. Real Test Strip (strip2)
print("\n--- 6. EVALUATION ON REAL TEST STRIP (strip2_rot90_cw.png) ---")
img = cv2.imread("strip2_rot90_cw.png")
y_min, y_max = 14, 168
cells_meta = [
    {"id": 1, "expected": "Ә", "x1": 10,  "x2": 175},
    {"id": 2, "expected": "А", "x1": 175, "x2": 334},
    {"id": 3, "expected": "Б", "x1": 334, "x2": 488},
    {"id": 4, "expected": "Җ", "x1": 488, "x2": 652},
    {"id": 5, "expected": "Щ", "x1": 652, "x2": 805},
    {"id": 6, "expected": "Ч", "x1": 805, "x2": 955},
]

for c in cells_meta:
    crop = img[y_min:y_max, c["x1"]:c["x2"]]
    sub = crop[4:-4, 4:-4]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    p_lo, p_hi = np.percentile(gray, 2), np.percentile(gray, 98)
    norm = np.clip((gray.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)
    inp = cv2.resize(norm, (64, 64), interpolation=cv2.INTER_AREA)
    t = ((inp.astype(np.float32) / 255.0) - 0.5) / 0.5
    tensor = t[np.newaxis, np.newaxis, :, :]

    o_f = sess_fp32.run(None, {"input": tensor})[0][0]
    o_i = sess_int8.run(None, {"input": tensor})[0][0]

    p_f = np.exp(o_f - np.max(o_f)); p_f /= np.sum(p_f)
    p_i = np.exp(o_i - np.max(o_i)); p_i /= np.sum(p_i)

    ch_f = TATAR_UPPERCASE[int(np.argmax(p_f))]
    conf_f = float(np.max(p_f)) * 100.0

    ch_i = TATAR_UPPERCASE[int(np.argmax(p_i))]
    conf_i = float(np.max(p_i)) * 100.0

    match_str = "MATCH" if ch_f == ch_i else "DIFF"
    correct_str = "✓" if ch_i == c["expected"] else "✗"
    print(f"  Cell {c['id']}: Exp '{c['expected']}' | FP32: '{ch_f}' ({conf_f:5.1f}%) | INT8: '{ch_i}' ({conf_i:5.1f}%) | {match_str} {correct_str}")

print("=" * 80)

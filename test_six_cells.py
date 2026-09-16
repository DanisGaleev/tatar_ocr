"""
Crop 6 cells from rotated strip, save as individual squares, 
and run OCR inference with TatarOCRNet.
"""

import sys
import os
import shutil
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import cv2
import numpy as np
import torch
from PIL import Image

from tatar_ocr_dataset import TatarOCRNet, ALL_CHARS

# 1. Load and rotate image
img_path = r"C:\Users\galee\.gemini\antigravity\brain\e6d31b5c-5d27-456f-ade7-c736653ad00b\.user_uploaded\media_1789499341069.jpg"
orig = cv2.imread(img_path)
rot = cv2.rotate(orig, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("strip2_rot90_cw.png", rot)
h, w, _ = rot.shape
print(f"Rotated image shape: {w}x{h}")

# 2. Define the 6 cells
y_min, y_max = 14, 168

cells_meta = [
    {"id": 1, "char_name": "Ә", "expected": "Ә", "x1": 10,  "x2": 175},
    {"id": 2, "char_name": "А", "expected": "А", "x1": 175, "x2": 334},
    {"id": 3, "char_name": "Б", "expected": "Б", "x1": 334, "x2": 488},
    {"id": 4, "char_name": "Җ", "expected": "Җ", "x1": 488, "x2": 652},
    {"id": 5, "char_name": "Щ", "expected": "Щ", "x1": 652, "x2": 805},
    {"id": 6, "char_name": "Ч", "expected": "Ч", "x1": 805, "x2": 955},
]

# 3. Load trained model
device = torch.device("cpu")
model = TatarOCRNet(num_classes=len(ALL_CHARS)).to(device)
weights_path = Path("models/best_tatar_ocr_net.pth")
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()
print(f"Loaded model weights from: {weights_path}")

out_dir = Path("dataset/six_cells_test")
out_dir.mkdir(parents=True, exist_ok=True)
art_dir = Path(r"C:\Users\galee\.gemini\antigravity\brain\e6d31b5c-5d27-456f-ade7-c736653ad00b")

results = []

for c in cells_meta:
    # 1. Raw crop
    raw_crop = rot[y_min:y_max, c["x1"]:c["x2"]]
    ch_h, ch_w, _ = raw_crop.shape
    
    # 2. Convert to square by padding with paper background color
    bg_color = np.median(raw_crop[:10, :10], axis=(0, 1)).astype(np.uint8)
    sq_size = max(ch_h, ch_w)
    sq_crop = np.full((sq_size, sq_size, 3), bg_color, dtype=np.uint8)
    py = (sq_size - ch_h) // 2
    px = (sq_size - ch_w) // 2
    sq_crop[py:py+ch_h, px:px+ch_w] = raw_crop
    
    # Save square image with ASCII filename for Windows cv2.imwrite compatibility
    ascii_names = {1: "Ae", 2: "A", 3: "B", 4: "Zh", 5: "Shch", 6: "Ch"}
    sq_filename = f"cell_{c['id']}_{ascii_names[c['id']]}.png"
    sq_path = out_dir / sq_filename
    is_saved = cv2.imwrite(str(sq_path), sq_crop)
    if not is_saved or not sq_path.exists():
        # Fallback using PIL
        Image.fromarray(cv2.cvtColor(sq_crop, cv2.COLOR_BGR2RGB)).save(sq_path)
    if art_dir.exists():
        shutil.copy(sq_path, art_dir / sq_filename)

    # 3. Model input normalization (64x64 grayscale, centered)
    crop_gray = cv2.cvtColor(sq_crop, cv2.COLOR_BGR2GRAY)
    inp_64 = cv2.resize(crop_gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    tensor = torch.from_numpy(inp_64.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    
    # 4. Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        
    top5_probs, top5_indices = torch.topk(probs, k=5)
    top5_chars = [ALL_CHARS[i] for i in top5_indices.numpy()]
    top5_conf = top5_probs.numpy() * 100.0

    exp_char = c["expected"]
    exp_idx = ALL_CHARS.index(exp_char)
    exp_conf = float(probs[exp_idx].item()) * 100.0
    
    pred_char = top5_chars[0]
    pred_conf = top5_conf[0]
    is_correct = (pred_char == exp_char)
    in_top3 = exp_char in top5_chars[:3]

    res = {
        "id": c["id"],
        "expected": exp_char,
        "pred": pred_char,
        "pred_conf": pred_conf,
        "exp_conf": exp_conf,
        "top5": list(zip(top5_chars, [round(p, 2) for p in top5_conf])),
        "is_correct": is_correct,
        "in_top3": in_top3,
        "sq_img": str(sq_path)
    }
    results.append(res)
    
    print(f"\n==================== CELL {c['id']}: '{exp_char}' ====================")
    print(f"Expected Letter:       '{exp_char}'")
    print(f"Top-1 Prediction:      '{pred_char}' ({pred_conf:.2f}%)")
    print(f"Confidence on '{exp_char}':  {exp_conf:.2f}%")
    print("Top-5 Candidates:")
    for rk, (ch, p) in enumerate(zip(top5_chars, top5_conf), 1):
        match_star = "  ★ EXPECTED" if ch == exp_char else ""
        print(f"   {rk}. '{ch}' (U+{ord(ch):04X}) : {p:5.2f}%{match_star}")

# 5. Build Montage
tile_size = 150
montage = np.full((tile_size + 40, 6 * tile_size, 3), 250, dtype=np.uint8)

for i, res in enumerate(results):
    sq = cv2.imread(res["sq_img"])
    sq_resized = cv2.resize(sq, (tile_size, tile_size))
    
    # Border color: Green if Top-1 match, Orange if Top-3 match, Red otherwise
    if res["is_correct"]:
        border_col = (0, 180, 0)
    elif res["in_top3"]:
        border_col = (0, 140, 255)
    else:
        border_col = (0, 0, 220)
        
    cv2.rectangle(sq_resized, (0, 0), (tile_size - 1, tile_size - 1), border_col, 3)
    montage[0:tile_size, i * tile_size:(i + 1) * tile_size] = sq_resized
    
    # Bottom label
    label_top = f"Exp: {res['expected']}"
    label_bot = f"Pred: {res['pred']} ({res['pred_conf']:.0f}%)"
    cv2.putText(montage, label_top, (i * tile_size + 8, tile_size + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
    cv2.putText(montage, label_bot, (i * tile_size + 8, tile_size + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, border_col, 1)

montage_path = out_dir / "six_cells_montage.png"
cv2.imwrite(str(montage_path), montage)
if art_dir.exists():
    shutil.copy(montage_path, art_dir / "six_cells_montage.png")
    shutil.copy("strip2_rot90_cw.png", art_dir / "strip2_rot90_cw.png")

print(f"\nSaved montage to {montage_path} and copied to artifact directory.")

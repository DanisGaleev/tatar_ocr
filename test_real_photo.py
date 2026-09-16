"""
Crop and Test Real Handwritten Cells on Trained Tatar OCR Model
"""

import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import cv2
import numpy as np
import torch
from PIL import Image

from tatar_ocr_dataset import TatarOCRNet, ALL_CHARS

# 1. Load full rotated image
rot_img_path = Path("full_rotated_cw.png")
if not rot_img_path.exists():
    img_path = r"C:\Users\galee\.gemini\antigravity\brain\e6d31b5c-5d27-456f-ade7-c736653ad00b\.user_uploaded\media_1789498366286.jpg"
    full = cv2.imread(img_path)
    rot = cv2.rotate(full, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("full_rotated_cw.png", rot)
else:
    rot = cv2.imread(str(rot_img_path))

h, w, _ = rot.shape
print(f"Loaded image: {w}x{h}")

# 2. Cell coordinates (x_min, y_min, x_max, y_max)
# Visual inspection of the 5 cells:
# Top line: ~14, Bottom line: ~180
y_min, y_max = 14, 180

# The 5 cell boundary intervals along X:
cell_boxes = [
    {"name": "Cell_1", "expected": "А", "x_min": 10,  "x_max": 188},
    {"name": "Cell_2", "expected": "Ә", "x_min": 188, "x_max": 380},
    {"name": "Cell_3", "expected": "Ш", "x_min": 380, "x_max": 580},
    {"name": "Cell_4", "expected": "Щ", "x_min": 580, "x_max": 772},
    {"name": "Cell_5", "expected": "Д", "x_min": 772, "x_max": 1008},
]

# 3. Load trained model
device = torch.device("cpu")
model = TatarOCRNet(num_classes=len(ALL_CHARS)).to(device)
weights_path = Path("models/best_tatar_ocr_net.pth")
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()
print(f"Loaded weights from {weights_path}")

out_dir = Path("dataset/real_crop_test")
out_dir.mkdir(parents=True, exist_ok=True)

results = []

for idx, box in enumerate(cell_boxes):
    # Crop cell
    crop_bgr = rot[y_min:y_max, box["x_min"]:box["x_max"]]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Save raw cell crop
    raw_crop_path = out_dir / f"{box['name']}_raw.png"
    cv2.imwrite(str(raw_crop_path), crop_bgr)
    
    # Make square by center cropping or padding
    ch_h, ch_w = crop_gray.shape
    size = max(ch_h, ch_w)
    square = np.full((size, size), 255, dtype=np.uint8)
    pad_y = (size - ch_h) // 2
    pad_x = (size - ch_w) // 2
    square[pad_y:pad_y + ch_h, pad_x:pad_x + ch_w] = crop_gray
    
    # Resize to model input size (64x64)
    cell_64 = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
    norm_crop_path = out_dir / f"{box['name']}_64x64.png"
    cv2.imwrite(str(norm_crop_path), cell_64)

    # Normalize to [-1, 1] tensor
    tensor = torch.from_numpy(cell_64.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    
    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        
    top5_probs, top5_indices = torch.topk(probs, k=5)
    top5_chars = [ALL_CHARS[i] for i in top5_indices.numpy()]
    top5_conf = top5_probs.numpy() * 100.0

    pred_char = top5_chars[0]
    pred_conf = top5_conf[0]
    is_match = (pred_char == box["expected"])
    
    res = {
        "cell": box["name"],
        "expected": box["expected"],
        "pred": pred_char,
        "conf": pred_conf,
        "top5": list(zip(top5_chars, [round(c, 2) for c in top5_conf])),
        "is_correct": is_match,
        "raw_img": str(raw_crop_path),
        "norm_img": str(norm_crop_path)
    }
    results.append(res)
    
    print(f"\n--- {box['name']} ---")
    print(f"Expected (Vision Ground Truth): '{box['expected']}'")
    print(f"Model Prediction:             '{pred_char}' (Confidence: {pred_conf:.2f}%)")
    print("Top-5 Candidates:")
    for rank, (c, p) in enumerate(zip(top5_chars, top5_conf), 1):
        print(f"  {rank}. '{c}' (U+{ord(c):04X}) -> {p:5.2f}%")

# Generate visual montage of results
montage = np.full((140, 5 * 140, 3), 255, dtype=np.uint8)
for i, res in enumerate(results):
    cell_img = cv2.imread(res["raw_img"])
    cell_resized = cv2.resize(cell_img, (140, 140))
    # Border: Green if correct, Orange if top-3, Red if mismatch
    top3_chars = [c for c, _ in res["top5"][:3]]
    if res["is_correct"]:
        border_color = (0, 180, 0)
    elif res["expected"] in top3_chars:
        border_color = (0, 140, 255)
    else:
        border_color = (0, 0, 220)
        
    cv2.rectangle(cell_resized, (0, 0), (139, 139), border_color, 4)
    # Add label
    label = f"Exp:{res['expected']} Pred:{res['pred']}"
    cv2.putText(cell_resized, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2)
    montage[:, i * 140:(i + 1) * 140] = cell_resized

montage_path = Path("dataset/real_crop_test/montage_results.png")
cv2.imwrite(str(montage_path), montage)
print(f"\nSaved visual comparison montage: {montage_path}")

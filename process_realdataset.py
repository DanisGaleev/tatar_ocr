"""
Rectify and slice handwritten collection sheet photos into labeled dataset.
Grid specification:
- 24 rows per sheet, 17 cells per row (10x10 mm each)
- Left prompt box (12 mm + 3 mm gap)
- Row height: 10 mm, Row gap: 1.8 mm
- Total cells across 5 pages: 5 * 24 * 17 = 2,040 samples
"""

import os
import sys
import json
import csv
from pathlib import Path
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# 1. Letter Schedule matching generate_collection_pdf.py
LETTER_SCHEDULE = [
    ("Ә", 5), ("Җ", 5), ("Ң", 5), ("Ө", 5), ("Ү", 5), ("Һ", 5),
    ("Щ", 4), ("Ц", 4), ("Ч", 4), ("Ш", 4), ("Д", 4), ("Ю", 4),
    ("Б", 4), ("Э", 4),
    ("А", 2), ("В", 2), ("Г", 2), ("Е", 2), ("Ё", 2), ("Ж", 2),
    ("З", 2), ("И", 2), ("Й", 2), ("К", 2), ("Л", 2), ("М", 2),
    ("Н", 2), ("О", 2), ("П", 2), ("Р", 2), ("С", 2), ("Т", 2),
    ("У", 2), ("Ф", 2), ("Х", 2), ("Ъ", 2), ("Ы", 2), ("Ь", 2),
    ("Я", 2),
    ("Ә", 2), ("Җ", 2), ("Ң", 2), ("Ө", 2), ("Щ", 2),
]

all_rows = []
for char, count in LETTER_SCHEDULE:
    for _ in range(count):
        all_rows.append(char)

ROWS_PER_PAGE = 24
COLS_PER_ROW = 17

PAGE_LETTERS = {}
for p in range(5):
    PAGE_LETTERS[p + 1] = all_rows[p * ROWS_PER_PAGE : (p + 1) * ROWS_PER_PAGE]

# 2. Source Images
PAGE_IMAGES = {
    1: r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689337.jpg",
    2: r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689351.jpg",
    3: r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689319.jpg",
    4: r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689304.jpg",
    5: r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689366.jpg",
}

# 3. Exact Corner Coordinates for Grid Table [TL, TR, BR, BL]
GRID_CORNERS = {
    1: {'TL': [44, 16], 'TR': [693, 15], 'BR': [707, 972], 'BL': [42, 975]},
    2: {'TL': [52, 15], 'TR': [700, 35], 'BR': [723, 986], 'BL': [59, 989]},
    3: {'TL': [59, 16], 'TR': [694, 27], 'BR': [717, 978], 'BL': [54, 992]},
    4: {'TL': [36, 28], 'TR': [692, 15], 'BR': [717, 994], 'BL': [59, 974]},
    5: {'TL': [38, 15], 'TR': [704, 28], 'BR': [709, 985], 'BL': [49, 978]},
}

# Canvas geometry: 10 px / mm
CANVAS_W = 1850
CANVAS_H = 2814
BOX_SIZE = 100
ROW_HEIGHT = 100
ROW_GAP = 18
BOXES_X_START = 150  # prompt (120) + gap (30)


def process_dataset(out_base="realdataset"):
    out_dir = Path(out_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Folders
    by_class_dir = out_dir / "by_class"
    raw_dir = out_dir / "cells_100x100"
    norm_dir = out_dir / "cells_64x64"
    prompts_dir = out_dir / "prompts"
    
    by_class_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    dst_canvas = np.array([
        [0, 0],
        [CANVAS_W, 0],
        [CANVAS_W, CANVAS_H],
        [0, CANVAS_H]
    ], dtype=np.float32)
    
    records = []
    char_counts = {}
    
    for page_num in range(1, 6):
        img_path = PAGE_IMAGES[page_num]
        print(f"Processing Page {page_num}/5 from {Path(img_path).name}...")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image {img_path}")
            
        c = GRID_CORNERS[page_num]
        src_quad = np.array([c['TL'], c['TR'], c['BR'], c['BL']], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_quad, dst_canvas)
        warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H))
        
        # Save rectified page preview
        cv2.imwrite(str(out_dir / f"rectified_page_{page_num}.jpg"), cv2.resize(warped, (925, 1407)))
        
        page_rows = PAGE_LETTERS[page_num]
        
        for r_idx in range(ROWS_PER_PAGE):
            char = page_rows[r_idx]
            y1 = int(r_idx * (ROW_HEIGHT + ROW_GAP))
            y2 = y1 + ROW_HEIGHT
            
            # Save prompt box
            prompt_crop = warped[y1:y2, 0:120]
            cv2.imwrite(str(prompts_dir / f"prompt_p{page_num}_r{r_idx:02d}_{char}.png"), prompt_crop)
            
            char_class_dir = by_class_dir / char
            char_class_dir.mkdir(exist_ok=True)
            
            for c_idx in range(COLS_PER_ROW):
                x1 = BOXES_X_START + c_idx * BOX_SIZE
                x2 = x1 + BOX_SIZE
                
                # 100x100 raw cell
                cell_bgr = warped[y1:y2, x1:x2]
                
                cell_id = f"p{page_num}_r{r_idx:02d}_c{c_idx:02d}_{char}"
                
                # 1. Save raw BGR 100x100
                raw_filename = f"{cell_id}.png"
                cv2.imwrite(str(raw_dir / raw_filename), cell_bgr)
                
                # 2. Save in by_class folder
                cv2.imwrite(str(char_class_dir / raw_filename), cell_bgr)
                
                # 3. Save standardized 64x64 grayscale (matching OCR model input)
                cell_gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
                cell_64 = cv2.resize(cell_gray, (64, 64), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(norm_dir / raw_filename), cell_64)
                
                # Record metadata
                records.append({
                    "cell_id": cell_id,
                    "char": char,
                    "page": page_num,
                    "row": r_idx,
                    "col": c_idx,
                    "raw_path": f"cells_100x100/{raw_filename}",
                    "norm_64_path": f"cells_64x64/{raw_filename}",
                    "class_path": f"by_class/{char}/{raw_filename}"
                })
                
                char_counts[char] = char_counts.get(char, 0) + 1
                
    # Save metadata.csv
    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell_id", "char", "page", "row", "col", "raw_path", "norm_64_path", "class_path"])
        writer.writeheader()
        writer.writerows(records)
        
    # Save class_summary.json
    summary_path = out_dir / "class_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_cells": len(records),
            "total_pages": 5,
            "rows_per_page": 24,
            "cols_per_row": 17,
            "classes_count": len(char_counts),
            "char_counts": char_counts
        }, f, ensure_ascii=False, indent=2)
        
    print("\n=======================================================")
    print(f"DONE! Processed {len(records)} total handwritten cells.")
    print(f"Unique classes: {len(char_counts)}")
    print(f"Dataset saved to directory: {out_dir.resolve().as_posix()}")
    print("=======================================================")


if __name__ == "__main__":
    process_dataset("realdataset")

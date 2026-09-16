"""
Batch Slicing & Exporter using calibrated grid_corners.json
Produces realdataset with:
- by_class/<char>/ (all 2040 cells organized into folders)
- cells_raw_100x100/
- cells_64x64/
- metadata.csv
- class_summary.json
"""

import sys
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# 1. Source Images
PAGE_IMAGES = [
    (1, r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689337.jpg"),
    (2, r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689351.jpg"),
    (3, r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689319.jpg"),
    (4, r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689304.jpg"),
    (5, r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689366.jpg"),
]

# 2. Letter Schedule matching generate_collection_pdf.py
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

# Target Rectified Canvas Dimensions
CANVAS_W = 1700
CANVAS_H = 2814
CELL_W = 100
CELL_H = 100
ROW_UNIT = 118


def save_image_unicode(target_path, img_mat):
    """Safely saves OpenCV image to Windows Unicode paths without character corruption."""
    ext = Path(target_path).suffix.lower()
    if not ext:
        ext = ".png"
    is_ok, buf = cv2.imencode(ext, img_mat)
    if is_ok:
        with open(target_path, "wb") as f:
            f.write(buf.tobytes())
    else:
        raise IOError(f"Could not encode image for {target_path}")


def export_dataset(corners_file="grid_corners.json", out_base="realdataset", inner_margin=2):
    corners_path = Path(corners_file)
    if not corners_path.exists():
        print(f"Error: {corners_file} not found. Please run select_corners_and_preview.py first.")
        sys.exit(1)
        
    with open(corners_path, "r", encoding="utf-8") as f:
        corners_data = json.load(f)
        
    out_dir = Path(out_base)
    # Clean previous contents to avoid mixing
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    by_class_dir = out_dir / "by_class"
    raw_dir = out_dir / "cells_raw_100x100"
    norm_dir = out_dir / "cells_64x64"
    
    by_class_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    
    dst_canvas = np.array([
        [0, 0],
        [CANVAS_W, 0],
        [CANVAS_W, CANVAS_H],
        [0, CANVAS_H]
    ], dtype=np.float32)
    
    records = []
    char_counts = {}
    
    for page_num, img_path in PAGE_IMAGES:
        key = str(page_num)
        if key not in corners_data:
            print(f"Warning: Page {page_num} missing in {corners_file}!")
            continue
            
        c = corners_data[key]
        src_quad = np.array([c["TL"], c["TR"], c["BR"], c["BL"]], dtype=np.float32)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue
            
        M = cv2.getPerspectiveTransform(src_quad, dst_canvas)
        warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H))
        
        page_rows = PAGE_LETTERS[page_num]
        
        for r_idx in range(ROWS_PER_PAGE):
            char = page_rows[r_idx]
            y1 = int(r_idx * ROW_UNIT)
            y2 = y1 + CELL_H
            
            char_folder = by_class_dir / char
            char_folder.mkdir(exist_ok=True)
            
            for c_idx in range(COLS_PER_ROW):
                x1 = c_idx * CELL_W
                x2 = x1 + CELL_W
                
                # Apply tiny inner margin (e.g. 2px) to exclude printed border lines
                crop_y1 = min(y1 + inner_margin, y2 - 1)
                crop_y2 = max(y2 - inner_margin, crop_y1 + 1)
                crop_x1 = min(x1 + inner_margin, x2 - 1)
                crop_x2 = max(x2 - inner_margin, crop_x1 + 1)
                
                cell_bgr = warped[crop_y1:crop_y2, crop_x1:crop_x2]
                cell_100 = cv2.resize(cell_bgr, (100, 100), interpolation=cv2.INTER_AREA)
                
                cell_id = f"p{page_num}_r{r_idx+1:02d}_c{c_idx+1:02d}_{char}"
                filename = f"{cell_id}.png"
                
                # 1. Raw 100x100
                save_image_unicode(raw_dir / filename, cell_100)
                # 2. By class
                save_image_unicode(char_folder / filename, cell_100)
                # 3. Normalized 64x64 grayscale
                cell_gray = cv2.cvtColor(cell_100, cv2.COLOR_BGR2GRAY)
                cell_64 = cv2.resize(cell_gray, (64, 64), interpolation=cv2.INTER_AREA)
                save_image_unicode(norm_dir / filename, cell_64)
                
                records.append({
                    "cell_id": cell_id,
                    "char": char,
                    "page": page_num,
                    "row": r_idx + 1,
                    "col": c_idx + 1,
                    "raw_path": f"cells_raw_100x100/{filename}",
                    "norm_64_path": f"cells_64x64/{filename}",
                    "class_path": f"by_class/{char}/{filename}"
                })
                char_counts[char] = char_counts.get(char, 0) + 1
                
        print(f"Exported Page {page_num}/5 ({ROWS_PER_PAGE * COLS_PER_ROW} cells)")
        
    # Write metadata.csv
    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell_id", "char", "page", "row", "col", "raw_path", "norm_64_path", "class_path"])
        writer.writeheader()
        writer.writerows(records)
        
    # Write class_summary.json
    summary_path = out_dir / "class_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_cells": len(records),
            "total_pages": len(PAGE_IMAGES),
            "rows_per_page": ROWS_PER_PAGE,
            "cols_per_row": COLS_PER_ROW,
            "classes_count": len(char_counts),
            "char_counts": char_counts
        }, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully exported {len(records)} cells to {out_dir.resolve().as_posix()}!")


if __name__ == "__main__":
    export_dataset()

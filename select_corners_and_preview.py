"""
Interactive 4-Corner Calibration & Grid Slicing Preview Tool
Tatar Handwriting Collection Sheets (17 cols x 24 rows)

Instructions:
1. For each sheet, click the 4 OUTER CORNERS of the 17-CELL HANDWRITTEN GRID:
   - 1st click: Top-Left (TL) of Row 1, Cell 1
   - 2nd click: Top-Right (TR) of Row 1, Cell 17
   - 3rd click: Bottom-Right (BR) of Row 24, Cell 17
   - 4th click: Bottom-Left (BL) of Row 24, Cell 1
2. Press [Enter] or [Space] to see the Rectified Slicing Preview.
3. In Preview:
   - Press [Enter] or [Space] to ACCEPT and proceed to the next sheet.
   - Press [R] to REJECT and re-click points for this sheet.
4. When all 5 sheets are accepted, grid_corners.json and previews/ will be saved.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np

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

# Target Rectified Canvas Dimensions (170 mm width x 281.4 mm height -> 1700 x 2814 px)
# 10 px per mm:
CANVAS_W = 1700
CANVAS_H = 2814
CELL_W = 100       # 10 mm
CELL_H = 100       # 10 mm
ROW_GAP = 18       # 1.8 mm between rows
ROW_UNIT = 118     # 10 mm cell + 1.8 mm gap


def build_overlay_preview(warped, page_num):
    vis = warped.copy()
    rows = PAGE_LETTERS[page_num]
    
    for r in range(ROWS_PER_PAGE):
        y1 = int(r * ROW_UNIT)
        y2 = y1 + CELL_H
        
        # Highlight the 1.8 mm gap after row (except last row) in faint red
        if r < ROWS_PER_PAGE - 1:
            cv2.rectangle(vis, (0, y2), (CANVAS_W, y2 + ROW_GAP), (0, 0, 180), -1)
            
        # Draw cells
        for c in range(COLS_PER_ROW):
            x1 = c * CELL_W
            x2 = x1 + CELL_W
            # Draw green cell border
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw label on the left edge
        cv2.putText(vis, f"R{r+1:02d} {rows[r]}", (10, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 3)
        
    return vis


def calibrate_sheets():
    previews_dir = Path("previews")
    previews_dir.mkdir(exist_ok=True)
    
    corners_data = {}
    
    # Load existing if available
    json_path = Path("grid_corners.json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                corners_data = json.load(f)
        except Exception:
            pass
            
    print("=" * 65)
    print("  TATAR OCR GRID CALIBRATION & SLICING TOOL")
    print("=" * 65)
    print("Instructions:")
    print("For each sheet, click the 4 OUTER CORNERS of the 17-CELL GRID:")
    print("  1. TOP-LEFT     (Row 1, Cell 1)")
    print("  2. TOP-RIGHT    (Row 1, Cell 17)")
    print("  3. BOTTOM-RIGHT (Row 24, Cell 17)")
    print("  4. BOTTOM-LEFT  (Row 24, Cell 1)")
    print("\nKeys in selection window:")
    print("  [Enter] / [Space] : Confirm 4 points and view Slicing Preview")
    print("  [R]               : Reset points for this sheet")
    print("  [Q] / [Esc]       : Quit")
    print("=" * 65)
    
    page_idx = 0
    while page_idx < len(PAGE_IMAGES):
        page_num, img_path = PAGE_IMAGES[page_idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: could not read {img_path}")
            sys.exit(1)
            
        h, w = img.shape[:2]
        pts = []
        
        # Check if already had points
        if str(page_num) in corners_data:
            existing = corners_data[str(page_num)]
            pts = [existing["TL"], existing["TR"], existing["BR"], existing["BL"]]
            
        window_name = f"Sheet {page_num}/5: Click 4 Corners of 17-cell Grid [TL -> TR -> BR -> BL]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 900, 1200)
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal pts
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(pts) < 4:
                    pts.append([x, y])
                    print(f"  Page {page_num} Point {len(pts)}/4: ({x}, {y})")
                    
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Point selection loop
        confirmed_points = False
        while True:
            display = img.copy()
            labels = ["1. TL (Top-Left)", "2. TR (Top-Right)", "3. BR (Bottom-Right)", "4. BL (Bottom-Left)"]
            colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]
            
            for i, p in enumerate(pts):
                cv2.circle(display, tuple(p), 6, colors[i], -1)
                cv2.circle(display, tuple(p), 8, (255, 255, 255), 2)
                cv2.putText(display, labels[i], (p[0] + 10, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
                
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.line(display, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 255), 2)
            if len(pts) == 4:
                cv2.line(display, tuple(pts[3]), tuple(pts[0]), (0, 255, 255), 2)
                cv2.putText(display, "Press ENTER/SPACE to view Slicing Preview, or 'R' to reset", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"Click {labels[len(pts)]} of 17-box grid", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key in [13, 32]:  # Enter or Space
                if len(pts) == 4:
                    confirmed_points = True
                    break
                else:
                    print("  Please click all 4 corners first!")
            elif key in [ord('r'), ord('R')]:
                pts = []
                print("  Points reset. Please click 4 corners again.")
            elif key in [ord('q'), ord('Q'), 27]:
                cv2.destroyAllWindows()
                print("Exiting tool.")
                return False
                
        cv2.destroyWindow(window_name)
        
        if confirmed_points:
            # Generate perspective warp & overlay preview
            src_quad = np.array(pts, dtype=np.float32)
            dst_quad = np.array([
                [0, 0],
                [CANVAS_W, 0],
                [CANVAS_W, CANVAS_H],
                [0, CANVAS_H]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_quad, dst_quad)
            warped = cv2.warpPerspective(img, M, (CANVAS_W, CANVAS_H))
            preview_img = build_overlay_preview(warped, page_num)
            
            preview_win = f"PREVIEW Slicing: Page {page_num}/5 [ENTER=Accept, R=Retry]"
            cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_win, 900, 1200)
            
            accepted = False
            while True:
                cv2.imshow(preview_win, preview_img)
                key = cv2.waitKey(20) & 0xFF
                if key in [13, 32]:  # Enter or Space
                    accepted = True
                    break
                elif key in [ord('r'), ord('R')]:
                    accepted = False
                    break
                elif key in [ord('q'), ord('Q'), 27]:
                    cv2.destroyAllWindows()
                    print("Exiting tool.")
                    return False
                    
            cv2.destroyWindow(preview_win)
            
            if accepted:
                print(f"  --> Page {page_num} ACCEPTED!")
                corners_data[str(page_num)] = {
                    "TL": pts[0],
                    "TR": pts[1],
                    "BR": pts[2],
                    "BL": pts[3]
                }
                # Save preview to disk
                cv2.imwrite(str(previews_dir / f"preview_page_{page_num}_grid.jpg"), cv2.resize(preview_img, (850, 1407)))
                cv2.imwrite(str(previews_dir / f"rectified_page_{page_num}.jpg"), cv2.resize(warped, (850, 1407)))
                page_idx += 1
            else:
                print(f"  --> Page {page_num} REJECTED. Re-selecting corners...")
                pts = []
                
    # Save corners to JSON
    with open("grid_corners.json", "w", encoding="utf-8") as f:
        json.dump(corners_data, f, indent=2)
        
    print("\n" + "=" * 65)
    print("ALL 5 PAGES CALIBRATED & VERIFIED!")
    print("Corners saved to: grid_corners.json")
    print("High-res preview overlays saved to: previews/preview_page_X_grid.jpg")
    print("=" * 65)
    return True


if __name__ == "__main__":
    calibrate_sheets()

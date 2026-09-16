"""
Find grid corners for all 5 sheets.
We will inspect the detected grid / row lines and find the precise 4 corners of the 24-row grid.
"""
import cv2
import numpy as np
from pathlib import Path

image_paths = [
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689337.jpg", # Page 1
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689351.jpg", # Page 2
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689319.jpg", # Page 3
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689304.jpg", # Page 4
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689366.jpg", # Page 5
]

out_debug = Path("debug_rectify")
out_debug.mkdir(exist_ok=True)

for p_idx, img_path in enumerate(image_paths, 1):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Let's find horizontal line segments and vertical line segments
    # Adaptive threshold
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # Morphological extraction of horizontal lines
    h_len = w // 15
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_morph = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    
    # Morphological extraction of vertical lines
    v_len = h // 40
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_morph = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    
    # Combine horizontal and vertical lines
    grid_morph = cv2.bitwise_or(h_morph, v_morph)
    
    # Find contours of grid components
    contours, _ = cv2.findContours(grid_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that look like row strips or prompt boxes
    valid_boxes = []
    for c in contours:
        x, y, bw_box, bh_box = cv2.boundingRect(c)
        # Row strip is wide: e.g. bw_box > 400, bh_box between 20 and 50
        # Or prompt box: bw_box between 25 and 60, bh_box between 20 and 50
        if bw_box > 300 and 15 <= bh_box <= 60:
            valid_boxes.append(('row_strip', x, y, bw_box, bh_box))
        elif 20 <= bw_box <= 80 and 15 <= bh_box <= 60:
            valid_boxes.append(('prompt_box', x, y, bw_box, bh_box))
            
    print(f"Page {p_idx}: Found {len(valid_boxes)} candidate boxes (strips/prompts)")
    
    # Draw candidates on image
    vis = img.copy()
    for b_type, x, y, bw_box, bh_box in valid_boxes:
        color = (0, 255, 0) if b_type == 'row_strip' else (255, 0, 0)
        cv2.rectangle(vis, (x, y), (x + bw_box, y + bh_box), color, 2)
        
    cv2.imwrite(str(out_debug / f"page_{p_idx}_candidates.png"), vis)

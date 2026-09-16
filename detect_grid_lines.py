"""
Detect grid boundaries from lines directly.
"""
import cv2
import numpy as np

for p_idx, img_path in enumerate([
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689337.jpg", # 1
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689351.jpg", # 2
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689319.jpg", # 3
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689304.jpg", # 4
    r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689366.jpg", # 5
], 1):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to isolate lines
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    
    # Find long horizontal line segments
    # The cell strips are 170mm wide (~560 pixels), prompt boxes are ~40 pixels
    lines = cv2.HoughLinesP(bw, 1, np.pi / 180, threshold=150, minLineLength=200, maxLineGap=20)
    
    h_lines = []
    if lines is not None:
        for line in lines:
            pts = line.reshape(-1)
            x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 5 or angle > 175: # nearly horizontal
                h_lines.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                
    # Sort by y coordinate
    h_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
    print(f"Page {p_idx}: found {len(h_lines)} long horizontal lines")
    if h_lines:
        top_y = (h_lines[0][1] + h_lines[0][3]) / 2
        bottom_y = (h_lines[-1][1] + h_lines[-1][3]) / 2
        min_x = min(l[0] for l in h_lines)
        max_x = max(l[2] for l in h_lines)
        print(f"   y range: [{top_y:.1f}, {bottom_y:.1f}], span={bottom_y - top_y:.1f}")
        print(f"   x range: [{min_x}, {max_x}], span={max_x - min_x}")

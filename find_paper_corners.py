"""
Accurately find the 4 corners of the paper sheet for each image.
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

def order_points(pts):
    # pts: 4 points (x, y)
    # top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect

for p_idx, img_path in enumerate(image_paths, 1):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to preserve edges while smoothing background texture
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Otsu threshold
    _, thresh = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological close to merge paper into solid sheet
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    
    # Approximate polygon
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    print(f"Page {p_idx}: approx vertices = {len(approx)}")
    for pt in approx:
        print(f"   pt: {pt[0].tolist()}")
        
    # Also find minimum area bounding box
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = order_points(box)
    print(f"Page {p_idx}: MinAreaRect corners:\n{box.astype(int)}")

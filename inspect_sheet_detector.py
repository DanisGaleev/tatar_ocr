"""
Test sheet detection and perspective transform for each of the 5 sheets.
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

for idx, p in enumerate(image_paths, 1):
    img = cv2.imread(p)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Paper boundary detection
    # Smooth to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # The paper is light, background is darker
    # Let's find paper mask using Otsu or adaptive thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological close to fill holes inside paper (drawings, text)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(f"Page {idx}: image {w}x{h}, max contour area = {area:.0f} (ratio {area/(w*h):.2f}), vertices = {len(approx)}")
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        print(f"  MinAreaRect center={rect[0]}, size={rect[1]}, angle={rect[2]:.2f}")
    else:
        print(f"Page {idx}: No contours found")

"""
Inspect details of row strips and prompt boxes.
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

for p_idx, img_path in enumerate(image_paths, 1):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    h_len = w // 15
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_morph = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    
    # Let's check horizontal projections to find the 24 rows
    # A horizontal projection of h_morph in the grid area:
    proj_y = np.sum(h_morph[:, int(w*0.2):int(w*0.9)], axis=1) / 255
    # Peaks in proj_y correspond to horizontal grid lines
    # Each row has top and bottom line!
    # Let's find peaks
    import scipy.signal
    peaks, props = scipy.signal.find_peaks(proj_y, height=100, distance=10)
    print(f"Page {p_idx}: found {len(peaks)} horizontal line peaks in range y=[{peaks[0] if len(peaks)>0 else None}, {peaks[-1] if len(peaks)>0 else None}]")

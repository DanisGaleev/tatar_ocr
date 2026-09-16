"""
Inspect candidate boxes and analyze row positions.
"""
import cv2
import numpy as np

img_path = r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689337.jpg" # Page 1
img = cv2.imread(img_path)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adaptive threshold
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

# Morphological open to find horizontal lines of at least 150px
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
h_morph = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

# Morphological open to find vertical lines of at least 20px
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
v_morph = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

# Sum h_morph horizontally between x=200 and x=600 to find row y ranges
proj_h = np.sum(h_morph[:, 200:600], axis=1) / 255

import scipy.signal
peaks, _ = scipy.signal.find_peaks(proj_h, height=150, distance=8)
print(f"Detected {len(peaks)} peaks in horizontal projection:")
print(peaks.tolist())

# Check pairwise distances between peaks
diffs = np.diff(peaks)
print(f"Diffs between consecutive peaks: {diffs.tolist()}")

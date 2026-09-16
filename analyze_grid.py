"""
Analyze the grid structure on the first page:
find horizontal lines, vertical lines, row strips, and cells.
"""
import cv2
import numpy as np

img_path = r"C:/Users/galee/.gemini/antigravity/brain/4ab736f3-96bf-481a-bcfe-485478017736/.user_uploaded/media_1789575689337.jpg" # Page 1
img = cv2.imread(img_path)
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Let's check the paper bounds
# Threshold to find white paper
_, paper_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)
x, y, bw, bh = cv2.boundingRect(c)
print(f"Paper bounding box: x={x}, y={y}, w={bw}, h={bh}")

# Let's inspect horizontal line profile in the grid region
# Binarize with adaptive threshold to isolate grid lines & text
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

# Horizontal lines kernel
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

# Vertical lines kernel
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

print(f"H-lines non-zero: {cv2.countNonZero(h_lines)}")
print(f"V-lines non-zero: {cv2.countNonZero(v_lines)}")

cv2.imwrite("debug_h_lines.png", h_lines)
cv2.imwrite("debug_v_lines.png", v_lines)

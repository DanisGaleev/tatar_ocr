import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
crop_dir = Path("dataset/user_crop_entire_cells")
for p in sorted(crop_dir.glob("cell_*.png")):
    im = Image.open(p).convert("L")
    arr = np.array(im)
    print(f"{p.name}: size={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    # Inspect edge borders (top, bottom, left, right 3 pixels)
    top_edge = arr[:3, :].mean()
    bottom_edge = arr[-3:, :].mean()
    left_edge = arr[:, :3].mean()
    right_edge = arr[:, -3:].mean()
    print(f"   edges mean: top={top_edge:.1f}, btm={bottom_edge:.1f}, left={left_edge:.1f}, right={right_edge:.1f}")

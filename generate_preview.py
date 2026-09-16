import os
import sys
import random
from pathlib import Path
from PIL import Image
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
from tatar_ocr_augmentation import (
    generate_augmented_cell_image,
    get_available_fonts,
    TATAR_UPPERCASE,
    TATAR_LOWERCASE,
    DIGIT_CHARS,
    PUNCTUATION_CHARS
)

print("Loading fonts...")
all_fonts, font_weights = get_available_fonts()
font_keys = list(all_fonts.keys())
weights_list = [font_weights[k] for k in font_keys]

print(f"Loaded {len(all_fonts)} fonts. Generating preview grid...")

# We will generate a 6x6 grid of 64x64 samples demonstrating different characters & effects
demo_chars = [
    # Specific Tatar uppercase
    "Ә", "Ө", "Ү", "Җ", "Ң", "Һ",
    # Specific Tatar lowercase
    "ә", "ө", "ү", "җ", "ң", "һ",
    # Standard Cyrillic block letters
    "А", "Б", "В", "Г", "Д", "Е",
    "К", "М", "Н", "О", "П", "Р",
    # Digits and punctuation
    "1", "2", "3", "8", "9", "0",
    "!", "?", ",", ".", "-", "«"
]

rows, cols = 6, 6
cell_size = 64
grid_img = np.full((rows * cell_size, cols * cell_size), 255, dtype=np.uint8)

for idx, ch in enumerate(demo_chars):
    r = idx // cols
    c = idx % cols
    
    # Pick font weighted towards handwritten print
    chosen_font_name = random.choices(font_keys, weights=weights_list, k=1)[0]
    font_path = all_fonts[chosen_font_name]
    
    # Generate realistic augmented cell image
    arr = generate_augmented_cell_image(
        char=ch,
        font_path=font_path,
        canvas_size=cell_size,
        elastic_strength=random.uniform(12.0, 22.0),
        pressure_variation=random.uniform(1.2, 2.2),
        enable_overwriting=True,
        enable_flourishes=True,
        enable_cell_borders=True,
        enable_paper_texture=True,
        enable_camera_blur=True
    )
    
    grid_img[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = arr

out_path = Path("dataset/augmented_preview_grid_6x6.png")
Image.fromarray(grid_img).save(out_path)
print(f"Preview saved successfully to {out_path.as_posix()}!")

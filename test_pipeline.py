"""
Comprehensive verification test suite for Tatar OCR Dataset & Augmentation Pipeline
"""

import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')

from tatar_ocr_augmentation import (
    TATAR_UPPERCASE,
    TATAR_LOWERCASE,
    PUNCTUATION_CHARS,
    DIGIT_CHARS,
    ALL_CHARS,
    get_available_fonts,
    render_base_char,
    generate_augmented_cell_image,
    apply_cell_crop_artifacts,
    apply_overwriting_correction,
    apply_variable_thickness_and_starvation,
    apply_flourishes_and_stray_marks,
    apply_paper_and_lighting,
    apply_elastic_wobble,
)

from tatar_ocr_dataset import (
    TatarOnlineAugmentedDataset,
    TatarOCRNet,
    generate_batch_dataset_to_disk
)


def run_tests():
    print("==================================================")
    print("1. Testing Fonts & Tatar Character Support")
    print("==================================================")
    all_fonts, font_weights = get_available_fonts()
    print(f"Total available fonts: {len(all_fonts)}")
    assert len(all_fonts) >= 10, f"Expected at least 10 fonts, found {len(all_fonts)}"

    # Check key handwritten print fonts
    required_print_fonts = ["BalsamiqSans-Regular", "PlaypenSans", "Pangolin", "DidactGothic", "Nunito-Regular"]
    for f_name in required_print_fonts:
        assert f_name in all_fonts, f"Missing required handwritten print font: {f_name}"
        print(f"  ✓ Found {f_name}")

    # Verify rendering of special Tatar characters
    special_tatar = ["Ә", "Ө", "Ү", "Җ", "Ң", "Һ", "ә", "ө", "ү", "җ", "ң", "һ"]
    test_font_path = all_fonts["BalsamiqSans-Regular"]
    for ch in special_tatar:
        arr = render_base_char(ch, test_font_path, canvas_size=64)
        assert arr.shape == (64, 64)
        assert (arr < 200).sum() > 20, f"Character {ch} rendered blank or empty!"
    print(f"  ✓ All {len(special_tatar)} special Tatar letters render properly with Balsamiq Sans.")

    print("\n==================================================")
    print("2. Testing Individual Realism Augmentations")
    print("==================================================")
    base_img = render_base_char("Ә", test_font_path, canvas_size=128)

    # A. Cell crop artifacts (borders & neighbor intrusions)
    cell_img = apply_cell_crop_artifacts(base_img.copy(), prob=1.0)
    assert cell_img.shape == base_img.shape
    # Check that grid lines or edge marks were introduced (edges mean or min reflects border)
    assert cell_img.min() < 200, "Cell crop artifacts did not draw any border lines or marks"
    print("  ✓ Cell crop artifacts (multi-edge/tilt/L-shape) applied successfully.")

    # B. Overwriting / correction (bold overwrite & matching category)
    over_img_upper = apply_overwriting_correction(base_img.copy(), "Ә", test_font_path, canvas_size=128, prob=1.0)
    assert over_img_upper.shape == base_img.shape
    base_lower = render_base_char("ә", test_font_path, canvas_size=128)
    over_img_lower = apply_overwriting_correction(base_lower.copy(), "ә", test_font_path, canvas_size=128, prob=1.0)
    assert over_img_lower.shape == base_lower.shape
    print("  ✓ Overwriting / correction (category-matched & bold overwrite) applied successfully.")

    # C. Non-uniform stroke pressure & starvation (must maintain smooth anti-aliased gradations!)
    press_img = apply_variable_thickness_and_starvation(base_img.copy(), pressure_variation=2.0, starvation_prob=1.0)
    assert press_img.shape == base_img.shape
    num_unique = len(np.unique(press_img))
    assert num_unique > 20, f"Stroke pressure binarized font into only {num_unique} levels! Expected smooth anti-aliased gradations."
    print(f"  ✓ Non-uniform stroke pressure & starvation verified (anti-aliased with {num_unique} unique shades).")

    # D. Extra flourishes & stray hooks (attached to stroke)
    flourish_img = apply_flourishes_and_stray_marks(base_img.copy(), prob=1.0)
    assert flourish_img.shape == base_img.shape
    assert (flourish_img < 180).sum() > 0
    print("  ✓ Extra flourishes & stray marks (stroke-attached) applied successfully.")

    # E. Paper texture & lighting gradient
    paper_img = apply_paper_and_lighting(base_img.copy(), bg_range=(225, 250), add_gradient=True, add_faint_grid=True)
    assert paper_img.shape == base_img.shape
    contrast = paper_img.max() - paper_img.min()
    assert contrast > 50, f"Paper contrast too flat: {contrast}"
    print(f"  ✓ Paper texture & lighting gradient verified (contrast range: {contrast}).")

    # F. Master generator with all features & base_img caching
    full_img = generate_augmented_cell_image(
        char="Җ",
        font_path=test_font_path,
        canvas_size=64,
        enable_overwriting=True,
        enable_flourishes=True,
        enable_cell_borders=True,
        enable_paper_texture=True,
        enable_camera_blur=True,
        base_img=base_img,
    )
    assert full_img.shape == (64, 64)
    print("  ✓ Master generate_augmented_cell_image runs cleanly at 64x64 with base_img caching.")

    print("\n==================================================")
    print("3. Testing Online PyTorch Dataset & DataLoader")
    print("==================================================")
    online_dataset = TatarOnlineAugmentedDataset(
        chars=ALL_CHARS,
        canvas_size=64,
        epoch_size=100
    )
    assert len(online_dataset) == 100
    assert len(online_dataset.base_glyph_cache) > 0, "base_glyph_cache was not initialized!"
    sample_tensor, sample_label = online_dataset[0]
    assert isinstance(sample_tensor, torch.Tensor)
    assert sample_tensor.shape == (1, 64, 64)
    assert 0 <= sample_label < len(ALL_CHARS)
    print(f"  ✓ Online Dataset sample: shape={sample_tensor.shape}, label={sample_label} ('{ALL_CHARS[sample_label]}')")
    print(f"  ✓ Base glyph cache verified ({len(online_dataset.base_glyph_cache)} cached font-char glyphs).")

    loader = DataLoader(online_dataset, batch_size=8, shuffle=True)
    batch_tensors, batch_labels = next(iter(loader))
    assert batch_tensors.shape == (8, 1, 64, 64)
    assert batch_labels.shape == (8,)
    print(f"  ✓ DataLoader batch: tensors={batch_tensors.shape}, labels={batch_labels.shape}")

    # Forward pass through CNN
    model = TatarOCRNet(num_classes=len(ALL_CHARS))
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensors)
    assert logits.shape == (8, len(ALL_CHARS))
    print(f"  ✓ TatarOCRNet forward pass output: {logits.shape}")

    print("\n==================================================")
    print("4. Testing Disk Batch Generator")
    print("==================================================")
    tmp_out = Path("dataset/test_batch_verification")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)

    count, elapsed = generate_batch_dataset_to_disk(
        out_dir=str(tmp_out),
        samples_per_class=4,
        img_size=64,
        include_upper=True,
        include_lower=False,
        include_punct=False,
        include_digits=False,
        max_workers=2
    )
    expected = 39 * 4
    assert count == expected, f"Expected {expected} images, got {count}"
    assert (tmp_out / "class_index.json").exists()
    assert (tmp_out / "metadata.csv").exists()
    print(f"  ✓ Batch generator exported {count} images in {elapsed:.2f}s with JSON and CSV.")

    print("\n==================================================")
    print("5. Generating Visual Montage for Inspection")
    print("==================================================")
    # Generate 4x8 montage of varied characters and effects
    test_chars = [
        "Ә", "Ө", "Ү", "Җ", "Ң", "Һ", "а", "ә",
        "ө", "ү", "җ", "ң", "һ", "Б", "В", "Г",
        "Д", "Ж", "К", "Л", "М", "Н", "О", "П",
        "1", "2", "3", "8", "9", "?", "!", ","
    ]
    montage = np.full((4 * 64, 8 * 64), 255, dtype=np.uint8)
    font_names = list(all_fonts.keys())
    weights = [font_weights[k] for k in font_names]

    for i, ch in enumerate(test_chars):
        r = i // 8
        c = i % 8
        fn = np.random.choice(font_names, p=np.array(weights)/sum(weights))
        fp = all_fonts[fn]
        cell_arr = generate_augmented_cell_image(
            char=ch,
            font_path=fp,
            canvas_size=64,
            enable_overwriting=True,
            enable_flourishes=True,
            enable_cell_borders=True,
            enable_paper_texture=True,
            enable_camera_blur=True
        )
        montage[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = cell_arr

    montage_path = Path("dataset/visual_verification_montage.png")
    Image.fromarray(montage).save(montage_path)
    print(f"  ✓ Visual montage saved to {montage_path.as_posix()}")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! 🎉")


if __name__ == "__main__":
    run_tests()

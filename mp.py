import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import glob
    import random
    import math
    import io
    import json
    import time
    import base64
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import numpy as np
    import cv2
    import scipy.ndimage
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    mo.md(
        """
        # 📝 Tatar Handwritten Character & Punctuation Pipeline
        ### Synthetic Dataset Generator & CNN Pipeline for Tatar OCR («Кара куян»)
        This system generates realistic handwritten printed letters (**"рукописно-печатные буквы"**) for the Tatar Cyrillic alphabet (including the 6 specific Tatar letters: **Ә, Ө, Ү, Җ, Ң, Һ**), punctuation marks, and digits.

        **Realism Filters tailored for Cell Crops:**
        - 🔲 **Cell Border Lines & Neighbor Intrusions** (Partial crop box lines on 1-2 edges and stray strokes from adjacent cells)
        - ✍️ **Overwriting & Error Correction** (Bold correction over an underlying mistaken letter or double-strike re-tracing)
        - ⚖️ **Non-Uniform Stroke Pressure & Dry-Pen Starvation** (Downstrokes bold, crossbars fine, partial ink skipping)
        - 〰️ **Curved / Wobbly Strokes** (Elastic mesh displacement simulating muscle tremors & children's handwriting)
        - ➰ **Extra Flourishes & Hooks** (Starting/ending pen flicks, accidental loops, ink droplets)
        - 📄 **Classroom Paper & Lighting** (Uneven shadow gradients across the cell, paper pulp texture, faint graph grid)
        - ✏️ **Pencil Effect & Ink Bleed** (Porous graphite tooth or fountain/gel pen diffusion)
        - 📷 **Optical Softness** (Smartphone camera defocus & sensor noise)
        """
    )
    return (
        Dataset,
        Image,
        Path,
        ThreadPoolExecutor,
        base64,
        cv2,
        glob,
        io,
        json,
        mo,
        nn,
        np,
        os,
        random,
        time,
        torch,
    )


@app.cell
def _(mo):
    # 1. Tatar Cyrillic Alphabet Definition (39 letters, uppercase + lowercase)
    TATAR_UPPERCASE = list("АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯ")
    TATAR_LOWERCASE = list("аәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя")

    # 2. Punctuation Marks
    PUNCTUATION_CHARS = [".", ",", "!", "?", "-", ":", ";", '"', "'", "(", ")", "«", "»", "—"]

    # 3. Digits
    DIGIT_CHARS = list("0123456789")

    # Combined master list
    ALL_CHARS = TATAR_UPPERCASE + TATAR_LOWERCASE + PUNCTUATION_CHARS + DIGIT_CHARS
    char_to_id = {ch: i for i, ch in enumerate(ALL_CHARS)}
    id_to_char = {i: ch for i, ch in enumerate(ALL_CHARS)}

    mo.md(
        f"""
        ### 🔤 Vocabulary & Class Mapping
        - **Tatar Uppercase**: {len(TATAR_UPPERCASE)} classes (`{' '.join(TATAR_UPPERCASE[:10])}...`)
        - **Tatar Lowercase**: {len(TATAR_LOWERCASE)} classes (`{' '.join(TATAR_LOWERCASE[:10])}...`)
        - **Special Tatar Letters**: `Ә/ә, Ө/ө, Ү/ү, Җ/җ, Ң/ң, Һ/һ`
        - **Punctuation**: {len(PUNCTUATION_CHARS)} classes (`{' '.join(PUNCTUATION_CHARS)}`)
        - **Digits**: {len(DIGIT_CHARS)} classes (`{' '.join(DIGIT_CHARS)}`)
        - **Total Defined Classes**: **{len(ALL_CHARS)}**
        """
    )
    return (
        ALL_CHARS,
        DIGIT_CHARS,
        PUNCTUATION_CHARS,
        TATAR_LOWERCASE,
        TATAR_UPPERCASE,
    )


@app.cell
def _(Path, glob, mo, os):
    # Font Management
    FONTS_DIR = Path("fonts")
    FONTS_DIR.mkdir(exist_ok=True)

    local_font_paths = sorted(glob.glob(str(FONTS_DIR / "*.ttf")) + glob.glob(str(FONTS_DIR / "*.otf")))

    # Verified System fonts with 100% full Tatar support
    sys_fonts = [
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ]
    available_sys_fonts = [f for f in sys_fonts if os.path.exists(f)]

    ALL_FONT_PATHS = local_font_paths + available_sys_fonts
    font_options = {Path(p).stem: p for p in ALL_FONT_PATHS}

    # Weighting: prioritize authentic handwritten print fonts
    handwritten_print_names = {
        "BalsamiqSans-Regular", "BalsamiqSans-Bold", "BalsamiqSans-Italic",
        "PlaypenSans", "Pangolin", "DidactGothic", "Nunito-Regular",
        "Rubik-Regular", "Marmelad-Regular", "Comfortaa", "bahnschrift"
    }
    font_weights = {
        p: (5.0 if Path(p).stem in handwritten_print_names else 1.5)
        for p in ALL_FONT_PATHS
    }

    mo.md(
        f"""
        ### 🖋️ Loaded Handwriting & Base Fonts
        Found **{len(local_font_paths)}** handwriting fonts in `./fonts` (including *Balsamiq Sans, Playpen Sans, Pangolin, Didact Gothic, Nunito, Rubik*) and **{len(available_sys_fonts)}** system fallback fonts.
        *Active Fonts ({len(ALL_FONT_PATHS)}):* {', '.join([Path(p).stem for p in ALL_FONT_PATHS])}
        """
    )
    return ALL_FONT_PATHS, font_options, font_weights


@app.cell
def _(cv2, np, random):
    # Core Image Generation & Distortion Engine
    from tatar_ocr_augmentation import (
        render_base_char,
        apply_elastic_wobble,
        apply_variable_thickness_and_starvation as apply_variable_thickness,
        apply_overwriting_correction,
        apply_flourishes_and_stray_marks,
        apply_cell_crop_artifacts,
        apply_camera_defocus,
        apply_affine_cell_placement as apply_affine_jitter,
        apply_paper_and_lighting as apply_paper_background,
        generate_augmented_cell_image,
    )

    def apply_pencil_graphite_effect(arr, faintness=0.5, grain=30):
        """Simulates faint graphite pencil writing with porous texture."""
        stroke_mask = (arr < 220).astype(np.float32)
        if stroke_mask.sum() == 0:
            return arr
        base_intensity = 110 + int(faintness * 75)
        grain_noise = np.random.normal(0, grain, arr.shape)
        out = np.where(stroke_mask > 0.5, base_intensity + grain_noise, arr.astype(np.float32))
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_ink_bleed(arr, bleed_amount=1.5):
        """Simulates gel pen or fountain pen ink spreading slightly into paper fibers."""
        if bleed_amount <= 0.1:
            return arr
        stroke_inv = 255 - arr
        kernel_size = max(3, int(bleed_amount * 2) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(stroke_inv, kernel)
        blurred = cv2.GaussianBlur(dilated, (3, 3), 0.8)
        combined = np.maximum(stroke_inv, (blurred * 0.7).astype(np.uint8))
        return 255 - combined

    def apply_stray_artifacts(arr, num_specks=4):
        """Adds accidental stray ink dots and paper fiber dust."""
        res = arr.copy()
        h, w = res.shape
        for _ in range(num_specks):
            sx = random.randint(2, w - 3)
            sy = random.randint(2, h - 3)
            cv2.circle(res, (sx, sy), random.choice([1, 1, 2]), random.randint(30, 120), -1)
        return res

    def generate_augmented_char(
        char,
        font_path,
        canvas_size=64,
        elastic_strength=18.0,
        pressure_variation=1.6,
        pencil_mode=False,
        pencil_faintness=0.5,
        ink_bleed_mode=False,
        stray_dots=4,
        max_rotation=12.0,
        apply_paper=True,
        cell_borders_mode=True,
        overwriting_mode=False,
        flourishes_mode=True,
        camera_blur=True,
        **kwargs,
    ):
        """Comprehensive character generator utilizing the optimized Tatar OCR engine."""
        internal_size = max(128, canvas_size * 2)
        raw = render_base_char(char, font_path, canvas_size=internal_size)

        # 1. Overwriting / correction (underlying error letter, double-strike, or scratch)
        if overwriting_mode:
            img = apply_overwriting_correction(raw, char, font_path, canvas_size=internal_size, prob=0.85)
        else:
            img = raw.copy()

        # 2. Non-uniform stroke thickness & pen starvation (anti-aliased)
        img = apply_variable_thickness(img, pressure_variation=pressure_variation)

        # 3. Ink bleed or pencil graphite texture
        if ink_bleed_mode:
            img = apply_ink_bleed(img, bleed_amount=2.0)
        elif pencil_mode:
            img = apply_pencil_graphite_effect(img, faintness=pencil_faintness, grain=30)

        # 4. Extra flourishes & stray hooks (attached to strokes)
        if flourishes_mode:
            img = apply_flourishes_and_stray_marks(img, prob=0.50)

        # 5. Elastic wobbly curves
        img = apply_elastic_wobble(img, alpha=elastic_strength, sigma=6.0)

        # 6. Affine slant, rotation, scale, offset
        img = apply_affine_jitter(img, max_angle=max_rotation, max_shear=8.0)

        # 7. Stray specks
        if stray_dots > 0:
            img = apply_stray_artifacts(img, num_specks=stray_dots)

        # 8. Resize to target canvas size
        if internal_size != canvas_size:
            img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

        # 9. Cell borders & neighbor stroke intrusions
        if cell_borders_mode:
            img = apply_cell_crop_artifacts(img, prob=0.85)

        # 10. Camera defocus
        if camera_blur:
            img = apply_camera_defocus(img, prob=0.40)

        # 11. Paper texture & lighting gradient
        if apply_paper:
            img = apply_paper_background(img)

        return img

    return generate_augmented_char, render_base_char


@app.cell
def _(ALL_CHARS, font_options, mo):
    # Interactive UI Widgets for Live Tuning

    char_dropdown = mo.ui.dropdown(
        options=ALL_CHARS,
        value="Ә",
        label="Select Character:",
    )

    random_char_btn = mo.ui.button(
        label="🎲 Random Letter",
        kind="neutral",
    )

    font_dropdown = mo.ui.dropdown(
        options=font_options,
        value="BalsamiqSans-Regular" if "BalsamiqSans-Regular" in font_options else list(font_options.keys())[0],
        label="Base Font:",
    )

    preset_dropdown = mo.ui.dropdown(
        options=[
            "Balanced Real Cell",
            "Handwritten Print in Cell",
            "Overwritten / Corrected Letter",
            "Pencil in Grid Cell",
            "Heavy Ink Bleed",
            "Uneven Pressure & Faint Skips",
            "Wobbly / Shaky Hand",
        ],
        value="Balanced Real Cell",
        label="Style Preset:",
    )

    size_dropdown = mo.ui.dropdown(
        options=[32, 48, 64, 96, 128],
        value=64,
        label="Output Size (px):",
    )

    elastic_slider = mo.ui.slider(
        start=0,
        stop=45,
        step=1,
        value=18,
        label="Wobbly Curve Distortion",
    )

    pressure_slider = mo.ui.slider(
        start=0.0,
        stop=3.0,
        step=0.1,
        value=1.6,
        label="Variable Stroke Thickness (Uneven Pressure)",
    )

    rotation_slider = mo.ui.slider(
        start=0,
        stop=30,
        step=1,
        value=12,
        label="Max Rotation (°)",
    )

    stray_dots_slider = mo.ui.slider(
        start=0,
        stop=12,
        step=1,
        value=4,
        label="Stray Ink Specks",
    )

    cell_borders_toggle = mo.ui.checkbox(
        value=True,
        label="Cut Cell Borders & Intrusions",
    )

    overwriting_toggle = mo.ui.checkbox(
        value=False,
        label="Overwriting / Correction (Mistake Overwrite)",
    )

    flourishes_toggle = mo.ui.checkbox(
        value=True,
        label="Stray Flourishes & Hooks",
    )

    pencil_toggle = mo.ui.checkbox(
        value=False,
        label="Pencil Mode (Pale & Grainy)",
    )

    ink_bleed_toggle = mo.ui.checkbox(
        value=False,
        label="Thick Ink Bleed",
    )

    samples_count_slider = mo.ui.slider(
        start=2,
        stop=12,
        step=1,
        value=6,
        label="Preview Grid Samples",
    )

    refresh_preview_btn = mo.ui.button(
        label="🔄 Regenerate Preview",
        kind="primary",
    )

    mo.vstack([
        mo.md("### ⚙️ Live Interactive Generator Controls"),
        mo.hstack([char_dropdown, random_char_btn, font_dropdown, preset_dropdown, size_dropdown], wrap=True),
        mo.hstack([elastic_slider, pressure_slider, rotation_slider, stray_dots_slider], wrap=True),
        mo.hstack([cell_borders_toggle, overwriting_toggle, flourishes_toggle, pencil_toggle, ink_bleed_toggle], wrap=True),
        mo.hstack([samples_count_slider, refresh_preview_btn], wrap=True),
    ])
    return (
        cell_borders_toggle,
        char_dropdown,
        elastic_slider,
        flourishes_toggle,
        font_dropdown,
        ink_bleed_toggle,
        overwriting_toggle,
        pencil_toggle,
        preset_dropdown,
        pressure_slider,
        random_char_btn,
        refresh_preview_btn,
        rotation_slider,
        samples_count_slider,
        size_dropdown,
        stray_dots_slider,
    )


@app.cell
def _(
    ALL_CHARS,
    Image,
    base64,
    cell_borders_toggle,
    char_dropdown,
    elastic_slider,
    flourishes_toggle,
    font_dropdown,
    font_options,
    generate_augmented_char,
    ink_bleed_toggle,
    io,
    mo,
    overwriting_toggle,
    pencil_toggle,
    preset_dropdown,
    pressure_slider,
    random,
    random_char_btn,
    refresh_preview_btn,
    render_base_char,
    rotation_slider,
    samples_count_slider,
    size_dropdown,
    stray_dots_slider,
):
    # Reactive Live Preview Display Cell
    _ = refresh_preview_btn.value

    # Pick character
    current_char = char_dropdown.value
    if random_char_btn.value:
        current_char = random.choice(ALL_CHARS)

    current_font = font_options.get(font_dropdown.value, list(font_options.values())[0])
    current_size = int(size_dropdown.value)

    # Preset logic override
    p_name = preset_dropdown.value
    cur_elastic = elastic_slider.value
    cur_pressure = pressure_slider.value
    cur_pencil = pencil_toggle.value
    cur_bleed = ink_bleed_toggle.value
    cur_rot = rotation_slider.value
    cur_dots = stray_dots_slider.value
    cur_borders = cell_borders_toggle.value
    cur_overwrite = overwriting_toggle.value
    cur_flourishes = flourishes_toggle.value

    if p_name == "Balanced Real Cell":
        cur_borders = True
        cur_flourishes = True
        cur_pressure = 1.6
    elif p_name == "Handwritten Print in Cell":
        cur_borders = True
        cur_flourishes = False
        cur_pressure = 1.3
    elif p_name == "Overwritten / Corrected Letter":
        cur_overwrite = True
        cur_borders = True
        cur_pressure = 2.2
    elif p_name == "Pencil in Grid Cell":
        cur_pencil = True
        cur_bleed = False
        cur_borders = True
    elif p_name == "Heavy Ink Bleed":
        cur_bleed = True
        cur_pencil = False
        cur_borders = True
    elif p_name == "Uneven Pressure & Faint Skips":
        cur_pressure = 2.5
        cur_borders = True
    elif p_name == "Wobbly / Shaky Hand":
        cur_elastic = 35
        cur_borders = True

    def np_to_data_url(arr):
        pil_im = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # Generate baseline clean glyph
    clean_arr = render_base_char(current_char, current_font, canvas_size=current_size)
    clean_url = np_to_data_url(clean_arr)

    # Generate N augmented variations
    num_samples = samples_count_slider.value
    sample_urls = []
    for _ in range(num_samples):
        aug_arr = generate_augmented_char(
            char=current_char,
            font_path=current_font,
            canvas_size=current_size,
            elastic_strength=cur_elastic,
            pressure_variation=cur_pressure,
            pencil_mode=cur_pencil,
            ink_bleed_mode=cur_bleed,
            stray_dots=cur_dots,
            max_rotation=cur_rot,
            cell_borders_mode=cur_borders,
            overwriting_mode=cur_overwrite,
            flourishes_mode=cur_flourishes,
            camera_blur=True,
        )
        sample_urls.append(np_to_data_url(aug_arr))

    preview_cards_html = f"""
    <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 12px; font-family: sans-serif;">
        <div style="display: flex; align-items: center; gap: 24px; padding: 14px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div>
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 4px;">Original Glyph</div>
                <img src="{clean_url}" style="width: {current_size}px; height: {current_size}px; border: 2px solid #cbd5e1; border-radius: 6px; image-rendering: pixelated;"/>
            </div>
            <div>
                <div style="font-size: 18px; font-weight: 700; color: #0f172a;">Active Character: <span style="font-size: 26px; color: #2563eb;">{current_char}</span> (Code: U+{ord(current_char):04X})</div>
                <div style="font-size: 13px; color: #475569;">Font: <b>{font_dropdown.value}</b> | Canvas: <b>{current_size}x{current_size} px</b> | Preset: <b>{p_name}</b></div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    Cell Borders: <b>{'Yes' if cur_borders else 'No'}</b> |
                    Overwrite: <b>{'Yes' if cur_overwrite else 'No'}</b> |
                    Flourishes: <b>{'Yes' if cur_flourishes else 'No'}</b> |
                    Pressure: <b>{cur_pressure:.1f}</b> |
                    Pencil: <b>{'Yes' if cur_pencil else 'No'}</b>
                </div>
            </div>
        </div>

        <div>
            <div style="font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 8px;">🎨 Generated Realistic Handwriting Cell Variations:</div>
            <div style="display: flex; flex-wrap: wrap; gap: 14px;">
                {''.join([f'''
                <div style="display: flex; flex-direction: column; align-items: center; background: #ffffff; padding: 6px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                    <img src="{url}" style="width: {current_size}px; height: {current_size}px; border-radius: 4px; image-rendering: pixelated;"/>
                    <span style="font-size: 10px; color: #94a3b8; margin-top: 4px;">var #{i+1}</span>
                </div>
                ''' for i, url in enumerate(sample_urls)])}
            </div>
        </div>
    </div>
    """
    mo.Html(preview_cards_html)
    return


@app.cell
def _(mo):
    # Batch Dataset Generator UI Form (Configured for 15k-20k Images)
    dataset_dir_input = mo.ui.text(
        value="dataset/tatar_hw_dataset",
        label="Export Directory:",
    )

    samples_per_class_slider = mo.ui.slider(
        start=10,
        stop=300,
        step=10,
        value=180,
        label="Samples per Character Class (180 samples * 102 classes = ~18,360 images):",
    )

    export_size_dropdown = mo.ui.dropdown(
        options=[32, 64, 96, 128],
        value=64,
        label="Image Resolution (px):",
    )

    inc_upper_cb = mo.ui.checkbox(value=True, label="Uppercase (39)")
    inc_lower_cb = mo.ui.checkbox(value=True, label="Lowercase (39)")
    inc_punct_cb = mo.ui.checkbox(value=True, label="Punctuation (14)")
    inc_digits_cb = mo.ui.checkbox(value=True, label="Digits (10)")

    generate_dataset_btn = mo.ui.run_button(
        label="🚀 Generate 15k-20k Tatar Dataset to Disk",
        kind="primary",
    )

    mo.vstack([
        mo.md("--- \n### 📦 Batch Synthetic Dataset Generator (15,000 – 20,000 Samples)"),
        mo.hstack([dataset_dir_input, samples_per_class_slider, export_size_dropdown], wrap=True),
        mo.hstack([inc_upper_cb, inc_lower_cb, inc_punct_cb, inc_digits_cb], wrap=True),
        generate_dataset_btn,
    ])
    return (
        dataset_dir_input,
        export_size_dropdown,
        generate_dataset_btn,
        inc_digits_cb,
        inc_lower_cb,
        inc_punct_cb,
        inc_upper_cb,
        samples_per_class_slider,
    )


@app.cell
def _(
    DIGIT_CHARS,
    Image,
    PUNCTUATION_CHARS,
    Path,
    TATAR_LOWERCASE,
    TATAR_UPPERCASE,
    ThreadPoolExecutor,
    dataset_dir_input,
    export_size_dropdown,
    font_weights,
    generate_augmented_char,
    generate_dataset_btn,
    inc_digits_cb,
    inc_lower_cb,
    inc_punct_cb,
    inc_upper_cb,
    json,
    mo,
    random,
    samples_per_class_slider,
    time,
):
    # High-Speed Multi-Threaded Batch Generator Execution Cell
    if not generate_dataset_btn.value:
        _gen_msg = mo.md("*Configure parameters above and click **Generate 15k-20k Tatar Dataset to Disk** to export.*")
    else:
        selected_chars = []
        if inc_upper_cb.value:
            selected_chars.extend(TATAR_UPPERCASE)
        if inc_lower_cb.value:
            selected_chars.extend(TATAR_LOWERCASE)
        if inc_punct_cb.value:
            selected_chars.extend(PUNCTUATION_CHARS)
        if inc_digits_cb.value:
            selected_chars.extend(DIGIT_CHARS)

        out_dir = Path(dataset_dir_input.value)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_size = int(export_size_dropdown.value)
        samples_per_char = int(samples_per_class_slider.value)
        total_target = len(selected_chars) * samples_per_char

        start_time = time.time()
        class_map = {ch: idx for idx, ch in enumerate(selected_chars)}
        with open(out_dir / "class_index.json", "w", encoding="utf-8") as f:
            json.dump({str(idx): ch for ch, idx in class_map.items()}, f, ensure_ascii=False, indent=2)

        font_keys = list(font_weights.keys())
        weights_list = [font_weights[k] for k in font_keys]

        def generate_sample(item):
            ch, cls_id, s_idx, char_folder = item
            chosen_font = random.choices(font_keys, weights=weights_list, k=1)[0]
            arr = generate_augmented_char(
                char=ch,
                font_path=chosen_font,
                canvas_size=img_size,
                elastic_strength=random.uniform(8.0, 24.0),
                pressure_variation=random.uniform(1.0, 2.4),
                pencil_mode=(random.random() < 0.20),
                ink_bleed_mode=(random.random() < 0.20),
                stray_dots=random.randint(0, 5),
                max_rotation=random.uniform(3.0, 14.0),
                cell_borders_mode=(random.random() < 0.70),
                overwriting_mode=(random.random() < 0.20),
                flourishes_mode=(random.random() < 0.40),
                camera_blur=(random.random() < 0.40),
                apply_paper=True,
            )
            img_path = char_folder / f"char_{cls_id:03d}_{s_idx:04d}.png"
            Image.fromarray(arr).save(img_path)
            return f"{img_path.as_posix()},{cls_id},{ord(ch)}"

        tasks = []
        for ch in selected_chars:
            cls_id = class_map[ch]
            char_folder = out_dir / f"cls_{cls_id:03d}"
            char_folder.mkdir(exist_ok=True)
            for s_idx in range(samples_per_char):
                tasks.append((ch, cls_id, s_idx, char_folder))

        metadata_records = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for rec in executor.map(generate_sample, tasks):
                metadata_records.append(rec)

        with open(out_dir / "metadata.csv", "w", encoding="utf-8") as f:
            f.write("image_path,class_id,unicode_ord\n" + "\n".join(metadata_records) + "\n")

        elapsed = time.time() - start_time
        _gen_msg = mo.md(
            f"""
            ✅ **Dataset Generation Complete!**
            - **Total Images Generated:** `{len(metadata_records)}`
            - **Unique Classes:** `{len(selected_chars)}`
            - **Samples per Class:** `{samples_per_char}`
            - **Image Size:** `{img_size}x{img_size} px`
            - **Throughput:** `{len(metadata_records) / elapsed:.1f} images/sec`
            - **Time Elapsed:** `{elapsed:.2f} seconds`
            - **Saved to:** `{out_dir.resolve().as_posix()}`
            - **Index Files:** `class_index.json` and `metadata.csv` written.
            """
        )

    _gen_msg
    return


@app.cell
def _(
    ALL_CHARS,
    ALL_FONT_PATHS,
    Dataset,
    Image,
    Path,
    font_weights,
    generate_augmented_char,
    mo,
    nn,
    np,
    random,
    torch,
):
    # PyTorch Dataset Loaders: Online On-The-Fly Augmentation & Disk Loader

    class TatarOnlineAugmentedDataset(Dataset):
        """
        PyTorch Dataset generating infinite, realistic handwriting augmentations on the fly.
        Optimal for CNN training:
        - Zero disk space footprint (no 20,000 PNG files stored on disk).
        - Every epoch sees completely new variations of stroke tremor, pen starvation,
          overwriting, and cell boundaries (prevents memorization / overfitting).
        """

        def __init__(self, chars=None, canvas_size=64, epoch_size=18360):
            self.chars = chars if chars is not None else ALL_CHARS
            self.canvas_size = canvas_size
            self.epoch_size = epoch_size
            self.font_paths = ALL_FONT_PATHS
            self.weights = [font_weights.get(p, 1.0) for p in self.font_paths]

        def __len__(self):
            return self.epoch_size

        def __getitem__(self, idx):
            char_idx = idx % len(self.chars)
            target_char = self.chars[char_idx]
            chosen_font = random.choices(self.font_paths, weights=self.weights, k=1)[0]

            img_arr = generate_augmented_char(
                char=target_char,
                font_path=chosen_font,
                canvas_size=self.canvas_size,
                elastic_strength=random.uniform(8.0, 24.0),
                pressure_variation=random.uniform(1.0, 2.4),
                pencil_mode=(random.random() < 0.20),
                ink_bleed_mode=(random.random() < 0.20),
                stray_dots=random.randint(0, 5),
                max_rotation=random.uniform(3.0, 14.0),
                cell_borders_mode=(random.random() < 0.70),
                overwriting_mode=(random.random() < 0.20),
                flourishes_mode=(random.random() < 0.40),
                camera_blur=(random.random() < 0.40),
                apply_paper=True,
            )

            # Convert to [1, H, W] tensor in [-1, 1]
            tensor = torch.from_numpy(img_arr.astype(np.float32) / 255.0).unsqueeze(0)
            tensor = (tensor - 0.5) / 0.5
            return tensor, char_idx

    class TatarCharDataset(Dataset):
        """PyTorch Dataset loader for saved static disk images."""

        def __init__(self, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.samples = []

            meta_file = self.root_dir / "metadata.csv"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")[1:]
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split(",")
                            self.samples.append((parts[0], int(parts[1])))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("L")
            img_arr = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_arr).unsqueeze(0)
            tensor = (tensor - 0.5) / 0.5
            return tensor, label

    class TatarOCRNet(nn.Module):
        """
        Lightweight Convolutional Neural Network for Tatar handwritten character recognition.
        Designed for fast mobile inference in the «Кара куян» app.
        """

        def __init__(self, num_classes=102):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),
                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2),
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    mo.md(
        f"""
        ---
        ### 🧠 PyTorch Pipeline: Online vs. Offline Augmentation

        #### 💡 Optimization Insight: Why Online Augmentation is Recommended for Training
        | Parameter | Offline Disk Generation | Online On-The-Fly Dataset (`TatarOnlineAugmentedDataset`) |
        | :--- | :--- | :--- |
        | **Disk Storage** | Hundreds of MBs / GBs for 20k images | **0 MB** (Zero disk footprint) |
        | **Sample Diversity** | Fixed set of static images | **Infinite** (New wobble, pressure & overwrite per epoch) |
        | **Overfitting Risk** | Higher (model memorizes static specks) | **Minimal** (Continual regularizing noise) |
        | **Use Case** | Fixed benchmark / manual visual audits | **Training production models** |

        ```python
        # Recommended Training Setup (Online Infinite Augmentation):
        train_dataset = TatarOnlineAugmentedDataset(epoch_size=18360)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        model = TatarOCRNet(num_classes=len(train_dataset.chars))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()

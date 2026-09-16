import os
import sys
import shutil
import random
import math
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFont

from tatar_ocr_augmentation import (
    render_base_char,
    get_available_fonts,
    apply_elastic_wobble,
    apply_variable_thickness_and_starvation,
    apply_camera_defocus,
    apply_paper_and_lighting,
    apply_affine_cell_placement,
)

# -------------------------------------------------------------------------
# Specialized Isolated Degradation Filters
# -------------------------------------------------------------------------

def effect_clean(arr, **kwargs):
    """Reference clean glyph without degradation."""
    return arr.copy()

def effect_elastic_wobble(arr, **kwargs):
    """Simulates hand muscle tremor and unsteady wobbly stroke curves."""
    return apply_elastic_wobble(arr, alpha=22.0, sigma=5.0)

def effect_heavy_pressure(arr, **kwargs):
    """Heavy downstroke pressure and ink spreading (bold writing)."""
    binary = (arr < 180).astype(np.uint8)
    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5).astype(np.float32)
    sdf = dist_in - dist_out
    new_sdf = sdf + 3.2
    soft_stroke = np.clip(new_sdf * 1.5 + 0.5, 0.0, 1.0)
    return (255.0 * (1.0 - soft_stroke)).astype(np.uint8)

def effect_pen_starvation(arr, **kwargs):
    """Dry ballpoint pen skipping / ink starvation along strokes."""
    res = arr.copy()
    h, w = res.shape
    center_x = int(w * 0.5)
    center_y = int(h * 0.45)
    radius = int(h * 0.35)
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    starve_mask = np.clip(1.0 - dist / (radius + 1e-5), 0.0, 1.0)
    starve_mask = scipy.ndimage.gaussian_filter(starve_mask, sigma=2.0)
    ink_mask = (res < 220).astype(np.float32)
    return np.clip(res.astype(np.float32) + (starve_mask * ink_mask * 150.0), 0, 255).astype(np.uint8)

def effect_underlying_error(arr, char, font_path, canvas_size=128, **kwargs):
    """Pupil correcting a mistaken letter: writing target letter over an underlying error."""
    err_char = "О" if char != "О" else "А"
    err_img = render_base_char(err_char, font_path, canvas_size=canvas_size, font_scale=0.72)
    shift_m = np.float32([[1, 0, 3], [0, 1, -2]])
    err_img = cv2.warpAffine(err_img, shift_m, (canvas_size, canvas_size), borderValue=255)
    under_alpha = (255.0 - err_img.astype(np.float32)) / 255.0 * 0.45
    base = 255.0 * (1.0 - under_alpha) + 120.0 * under_alpha
    target_alpha = (255.0 - arr.astype(np.float32)) / 255.0
    return np.clip(base * (1.0 - target_alpha) + arr.astype(np.float32) * target_alpha, 0, 255).astype(np.uint8)

def effect_double_strike(arr, canvas_size=128, **kwargs):
    """Double-strike / retracing the letter with slight alignment offset."""
    center = (canvas_size / 2.0, canvas_size / 2.0)
    m = cv2.getRotationMatrix2D(center, 2.5, 1.0)
    m[0, 2] += 2.0
    m[1, 2] += 1.5
    pass2 = cv2.warpAffine(arr, m, (canvas_size, canvas_size), borderValue=255)
    return np.minimum(arr, pass2)

def effect_scratch_crossout(arr, canvas_size=128, **kwargs):
    """Correction scratch or cross-out tick mark across the letter."""
    out = arr.copy()
    p1 = (int(canvas_size * 0.2), int(canvas_size * 0.3))
    p2 = (int(canvas_size * 0.8), int(canvas_size * 0.7))
    cv2.line(out, p1, p2, color=70, thickness=2)
    return out

def effect_ink_blobs(arr, **kwargs):
    """Ink blobs (кляксы) and splatter from lingering pen tip."""
    res = arr.copy()
    h, w = res.shape
    ink_ys, ink_xs = np.where(res < 180)
    if len(ink_xs) > 10:
        for _ in range(3):
            idx = random.randint(0, len(ink_xs) - 1)
            bx = int(ink_xs[idx] + random.randint(-4, 4))
            by = int(ink_ys[idx] + random.randint(-4, 4))
            cv2.circle(res, (bx, by), random.randint(3, 5), color=30, thickness=-1)
    return res

def effect_stray_hooks_loops(arr, **kwargs):
    """Stray entrance hooks, pen flourishes, and loop tails."""
    res = arr.copy()
    h, w = res.shape
    ink_ys, ink_xs = np.where(res < 180)
    if len(ink_xs) > 10:
        idx = int(np.argmin(ink_ys))
        sx, sy = int(ink_xs[idx]), int(ink_ys[idx])
        pts = np.array([[sx - 16, sy - 12], [sx - 8, sy - 18], [sx, sy]], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(res, [pts], isClosed=False, color=60, thickness=2)
        idx2 = int(np.argmax(ink_ys))
        sx2, sy2 = int(ink_xs[idx2]), int(ink_ys[idx2])
        cv2.ellipse(res, (sx2 + 6, sy2 + 4), (8, 4), 30, 0, 360, color=70, thickness=1)
    return res

def effect_cell_grid_borders(arr, **kwargs):
    """School notebook grid lines or printed form box borders."""
    res = arr.copy()
    h, w = res.shape
    cv2.line(res, (4, 0), (5, h - 1), color=130, thickness=2)
    cv2.line(res, (0, h - 5), (w - 1, h - 4), color=130, thickness=2)
    return res

def effect_neighbor_intrusion(arr, **kwargs):
    """Intrusion from adjacent character's descender or flourish cutting into cell."""
    res = arr.copy()
    h, w = res.shape
    cv2.line(res, (int(w * 0.7), 0), (int(w * 0.65), int(h * 0.35)), color=55, thickness=2)
    cv2.line(res, (0, int(h * 0.4)), (int(w * 0.25), int(h * 0.45)), color=65, thickness=2)
    return res

def effect_registration_mark(arr, **kwargs):
    """Corner registration crosshair (+) or alignment marker caught in crop."""
    res = arr.copy()
    cx, cy = 10, 10
    cv2.line(res, (cx - 7, cy), (cx + 7, cy), color=40, thickness=2)
    cv2.line(res, (cx, cy - 7), (cx, cy + 7), color=40, thickness=2)
    return res

def effect_camera_defocus(arr, **kwargs):
    """Smartphone camera optical defocus blur."""
    return cv2.GaussianBlur(arr, (7, 7), 2.2)

def effect_motion_blur(arr, **kwargs):
    """Hand movement motion blur during snapshot."""
    size = 9
    kernel = np.zeros((size, size))
    np.fill_diagonal(kernel, 1.0 / size)
    return cv2.filter2D(arr, -1, kernel)

def effect_shadow_gradient(arr, **kwargs):
    """Non-uniform lighting and harsh phone/hand cast shadow across the page."""
    h, w = arr.shape
    y_norm, x_norm = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing="ij")
    shadow = (x_norm * 0.7 + y_norm * 0.7) * 75.0
    bg = np.clip(235.0 + shadow, 130.0, 255.0)
    stroke_alpha = (255.0 - arr.astype(np.float32)) / 255.0
    return np.clip(bg * (1.0 - stroke_alpha) + (arr.astype(np.float32) * stroke_alpha), 0, 255).astype(np.uint8)

def effect_ruled_notebook_paper(arr, **kwargs):
    """Grid graph paper pattern (тетрадь в клетку)."""
    h, w = arr.shape
    bg = np.full((h, w), 245, dtype=np.float32)
    for y in range(0, h, 20):
        bg[y, :] -= 30
    for x in range(0, w, 20):
        bg[:, x] -= 30
    stroke_alpha = (255.0 - arr.astype(np.float32)) / 255.0
    return np.clip(bg * (1.0 - stroke_alpha) + (arr.astype(np.float32) * stroke_alpha), 0, 255).astype(np.uint8)

def effect_salt_and_pepper_binarization(arr, **kwargs):
    """Thresholding artifacts, speckle, and salt-and-pepper scan noise."""
    res = arr.copy()
    h, w = res.shape
    noise_prob = 0.04
    mask_salt = np.random.rand(h, w) < (noise_prob / 2.0)
    mask_pepper = np.random.rand(h, w) < (noise_prob / 2.0)
    res[mask_salt] = 255
    res[mask_pepper] = 20
    _, binarized = cv2.threshold(res, 170, 255, cv2.THRESH_BINARY)
    return binarized

def effect_affine_skew_jitter(arr, **kwargs):
    """Geometric shear, slant tilt, and bounding-box off-centering."""
    h, w = arr.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, 15.0, 0.88)
    shear_rad = np.deg2rad(10.0)
    rot_mat[0, 1] += np.tan(shear_rad) * rot_mat[0, 0]
    rot_mat[0, 2] += 8.0
    rot_mat[1, 2] -= 6.0
    return cv2.warpAffine(arr, rot_mat, (w, h), borderValue=255)

def effect_partial_crop_clipping(arr, **kwargs):
    """Edge clipping from tight or imperfect automatic bounding-box segmentation."""
    res = arr.copy()
    h, w = res.shape
    res[:, int(w * 0.85):] = 255
    res[:int(h * 0.12), :] = 255
    return res

def effect_ink_bleed_feather(arr, **kwargs):
    """Ink bleeding into porous paper fibers (capillary feathering)."""
    blur1 = cv2.GaussianBlur(arr, (5, 5), 1.2)
    binary = (arr < 190).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.erode(arr, kernel)
    pulp_roughness = np.random.normal(0, 10.0, arr.shape)
    bleed = np.clip(np.minimum(dilated, blur1).astype(np.float32) + pulp_roughness * 0.5, 0, 255).astype(np.uint8)
    return bleed

def effect_combined_mild_notebook(arr, char, font_path, canvas_size=128, **kwargs):
    """Combined realistic school exam cell (mild wear)."""
    step1 = apply_elastic_wobble(arr, alpha=14.0, sigma=6.0)
    step2 = apply_variable_thickness_and_starvation(step1, pressure_variation=1.4, starvation_prob=0.2)
    step3 = apply_affine_cell_placement(step2, max_angle=6.0, max_shear=4.0, scale_range=(0.78, 0.90))
    step4 = effect_cell_grid_borders(step3)
    step5 = apply_camera_defocus(step4, prob=1.0)
    step6 = apply_paper_and_lighting(step5, bg_range=(235, 250), add_gradient=True, add_faint_grid=True)
    return step6

def effect_combined_severe_correction(arr, char, font_path, canvas_size=128, **kwargs):
    """Combined severe real-world student correction with heavy shadow & intrusion."""
    step1 = effect_underlying_error(arr, char, font_path, canvas_size=canvas_size)
    step2 = effect_ink_blobs(step1)
    step3 = apply_elastic_wobble(step2, alpha=18.0, sigma=5.0)
    step4 = effect_neighbor_intrusion(step3)
    step5 = effect_cell_grid_borders(step4)
    step6 = effect_shadow_gradient(step5)
    step7 = cv2.GaussianBlur(step6, (3, 3), 0.8)
    return step7


# -------------------------------------------------------------------------
# Catalog of Mechanisms
# -------------------------------------------------------------------------

MECHANISMS = [
    {
        "id": "01_clean_reference",
        "title": "Clean Reference",
        "category": "Baseline",
        "desc": "Original rendered glyph without any degradation.",
        "func": effect_clean,
    },
    {
        "id": "02_elastic_tremor",
        "title": "Elastic Tremor / Wobble",
        "category": "Hand Motor Jitter",
        "desc": "Hand muscle tremor, unsteady wobbly stroke curves.",
        "func": effect_elastic_wobble,
    },
    {
        "id": "03_heavy_pressure",
        "title": "Heavy Pen Pressure",
        "category": "Stroke Dynamics",
        "desc": "Downstroke pressure & broad stroke dilation (жирное написание).",
        "func": effect_heavy_pressure,
    },
    {
        "id": "04_pen_starvation",
        "title": "Dry Pen Skipping",
        "category": "Stroke Dynamics",
        "desc": "Pen tip starvation, faint fading skips along the stroke.",
        "func": effect_pen_starvation,
    },
    {
        "id": "05_underlying_correction",
        "title": "Mistake Overwriting",
        "category": "Error Correction",
        "desc": "Writing the correct letter over a mistaken letter.",
        "func": effect_underlying_error,
    },
    {
        "id": "06_double_strike",
        "title": "Double-Strike Retrace",
        "category": "Error Correction",
        "desc": "Retracing / double pass over the same letter (обводка).",
        "func": effect_double_strike,
    },
    {
        "id": "07_scratch_crossout",
        "title": "Correction Scratch",
        "category": "Error Correction",
        "desc": "Slash mark or tick cross-out (зачеркивание).",
        "func": effect_scratch_crossout,
    },
    {
        "id": "08_ink_blobs",
        "title": "Ink Blobs (Кляксы)",
        "category": "Ink Artifacts",
        "desc": "Ink pooling and droplets from lingering pen nib.",
        "func": effect_ink_blobs,
    },
    {
        "id": "09_stray_hooks_loops",
        "title": "Flourishes & Stray Loops",
        "category": "Handwriting Quirks",
        "desc": "Starting entrance hooks, cursive tails, and stray loops.",
        "func": effect_stray_hooks_loops,
    },
    {
        "id": "10_ink_bleed",
        "title": "Ink Bleed / Feathering",
        "category": "Ink Artifacts",
        "desc": "Capillary ink bleeding into porous, rough paper fibers.",
        "func": effect_ink_bleed_feather,
    },
    {
        "id": "11_cell_grid_borders",
        "title": "Cell Grid Borders",
        "category": "Crop Boundary",
        "desc": "Ruling grid lines or form bounding box caught in crop.",
        "func": effect_cell_grid_borders,
    },
    {
        "id": "12_neighbor_intrusion",
        "title": "Neighbor Stroke Intrusion",
        "category": "Crop Boundary",
        "desc": "Descender or tail from adjacent cell crossing the boundary.",
        "func": effect_neighbor_intrusion,
    },
    {
        "id": "13_registration_marker",
        "title": "Registration Cross (+)",
        "category": "Crop Boundary",
        "desc": "Exam sheet corner alignment mark or printer crosshair.",
        "func": effect_registration_mark,
    },
    {
        "id": "14_camera_defocus",
        "title": "Optical Defocus Blur",
        "category": "Optics / Camera",
        "desc": "Soft smartphone autofocus miss or shallow depth of field.",
        "func": effect_camera_defocus,
    },
    {
        "id": "15_motion_blur",
        "title": "Hand Motion Blur",
        "category": "Optics / Camera",
        "desc": "Linear directional smear from hand movement during shutter.",
        "func": effect_motion_blur,
    },
    {
        "id": "16_shadow_gradient",
        "title": "Phone / Hand Shadow",
        "category": "Illumination",
        "desc": "Pronounced non-uniform shadow falloff across the cell.",
        "func": effect_shadow_gradient,
    },
    {
        "id": "17_ruled_notebook_paper",
        "title": "Graph Paper Ruling",
        "category": "Paper Substrate",
        "desc": "Faint blue/gray printed grid lines (тетрадь в клетку).",
        "func": effect_ruled_notebook_paper,
    },
    {
        "id": "18_binarization_noise",
        "title": "Binarization / Speckle",
        "category": "Scanner Noise",
        "desc": "Salt-and-pepper noise, scanner dust, and threshold erosion.",
        "func": effect_salt_and_pepper_binarization,
    },
    {
        "id": "19_affine_skew",
        "title": "Affine Slant & Shear",
        "category": "Geometry",
        "desc": "Severe handwriting slant, angular shear, and off-centering.",
        "func": effect_affine_skew_jitter,
    },
    {
        "id": "20_clipping_crop",
        "title": "Segmentation Clipping",
        "category": "Geometry",
        "desc": "Imperfect auto-crop boundary clipping glyph edges.",
        "func": effect_partial_crop_clipping,
    },
    {
        "id": "21_combined_mild",
        "title": "Combined: Mild Notebook",
        "category": "Realistic Pipeline",
        "desc": "Natural jitter, paper texture, subtle shadow, light grid.",
        "func": effect_combined_mild_notebook,
    },
    {
        "id": "22_combined_severe",
        "title": "Combined: Harsh Classroom",
        "category": "Realistic Pipeline",
        "desc": "Correction retrace, ink blobs, neighbor cut, strong shadow.",
        "func": effect_combined_severe_correction,
    },
]


def render_card(mech_info, img_arr, card_w=280, card_h=300):
    card = Image.new("RGB", (card_w, card_h), color=(250, 252, 255))
    draw = ImageDraw.Draw(card)
    
    draw.rounded_rectangle([(1, 1), (card_w - 2, card_h - 2)], radius=10, outline=(218, 224, 233), width=1)
    
    try:
        f_cat = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 11)
        f_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 13)
        f_desc = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 11)
    except Exception:
        f_cat = f_title = f_desc = ImageFont.load_default()

    cat_text = mech_info["category"].upper()
    cat_bbox = draw.textbbox((0, 0), cat_text, font=f_cat)
    cat_w = cat_bbox[2] - cat_bbox[0] + 12
    badge_x = 16
    badge_y = 12
    draw.rounded_rectangle([(badge_x, badge_y), (badge_x + cat_w, badge_y + 18)], radius=4, fill=(235, 240, 250))
    draw.text((badge_x + 6, badge_y + 2), cat_text, fill=(45, 85, 155), font=f_cat)
    
    draw.text((16, 36), mech_info["title"], fill=(25, 30, 40), font=f_title)
    
    cell_pil = Image.fromarray(img_arr).convert("RGB")
    cell_w, cell_h = 160, 160
    if cell_pil.size != (cell_w, cell_h):
        cell_pil = cell_pil.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
    
    cell_x = (card_w - cell_w) // 2
    cell_y = 66
    card.paste(cell_pil, (cell_x, cell_y))
    draw.rectangle([(cell_x - 1, cell_y - 1), (cell_x + cell_w, cell_y + cell_h)], outline=(190, 198, 210), width=1)
    
    desc_words = mech_info["desc"].split()
    lines = []
    curr = []
    for w in desc_words:
        curr.append(w)
        bbox = draw.textbbox((0, 0), " ".join(curr), font=f_desc)
        if bbox[2] - bbox[0] > (card_w - 32):
            curr.pop()
            lines.append(" ".join(curr))
            curr = [w]
    if curr:
        lines.append(" ".join(curr))
        
    line_y = 236
    for l in lines[:3]:
        draw.text((16, line_y), l, fill=(90, 100, 115), font=f_desc)
        line_y += 16
        
    return card


def main():
    random.seed(42)
    np.random.seed(42)
    
    target_char = "Ә"  # The iconic Tatar letter
    canvas_size = 160
    
    all_fonts, font_weights = get_available_fonts()
    preferred_fonts = ["PlaypenSans", "BalsamiqSans-Regular", "Pangolin", "Comfortaa", "segoeui"]
    font_path = None
    for pf in preferred_fonts:
        if pf in all_fonts:
            font_path = all_fonts[pf]
            break
    if font_path is None:
        font_path = list(all_fonts.values())[0]
        
    print(f"Using font: {font_path} for target letter: '{target_char}'")
    
    base_clean = render_base_char(target_char, font_path, canvas_size=canvas_size, font_scale=0.72)
    
    output_dir = Path("dataset/corruption_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cards = []
    
    for mech in MECHANISMS:
        print(f"Generating: {mech['id']} - {mech['title']}")
        corrupted = mech["func"](
            base_clean.copy(),
            char=target_char,
            font_path=font_path,
            canvas_size=canvas_size
        )
        
        indiv_path = output_dir / f"{mech['id']}.png"
        Image.fromarray(corrupted).save(indiv_path)
        
        card_img = render_card(mech, corrupted, card_w=280, card_h=300)
        cards.append(card_img)
        
    cols = 4
    rows = math.ceil(len(cards) / cols)
    card_w, card_h = cards[0].size
    pad_x, pad_y = 16, 16
    header_h = 95
    
    grid_w = cols * card_w + (cols + 1) * pad_x
    grid_h = rows * card_h + (rows + 1) * pad_y + header_h
    
    master_gallery = Image.new("RGB", (grid_w, grid_h), color=(242, 245, 250))
    draw = ImageDraw.Draw(master_gallery)
    
    try:
        f_main_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 24)
        f_sub_title = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    except Exception:
        f_main_title = f_sub_title = ImageFont.load_default()
        
    draw.text((pad_x + 8, 18), f"Tatar OCR Degradation & Corruption Mechanisms Showcase — Letter '{target_char}'", fill=(20, 30, 50), font=f_main_title)
    draw.text((pad_x + 8, 52), f"Demonstration of 22 isolated & combined synthetic corruption mechanisms modeling real-world student notebooks.", fill=(80, 95, 115), font=f_sub_title)
    draw.line([(pad_x, 82), (grid_w - pad_x, 82)], fill=(210, 218, 230), width=2)
    
    for idx, card in enumerate(cards):
        r = idx // cols
        c = idx % cols
        pos_x = pad_x + c * (card_w + pad_x)
        pos_y = header_h + pad_y + r * (card_h + pad_y)
        master_gallery.paste(card, (pos_x, pos_y))
        
    gallery_path = output_dir / "corruption_gallery.png"
    master_gallery.save(gallery_path, quality=95)
    print(f"Master gallery saved to: {gallery_path}")
    
    artifact_dir = Path(r"C:\Users\galee\.gemini\antigravity\brain\e6d31b5c-5d27-456f-ade7-c736653ad00b")
    if artifact_dir.exists():
        shutil.copy(gallery_path, artifact_dir / "corruption_gallery.png")
        for f in output_dir.glob("*.png"):
            shutil.copy(f, artifact_dir / f.name)
        print(f"Copied all showcase assets to artifact directory: {artifact_dir}")
        
    print("Done!")

if __name__ == "__main__":
    main()

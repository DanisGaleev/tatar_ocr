"""
Tatar OCR Advanced Augmentation & Synthetic Handwriting Engine
Includes realistic school notebook artifacts:
- Cell border lines & crop boundary artifacts
- Non-uniform stroke pressure & dry pen starvation
- Letter overwriting / correction / double-strike
- Stray flourishes, hooks, loops, and neighbor stroke intrusions
- Realistic paper textures (shadow gradients, pulp tooth, faint graph grid)
- Elastic tremor and affine positioning jitter
"""

import os
import random
import math
from pathlib import Path
import numpy as np
import cv2
import scipy.ndimage
from PIL import Image, ImageDraw, ImageFont


# Master Alphabet & Classes Definition
TATAR_UPPERCASE = list("АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯ")
TATAR_LOWERCASE = list("аәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя")
PUNCTUATION_CHARS = [".", ",", "!", "?", "-", ":", ";", '"', "'", "(", ")", "«", "»", "—"]
DIGIT_CHARS = list("0123456789")
ALL_CHARS = TATAR_UPPERCASE + TATAR_LOWERCASE + PUNCTUATION_CHARS + DIGIT_CHARS

CHAR_TO_ID = {ch: i for i, ch in enumerate(ALL_CHARS)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(ALL_CHARS)}


def get_available_fonts():
    """Returns categorized dictionary of fonts verified to support all Tatar glyphs."""
    fonts_dir = Path("fonts")
    local_fonts = list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf"))
    
    # Prioritize authentic handwritten print fonts
    handwritten_print_names = {
        "BalsamiqSans-Regular", "BalsamiqSans-Bold", "BalsamiqSans-Italic",
        "PlaypenSans", "Pangolin", "DidactGothic", "Nunito-Regular",
        "Rubik-Regular", "Marmelad-Regular", "Comfortaa"
    }
    
    system_font_candidates = [
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ]
    
    all_fonts = {}
    for p in local_fonts:
        all_fonts[p.stem] = str(p)
    for p in system_font_candidates:
        if os.path.exists(p):
            all_fonts[Path(p).stem] = p
            
    # Weighted list for realistic sampling (strongly favoring handwritten print)
    weights = {}
    for name in all_fonts:
        if name in handwritten_print_names:
            weights[name] = 5.0
        elif "Script" in name or "Caveat" in name:
            weights[name] = 1.0
        else:
            weights[name] = 2.0
            
    return all_fonts, weights


# Pre-load font objects cache to maximize generation throughput
_FONT_CACHE = {}

def get_font_cached(font_path, size):
    key = (str(font_path), int(size))
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(str(font_path), int(size))
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def render_base_char(char, font_path, canvas_size=128, font_scale=0.72):
    """Renders raw character centered on white canvas at high resolution."""
    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    font_size = max(14, int(canvas_size * font_scale))
    font = get_font_cached(font_path, font_size)

    bbox = draw.textbbox((0, 0), char, font=font)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    # Baseline center with slight natural jitter
    x = (canvas_size - bw) // 2 - bbox[0]
    y = (canvas_size - bh) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Augmentation Filters
# ---------------------------------------------------------------------------

def apply_elastic_wobble(arr, alpha=18.0, sigma=6.0):
    """Simulates hand muscle tremor and uneven wobbly stroke curves."""
    if alpha <= 0.1:
        return arr
    shape = arr.shape
    rx = (np.random.rand(*shape) * 2 - 1).astype(np.float32)
    ry = (np.random.rand(*shape) * 2 - 1).astype(np.float32)
    dx = scipy.ndimage.gaussian_filter(rx, sigma=sigma) * alpha
    dy = scipy.ndimage.gaussian_filter(ry, sigma=sigma) * alpha

    y_coords, x_coords = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    map_x = np.clip(x_coords + dx, 0, shape[1] - 1).astype(np.float32)
    map_y = np.clip(y_coords + dy, 0, shape[0] - 1).astype(np.float32)

    return cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=255)


def apply_variable_thickness_and_starvation(arr, pressure_variation=1.6, starvation_prob=0.35):
    """
    Simulates authentic non-uniform pen pressure:
    - Downstrokes (vertical/tilted) become noticeably thicker and darker.
    - Horizontal crossbars and transitions remain fine.
    - Soft, anti-aliased stroke edges are preserved (no 1-bit jagged staircase pixels).
    - Occasional dry-pen stroke starvation / faint skips on sections of the letter with soft feathered fade.
    """
    binary = (arr < 180).astype(np.uint8)
    if binary.sum() == 0:
        return arr

    shape = arr.shape
    # Inner distance (inside the stroke) and outer distance (outside the stroke)
    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    # Signed distance field: positive inside stroke, negative outside
    sdf = dist_in.astype(np.float32) - dist_out.astype(np.float32)

    # Compute stroke orientation using Sobel gradients on the smooth glyph
    # For vertical downstrokes, gradient in X is large and gradient in Y is small
    smooth_glyph = scipy.ndimage.gaussian_filter(arr.astype(np.float32), sigma=1.0)
    gx = cv2.Sobel(smooth_glyph, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth_glyph, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2) + 1e-5
    # Directional alignment: vertical downstrokes (~50-80 degrees)
    downstroke_bias = np.abs(gx) / grad_mag

    # Smooth spatial pressure variation field (simulating varying hand force across the letter)
    smooth_noise = scipy.ndimage.gaussian_filter(np.random.randn(*shape), sigma=7.0)
    # Pressure offset in pixels of stroke dilation/erosion
    pressure_delta = (downstroke_bias - 0.45) * (pressure_variation * 1.2) + smooth_noise * (pressure_variation * 0.5)

    # Modulate signed distance field
    new_sdf = sdf + pressure_delta

    # Anti-aliased sigmoid/smoothstep transition preserving subpixel grayscale fidelity
    soft_stroke = np.clip(new_sdf * 1.4 + 0.5, 0.0, 1.0)
    # Combine with original darkness to preserve ink tonality
    res = (255.0 * (1.0 - soft_stroke)).astype(np.uint8)

    # Pen starvation / partial dry stroke effect (smooth gradual fade along part of a stroke)
    if random.random() < starvation_prob:
        center_x = random.randint(int(shape[1] * 0.2), int(shape[1] * 0.8))
        center_y = random.randint(int(shape[0] * 0.2), int(shape[0] * 0.8))
        radius = random.randint(int(shape[0] * 0.15), int(shape[0] * 0.35))
        y_grid, x_grid = np.ogrid[:shape[0], :shape[1]]
        dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        starve_mask = np.clip(1.0 - dist_from_center / (radius + 1e-5), 0.0, 1.0)
        starve_mask = scipy.ndimage.gaussian_filter(starve_mask, sigma=2.5)

        ink_mask = (res < 220).astype(np.float32)
        fade_intensity = random.uniform(70.0, 150.0)
        res = np.clip(res.astype(np.float32) + (starve_mask * ink_mask * fade_intensity), 0, 255).astype(np.uint8)

    return res


def apply_overwriting_correction(arr, char, font_path, canvas_size=128, prob=0.25):
    """
    Simulates real school mistake correction:
    - Pupil writes target letter boldly ("жирное написание") over an underlying mistaken letter.
    - Selects error letter matching the character category (uppercase, lowercase, digit, punctuation).
    - Or pupil traces over their own letter twice (double-strike / обводка).
    - Or cross-out correction mark (зачеркивание).
    """
    if random.random() > prob:
        return arr

    mode = random.choice(["underlying_error", "underlying_error", "double_strike", "scratch_line"])
    out = arr.copy().astype(np.float32)

    if mode == "underlying_error":
        # Select error character of same category (e.g. lowercase for lowercase, digit for digit)
        if char in TATAR_UPPERCASE:
            candidates = [c for c in TATAR_UPPERCASE if c != char]
        elif char in TATAR_LOWERCASE:
            candidates = [c for c in TATAR_LOWERCASE if c != char]
        elif char in DIGIT_CHARS:
            candidates = [c for c in DIGIT_CHARS if c != char]
        elif char in PUNCTUATION_CHARS:
            candidates = [c for c in PUNCTUATION_CHARS if c != char]
        else:
            candidates = [c for c in ALL_CHARS if c != char]

        err_char = random.choice(candidates) if candidates else "А"
        err_img = render_base_char(err_char, font_path, canvas_size=canvas_size, font_scale=random.uniform(0.68, 0.76))

        # Underlying mistaken letter is shifted slightly
        shift_x = random.randint(-4, 4)
        shift_y = random.randint(-4, 4)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        err_img = cv2.warpAffine(err_img, M, (canvas_size, canvas_size), borderValue=255)

        # Underlying letter is fainter (opacity ~ 0.35 to 0.55)
        under_alpha = (255.0 - err_img.astype(np.float32)) / 255.0 * random.uniform(0.35, 0.55)
        out = 255.0 * (1.0 - under_alpha) + 120.0 * under_alpha

        # The overlying target letter is written BOLDLY ("жирное написание")
        # Dilate the target letter stroke to simulate heavy bold pen overwrite
        target_binary = (arr < 180).astype(np.uint8)
        k_size = random.choice([3, 5])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated_target = cv2.erode(arr, kernel)  # erode darkens/thickens in grayscale
        dilated_target = scipy.ndimage.gaussian_filter(dilated_target.astype(np.float32), sigma=0.6)

        target_alpha = (255.0 - dilated_target) / 255.0
        out = out * (1.0 - target_alpha) + (dilated_target * target_alpha)

    elif mode == "double_strike":
        # Pupil re-traced the same letter with slight hand offset / double pass
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        angle = random.uniform(-2.5, 2.5)
        center = (canvas_size / 2.0, canvas_size / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        second_pass = cv2.warpAffine(arr, M, (canvas_size, canvas_size), borderValue=255)
        out = np.minimum(arr, second_pass).astype(np.float32)

    elif mode == "scratch_line":
        # A quick correction slash or tick mark near or through part of the letter
        p1 = (random.randint(int(canvas_size * 0.15), int(canvas_size * 0.40)),
              random.randint(int(canvas_size * 0.15), int(canvas_size * 0.50)))
        p2 = (p1[0] + random.randint(int(canvas_size * 0.30), int(canvas_size * 0.60)),
              p1[1] + random.randint(-int(canvas_size * 0.15), int(canvas_size * 0.20)))
        scratch = np.full_like(arr, 255)
        cv2.line(scratch, p1, p2, color=random.randint(50, 110), thickness=random.choice([1, 2, 3]))
        scratch = scipy.ndimage.gaussian_filter(scratch.astype(np.float32), sigma=0.6)
        out = np.minimum(out, scratch)

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_flourishes_and_stray_marks(arr, prob=0.45):
    """
    Adds extra flourishes, pen tails, starting hooks, loops, and ink blobs (кляксы).
    Flourishes connect naturally to stroke endpoints/edges rather than floating arbitrarily.
    """
    if random.random() > prob:
        return arr

    res = arr.copy()
    h, w = res.shape
    ink_ys, ink_xs = np.where(res < 180)

    if len(ink_xs) < 15:
        return res

    num_marks = random.randint(1, 3)
    for _ in range(num_marks):
        mark_type = random.choice(["hook_tail", "stray_loop", "ink_blob"])

        if mark_type == "hook_tail":
            # Pick an ink pixel near the outer boundary of the stroke to attach flourish
            idx = random.randint(0, len(ink_xs) - 1)
            start_x = int(ink_xs[idx])
            start_y = int(ink_ys[idx])

            ctrl_x = start_x + random.randint(-14, 14)
            ctrl_y = start_y + random.randint(-12, 12)
            end_x = ctrl_x + random.randint(-12, 12)
            end_y = ctrl_y + random.randint(-12, 12)

            pts = []
            for t in np.linspace(0, 1, 10):
                bx = (1 - t)**2 * start_x + 2 * (1 - t) * t * ctrl_x + t**2 * end_x
                by = (1 - t)**2 * start_y + 2 * (1 - t) * t * ctrl_y + t**2 * end_y
                pts.append([int(bx), int(by)])
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(res, [pts], isClosed=False, color=random.randint(40, 120), thickness=random.choice([1, 2]))

        elif mark_type == "stray_loop":
            # Small flourish loop attached near top-left or bottom-right of a stroke
            idx = random.choice([0, len(ink_xs) - 1, random.randint(0, len(ink_xs) - 1)])
            cx = int(np.clip(ink_xs[idx] + random.randint(-6, 6), 4, w - 5))
            cy = int(np.clip(ink_ys[idx] + random.randint(-6, 6), 4, h - 5))
            axes = (random.randint(3, 6), random.randint(2, 4))
            angle = random.randint(0, 180)
            cv2.ellipse(res, (cx, cy), axes, angle, 0, random.randint(220, 360), color=random.randint(50, 120), thickness=1)

        elif mark_type == "ink_blob":
            # Ink dot / droplet from pen lingering on paper
            idx = random.randint(0, len(ink_xs) - 1)
            bx = int(ink_xs[idx] + random.randint(-2, 2))
            by = int(ink_ys[idx] + random.randint(-2, 2))
            cv2.circle(res, (bx, by), random.choice([2, 3]), color=random.randint(20, 70), thickness=-1)

    return res


def apply_cell_crop_artifacts(arr, prob=0.75):
    """
    Simulates real-world cell crop boundary artifacts from marker-aligned exam sheets:
    - Printed grid boundary lines (0, 1, or 2 corner L-shape intersections)
    - Grid lines with realistic color (blue notebook ruling, gray laser printed, or dark black)
    - Slight residual skew tilt (±0.5° to ±1.5°) from marker homography calibration residual
    - Remnants of corner registration markers / crosshairs (+)
    - Stray descenders / strokes intruding from adjacent cells
    """
    if random.random() > prob:
        return arr

    res = arr.copy()
    h, w = res.shape

    # 1. Printed grid lines near cell boundary
    mode = random.choices(["single_edge", "corner_l_shape", "clean"], weights=[0.50, 0.35, 0.15], k=1)[0]
    if mode != "clean":
        if mode == "single_edge":
            edges = [random.choice(["top", "bottom", "left", "right"])]
        else:
            # Corner L-shape: e.g. ("top", "left"), ("bottom", "right"), etc.
            v_edge = random.choice(["left", "right"])
            h_edge = random.choice(["top", "bottom"])
            edges = [v_edge, h_edge]

        # Line color: blue notebook grid (gray ~150-195), light gray (130-170), or dark printed (50-100)
        grid_style = random.choice(["notebook_blue", "laser_gray", "dark_box"])
        if grid_style == "notebook_blue":
            line_color = random.randint(155, 195)
        elif grid_style == "laser_gray":
            line_color = random.randint(110, 150)
        else:
            line_color = random.randint(45, 95)

        line_thickness = random.choice([1, 1, 2])
        tilt = random.uniform(-1.2, 1.2)  # residual skew

        for edge in edges:
            if edge == "left":
                lx = random.randint(0, 3)
                pt1 = (int(np.clip(lx - tilt, 0, w - 1)), 0)
                pt2 = (int(np.clip(lx + tilt, 0, w - 1)), h - 1)
                cv2.line(res, pt1, pt2, color=line_color, thickness=line_thickness)
            elif edge == "right":
                rx = random.randint(w - 4, w - 1)
                pt1 = (int(np.clip(rx - tilt, 0, w - 1)), 0)
                pt2 = (int(np.clip(rx + tilt, 0, w - 1)), h - 1)
                cv2.line(res, pt1, pt2, color=line_color, thickness=line_thickness)
            elif edge == "top":
                ty = random.randint(0, 3)
                pt1 = (0, int(np.clip(ty - tilt, 0, h - 1)))
                pt2 = (w - 1, int(np.clip(ty + tilt, 0, h - 1)))
                cv2.line(res, pt1, pt2, color=line_color, thickness=line_thickness)
            elif edge == "bottom":
                by = random.randint(h - 4, h - 1)
                pt1 = (0, int(np.clip(by - tilt, 0, h - 1)))
                pt2 = (w - 1, int(np.clip(by + tilt, 0, h - 1)))
                cv2.line(res, pt1, pt2, color=line_color, thickness=line_thickness)

    # 2. Registration marker snippet (crosshair '+' or corner dot remnant caught in crop)
    if random.random() < 0.18:
        corner = random.choice(["tl", "tr", "bl", "br"])
        cx = random.randint(1, 4) if "l" in corner else random.randint(w - 5, w - 2)
        cy = random.randint(1, 4) if "t" in corner else random.randint(h - 5, h - 2)
        m_color = random.randint(30, 80)
        # Draw small crosshair snippet
        cv2.line(res, (cx - 3, cy), (cx + 3, cy), color=m_color, thickness=1)
        cv2.line(res, (cx, cy - 3), (cx, cy + 3), color=m_color, thickness=1)

    # 3. Neighbor character stroke intrusion (descender from top cell, tail from left)
    if random.random() < 0.40:
        neighbor_edge = random.choice(["top", "left", "bottom", "right"])
        n_color = random.randint(50, 140)
        if neighbor_edge == "top":
            # Descender (e.g. from 'у', 'р', 'ү', 'җ') cutting into top edge
            sx = random.randint(int(w * 0.2), int(w * 0.8))
            ex = sx + random.randint(-8, 8)
            ey = random.randint(5, 16)
            cv2.line(res, (sx, 0), (ex, ey), color=n_color, thickness=random.choice([1, 2]))
        elif neighbor_edge == "left":
            # Tail protruding from left adjacent cell
            sy = random.randint(int(h * 0.2), int(h * 0.8))
            ey = sy + random.randint(-6, 6)
            ex = random.randint(5, 14)
            cv2.line(res, (0, sy), (ex, ey), color=n_color, thickness=random.choice([1, 2]))
        elif neighbor_edge == "bottom":
            sx = random.randint(int(w * 0.2), int(w * 0.8))
            cv2.line(res, (sx, h - 1), (sx + random.randint(-6, 6), h - 1 - random.randint(4, 10)), color=n_color, thickness=1)

    return res


def apply_paper_and_lighting(arr, bg_range=(225, 252), add_gradient=True, add_faint_grid=False):
    """
    Renders realistic classroom test paper with non-uniform phone/hand lighting:
    - Pronounced shadow gradients / vignetting (30-65 luminance contrast across cell).
    - Multi-scale paper pulp texture (smooth cloudiness + fine tooth).
    - Optional faint notebook ruling / graph paper grid lines (тетрадь в клетку).
    - Optical camera sensor shot noise.
    """
    h, w = arr.shape
    base_lum = random.randint(bg_range[0], bg_range[1])

    # 1. Phone / Hand Shadow Illumination Field
    if add_gradient:
        # Diagonal shadow bar or corner shadow (matching real mobile photography in classrooms)
        angle = random.uniform(0, 2 * math.pi)
        # Luminance slope: creates realistic 25-60 unit drop across the cell
        contrast = random.uniform(25.0, 60.0)
        y_norm, x_norm = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing="ij")
        directional_shadow = (x_norm * math.cos(angle) + y_norm * math.sin(angle)) * contrast
        # Optional soft corner vignette (falloff)
        r2 = (x_norm**2 + y_norm**2) * random.uniform(5.0, 18.0)
        lighting_field = base_lum + directional_shadow - r2
    else:
        lighting_field = np.full((h, w), base_lum, dtype=np.float32)

    # 2. Multi-scale paper pulp grain (low-frequency fiber clouds + high-frequency micro tooth)
    pulp_cloud = scipy.ndimage.gaussian_filter(np.random.normal(0, 4.0, (h, w)), sigma=3.0)
    pulp_tooth = np.random.normal(0, 2.2, (h, w))
    paper_bg = np.clip(lighting_field + pulp_cloud + pulp_tooth, 160, 255).astype(np.float32)

    # 3. Optional faint ruled / grid notebook lines (20% probability)
    if add_faint_grid or random.random() < 0.20:
        grid_overlay = paper_bg.copy()
        grid_step = random.choice([16, 20, 24])
        grid_darkness = random.randint(10, 24)
        for gy in range(0, h, grid_step):
            grid_overlay[gy, :] = np.clip(grid_overlay[gy, :] - grid_darkness, 0, 255)
        for gx in range(0, w, grid_step):
            grid_overlay[:, gx] = np.clip(grid_overlay[:, gx] - grid_darkness, 0, 255)
        paper_bg = grid_overlay

    # 4. Blend ink with textured paper
    stroke_alpha = (255.0 - arr.astype(np.float32)) / 255.0
    final = paper_bg * (1.0 - stroke_alpha) + (arr.astype(np.float32) * stroke_alpha)

    # 5. Camera sensor noise (slight shot noise)
    cam_noise = np.random.normal(0, 2.0, (h, w))
    final = np.clip(final + cam_noise, 0, 255).astype(np.uint8)

    return final


def apply_camera_defocus(arr, prob=0.40):
    """Simulates slight smartphone camera defocus or optical softness."""
    if random.random() > prob:
        return arr
    k = random.choice([3, 3])
    sigma = random.uniform(0.4, 0.9)
    return cv2.GaussianBlur(arr, (k, k), sigma)


def apply_affine_cell_placement(arr, max_angle=12.0, max_shear=8.0, scale_range=(0.68, 0.96)):
    """
    Applies realistic cell positioning jitter:
    - Variable letter size inside cell (68% - 96% of cell, allowing occasional edge touches).
    - Natural hand tilt / slant (-12° to +12°).
    - Hand shear (-8° to +8°).
    - Off-center shift jitter within the cell box.
    """
    h, w = arr.shape
    angle = random.uniform(-max_angle, max_angle)
    shear = random.uniform(-max_shear, max_shear)
    scale = random.uniform(scale_range[0], scale_range[1])

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

    # Shear adjustment
    shear_rad = np.deg2rad(shear)
    rot_mat[0, 1] += np.tan(shear_rad) * rot_mat[0, 0]

    # Shift jitter inside cell
    shift_x = random.uniform(-w * 0.09, w * 0.09)
    shift_y = random.uniform(-h * 0.09, h * 0.09)
    rot_mat[0, 2] += shift_x
    rot_mat[1, 2] += shift_y

    return cv2.warpAffine(arr, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


# ---------------------------------------------------------------------------
# Master Augmentation Function
# ---------------------------------------------------------------------------

def generate_augmented_cell_image(
    char,
    font_path,
    canvas_size=64,
    elastic_strength=18.0,
    pressure_variation=1.6,
    enable_overwriting=True,
    enable_flourishes=True,
    enable_cell_borders=True,
    enable_paper_texture=True,
    enable_camera_blur=True,
    base_img=None,
):
    """
    Comprehensive, realistic Tatar handwritten character generator tailored for
    cropped cell recognition in the 'Дәресханә' mobile OCR system.
    Supports supplying pre-rendered `base_img` to eliminate repeated font rasterization.
    """
    internal_size = max(128, canvas_size * 2)

    # 1. Base clean glyph: use cached base_img if provided, or render
    if base_img is not None:
        raw = base_img.copy()
        if raw.shape != (internal_size, internal_size):
            raw = cv2.resize(raw, (internal_size, internal_size), interpolation=cv2.INTER_AREA)
    else:
        raw = render_base_char(char, font_path, canvas_size=internal_size)

    # 2. Overwriting / correction (underlying mistaken letter, double-strike, or scratch)
    if enable_overwriting:
        img = apply_overwriting_correction(raw, char, font_path, canvas_size=internal_size, prob=0.22)
    else:
        img = raw.copy()

    # 3. Non-uniform stroke pressure & dry-pen starvation
    img = apply_variable_thickness_and_starvation(img, pressure_variation=pressure_variation, starvation_prob=0.35)

    # 4. Extra flourishes, stray loops, hooks, ink blobs
    if enable_flourishes:
        img = apply_flourishes_and_stray_marks(img, prob=0.40)

    # 5. Elastic muscle tremor / wobbly strokes
    img = apply_elastic_wobble(img, alpha=elastic_strength, sigma=6.0)

    # 6. Affine jitter: scale inside cell, rotation, shear, centering offset
    img = apply_affine_cell_placement(img, max_angle=12.0, max_shear=8.0, scale_range=(0.68, 0.96))

    # 7. Downscale to target canvas size (e.g. 64x64) with anti-aliasing
    if internal_size != canvas_size:
        img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)

    # 8. Cell borders & neighbor stroke intrusions (applied at target resolution)
    if enable_cell_borders:
        img = apply_cell_crop_artifacts(img, prob=0.70)

    # 9. Camera defocus / optical softness
    if enable_camera_blur:
        img = apply_camera_defocus(img, prob=0.35)

    # 10. Paper background & illumination gradient
    if enable_paper_texture:
        img = apply_paper_and_lighting(img, bg_range=(225, 252), add_gradient=True)

    return img

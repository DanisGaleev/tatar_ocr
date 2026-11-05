"""
Tatar OCR Advanced Augmentation & Synthetic Handwriting Engine (v2)
Includes realistic school exam sheet & distant mobile camera physics:
- Procedural handwritten allographs for confusion-prone letters (П/п, И/и, Ә/∂, Л, Н, Ң, М, У, Ү, Ч, В, Ь, Ц, Щ, Ж, Җ, Х, Һ)
- Pen & pencil stroke physics: downstroke pressure, vertex/intersection ink pooling, ballpoint skipping/striation, pencil graphite grain
- Structural handwriting variations: aspect-ratio squeeze/stretch, slant/shear, elastic muscle tremor, overwriting/double-strike
- Distant smartphone camera resolution bottleneck (16-38px per cell) + optical PSF blur + INTER_LANCZOS4 perspective rectification
- Exact BlankOCRScanner.classify_cell alignment: tight ink bounding-box crop, max_dim=48 scaling, p2..p98 -> 10..250 contrast normalization, centered on a 64x64 canvas with 250 background
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

TATAR_SPECIFIC_CHARS = set("ӘӨҮҖҢҺәөүҗңһ")


def get_available_fonts():
    """Returns categorized dictionary of fonts verified to support Tatar glyphs."""
    fonts_dir = Path("fonts")
    local_fonts = list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf"))

    handwritten_print_names = {
        "BalsamiqSans-Regular", "BalsamiqSans-Bold", "BalsamiqSans-Italic",
        "PlaypenSans", "Pangolin", "Caveat", "BadScript", "DidactGothic",
        "Nunito-Regular", "Rubik-Regular", "Marmelad-Regular", "Comfortaa"
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

    weights = {}
    for name in all_fonts:
        if name in {"PlaypenSans", "Pangolin", "BalsamiqSans-Regular", "BalsamiqSans-Italic", "Caveat", "BadScript"}:
            weights[name] = 6.0
        elif name in handwritten_print_names:
            weights[name] = 4.5
        else:
            weights[name] = 1.8

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


# Characters where students frequently write lowercase/cursive allographs inside uppercase cells
_LOWERCASE_ALLOGRAPH_PROB = {
    "П": 0.45,
    "И": 0.45,
    "Й": 0.35,
    "Л": 0.35,
    "М": 0.30,
    "Н": 0.25,
    "К": 0.30,
    "С": 0.30,
    "В": 0.25,
    "У": 0.35,
    "Х": 0.30,
    "Ц": 0.30,
    "Щ": 0.30,
    "Ш": 0.30,
    "Я": 0.30,
    "Э": 0.25,
}


def render_base_char(char, font_path, canvas_size=128, font_scale=0.72):
    """Renders raw character centered on white canvas at high resolution."""
    # Fallback font if a script font lacks special Tatar glyphs
    fpath_str = str(font_path)
    if char in TATAR_SPECIFIC_CHARS and ("BadScript" in fpath_str or "Caveat" in fpath_str):
        fallback = Path("fonts/PlaypenSans.ttf")
        if fallback.exists():
            fpath_str = str(fallback)

    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    font_size = max(14, int(canvas_size * font_scale))
    font = get_font_cached(fpath_str, font_size)

    bbox = draw.textbbox((0, 0), char, font=font)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    if bw <= 2 or bh <= 2:
        fallback = Path("fonts/Rubik-Regular.ttf")
        if fallback.exists():
            font = get_font_cached(str(fallback), font_size)
            bbox = draw.textbbox((0, 0), char, font=font)
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]

    x = (canvas_size - bw) // 2 - bbox[0]
    y = (canvas_size - bh) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Procedural Handwritten Allograph Generator (Confusion-Pair Anatomy)
# ---------------------------------------------------------------------------

def _quad_bezier(p0, p1, p2, num=24):
    t = np.linspace(0.0, 1.0, num)[:, None]
    p0, p1, p2 = np.array(p0, dtype=np.float32), np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32)
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return pts


def _cubic_bezier(p0, p1, p2, p3, num=32):
    t = np.linspace(0.0, 1.0, num)[:, None]
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    p3 = np.array(p3, dtype=np.float32)
    pts = (
        ((1 - t) ** 3) * p0
        + 3 * ((1 - t) ** 2) * t * p1
        + 3 * (1 - t) * (t ** 2) * p2
        + (t ** 3) * p3
    )
    return pts


def _draw_smooth_polyline(canvas, pts, thickness=5, color=0):
    pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_int], isClosed=False, color=int(color), thickness=max(1, int(round(thickness))), lineType=cv2.LINE_AA)


HANDWRITTEN_ALLOGRAPH_CHARS = {
    "П", "Һ", "И", "Л", "Н", "Ң", "М", "Ә", "У", "Ү", "Ч",
    "В", "Ь", "Ц", "Щ", "Ж", "Җ", "Х", "Е", "Э", "Я", "Р", "Ф",
    "А", "К", "Ю"
}


def render_handwritten_allograph(char, canvas_size=128):
    """
    Synthesizes authentic handwritten allographs observed in real student/teacher blanks:
    - 'П': cursive/block 'п' ('n'-arch with small top-left stem spur) or tilted/overhanging top bar
    - 'Һ': tall vertical ascender stem + strictly lower-half right arch ('h')
    - 'И': cursive rounded 'u'-bowl ('и') with straight right leg + optional tiny foot tick, or narrow diagonal 'И'
    - 'Л': sharp pointed-apex inverted-V tent ('/\\') with optional small left foot hook
    - 'Н': slanted crossbar, V-sagging/continuous-motion crossbar, or uneven stems
    - 'Ң': handwritten 'Н' (including slanted/continuous crossbar) + right descender hook/loop
    - 'М': outer legs meeting at top peaks + deep central V
    - 'Ә': handwritten cursive '∂' (lower belly loop + overarching top-left hook)
    - 'У', 'Ү', 'Ч': cursive rounded 'у'-cup vs symmetric 'Ү'-fork vs shallow top-cup 'Ч'
    - 'В', 'Ь', 'Р', 'Е': narrow/small top loop on 'В', optional left lead-in loop, open loop junctions
    - 'Ц', 'Щ': handwritten bottom bar + prominent right descender loop/hook + variable middle leg in 'Щ'
    - 'Ж', 'Җ', 'Х': crossed diagonals/chevrons + central vertical bar (+ descender tail on 'Җ')
    """
    s = float(canvas_size)
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    dense_chars = {"Ж", "Җ", "Щ", "Ң", "Ә", "В", "М", "Ф", "Я"}
    if char in dense_chars:
        thick = random.uniform(2.3, 4.3) * (s / 128.0)
    else:
        thick = random.uniform(2.6, 5.0) * (s / 128.0)

    cx = s * 0.5 + random.uniform(-3.0, 3.0)
    cy = s * 0.5 + random.uniform(-3.0, 3.0)
    w = s * random.uniform(0.30, 0.45)
    h = s * random.uniform(0.54, 0.68)

    x_l = cx - w * 0.5
    x_r = cx + w * 0.5
    y_t = cy - h * 0.5
    y_b = cy + h * 0.5

    if char == "П":
        if random.random() < 0.75:
            # Handwritten 'п' / Latin 'n' arch (matches 01 Q1[5], 11 Q3[1], 11 Q6[5], 12 Q2[6], 06 Q1[5])
            w_n = h * random.uniform(0.36, 0.60)
            x_l = cx - w_n * 0.5
            x_r = cx + w_n * 0.5
            slant = random.uniform(-3.0, 6.0)
            p_lt = (x_l + slant * 0.5, y_t + random.uniform(0, 4))
            p_lb = (x_l - slant * 0.5, y_b + random.uniform(-3, 3))
            _draw_smooth_polyline(canvas, np.array([p_lt, p_lb]), thick)

            # Arch branches near the top of left stem (spur is 4-24% of height, unlike 'Һ' which is 48-64%)
            spur_frac = random.uniform(0.04, 0.24)
            p_branch = (
                p_lt[0] * (1 - spur_frac) + p_lb[0] * spur_frac,
                p_lt[1] * (1 - spur_frac) + p_lb[1] * spur_frac,
            )
            p_peak = (cx + w_n * random.uniform(-0.05, 0.25), y_t + random.uniform(-2, 4))
            p_rt_mid = (x_r + slant * 0.3, y_t + h * random.uniform(0.15, 0.32))
            p_rb = (x_r - slant * 0.4 + random.uniform(-3, 3), y_b + random.uniform(-3, 4))
            arch1 = _quad_bezier(p_branch, (x_l + w_n * 0.28, y_t - 2), p_peak, 16)
            arch2 = _cubic_bezier(p_peak, (x_r + 2, y_t), p_rt_mid, p_rb, 24)
            _draw_smooth_polyline(canvas, np.vstack([arch1, arch2]), thick)
            if random.random() < 0.45:
                flick = _quad_bezier(p_rb, (p_rb[0] + 3, p_rb[1] + 2), (p_rb[0] + random.uniform(4, 8), p_rb[1] - random.uniform(1, 5)), 10)
                _draw_smooth_polyline(canvas, flick, thick * 0.85)
        else:
            # Two stems + tilted / sagging / overhanging top bar
            p_lt = (x_l + random.uniform(-3, 3), y_t + random.uniform(0, 5))
            p_lb = (x_l + random.uniform(-4, 4), y_b)
            p_rt = (x_r + random.uniform(-3, 3), y_t + random.uniform(-3, 5))
            p_rb = (x_r + random.uniform(-4, 4), y_b + random.uniform(-3, 3))
            _draw_smooth_polyline(canvas, np.array([p_lt, p_lb]), thick)
            _draw_smooth_polyline(canvas, np.array([p_rt, p_rb]), thick)
            over_l = random.uniform(-2, 7)
            over_r = random.uniform(-2, 7)
            sag = random.uniform(-4, 6)
            top_bar = _quad_bezier((p_lt[0] - over_l, p_lt[1]), (cx, min(p_lt[1], p_rt[1]) + sag), (p_rt[0] + over_r, p_rt[1]), 20)
            _draw_smooth_polyline(canvas, top_bar, thick * random.uniform(0.85, 1.05))

    elif char == "Һ":
        # Strict 'h' proportions: tall vertical stem + arch strictly in the lower 45-55%
        w_h = s * random.uniform(0.24, 0.36)
        x_l = cx - w_h * 0.5
        x_r = cx + w_h * 0.5
        slant = random.uniform(-4.0, 5.0)
        p_lt = (x_l + slant * 0.5, y_t - random.uniform(2, 6))
        p_lb = (x_l - slant * 0.5, y_b)
        _draw_smooth_polyline(canvas, np.array([p_lt, p_lb]), thick)

        branch_frac = random.uniform(0.50, 0.64)
        p_branch = (
            p_lt[0] * (1 - branch_frac) + p_lb[0] * branch_frac,
            p_lt[1] * (1 - branch_frac) + p_lb[1] * branch_frac,
        )
        y_arch_top = y_t + h * random.uniform(0.45, 0.55)
        p_rb = (x_r + random.uniform(-3, 4), y_b + random.uniform(-2, 4))
        arch = _cubic_bezier(
            p_branch,
            (x_l + w_h * 0.35, y_arch_top - 4),
            (x_r + 2, y_arch_top),
            p_rb,
            28,
        )
        _draw_smooth_polyline(canvas, arch, thick * 0.95)
        if random.random() < 0.40:
            flick = _quad_bezier(p_rb, (p_rb[0] + 3, p_rb[1] + 1), (p_rb[0] + random.uniform(4, 7), p_rb[1] - random.uniform(1, 4)), 8)
            _draw_smooth_polyline(canvas, flick, thick * 0.85)

    elif char == "И":
        if random.random() < 0.75:
            # Cursive 'u'-bowl 'и' (matches 01 Q2[7], 06 Q1[2], 01 Q1[2], 02 Q4[8])
            w_u = h * random.uniform(0.38, 0.62)
            x_l = cx - w_u * 0.5
            x_r = cx + w_u * 0.5
            y_bowl = y_t + h * random.uniform(0.68, 0.95)
            p_lt = (x_l + random.uniform(-2, 4), y_t + random.uniform(0, 10))
            p_lmid = (x_l + random.uniform(-2, 3), y_t + h * 0.52)
            p_bot = (cx - w_u * random.uniform(0.02, 0.18), y_bowl)
            p_rt = (x_r + random.uniform(-2, 4), y_t + random.uniform(-2, 5))
            p_rb = (x_r + random.uniform(-3, 4), y_b + random.uniform(-2, 5))

            bowl1 = _quad_bezier(p_lt, p_lmid, p_bot, 18)
            bowl2 = _quad_bezier(p_bot, (x_r - w_u * 0.15, y_bowl - 2), p_rt, 18)
            right_leg = _quad_bezier(p_rt, (x_r, (y_t + y_b) * 0.5), p_rb, 18)
            _draw_smooth_polyline(canvas, np.vstack([bowl1, bowl2]), thick)
            _draw_smooth_polyline(canvas, right_leg, thick)
            if random.random() < 0.45:
                tick = _quad_bezier(p_rb, (p_rb[0] + 3, p_rb[1] + 2), (p_rb[0] + random.uniform(4, 9), p_rb[1] - random.uniform(1, 5)), 10)
                _draw_smooth_polyline(canvas, tick, thick * 0.85)
        else:
            w_i = s * random.uniform(0.26, 0.40)
            x_l = cx - w_i * 0.5
            x_r = cx + w_i * 0.5
            p_lt = (x_l + random.uniform(-3, 3), y_t)
            p_lb = (x_l + random.uniform(-3, 3), y_b)
            p_rt = (x_r + random.uniform(-3, 3), y_t)
            p_rb = (x_r + random.uniform(-3, 3), y_b)
            _draw_smooth_polyline(canvas, np.array([p_lt, p_lb]), thick)
            _draw_smooth_polyline(canvas, np.array([p_rt, p_rb]), thick)
            diag_start = (p_lb[0], y_b - h * random.uniform(0.02, 0.18))
            diag_end = (p_rt[0], y_t + h * random.uniform(0.02, 0.18))
            diag = _quad_bezier(diag_start, (cx + random.uniform(-4, 4), cy + random.uniform(-4, 4)), diag_end, 20)
            _draw_smooth_polyline(canvas, diag, thick * 0.9)

    elif char == "Л":
        # Pointed-apex inverted-V tent '/\' (matches 01 Q4[6], 03 Q1[3], 03 Q4[8], 01 Q3[5], 02 Q3[2])
        w_l = s * random.uniform(0.26, 0.42)
        x_l = cx - w_l * 0.5
        x_r = cx + w_l * 0.5
        p_apex = (cx + random.uniform(-4, 4), y_t + random.uniform(-3, 3))
        p_lb = (x_l + random.uniform(-3, 3), y_b + random.uniform(-2, 3))
        p_rb = (x_r + random.uniform(-3, 3), y_b + random.uniform(-2, 3))
        left_leg = _quad_bezier(p_lb, ((p_lb[0] + p_apex[0]) * 0.5 + random.uniform(-4, 3), cy + random.uniform(-3, 5)), p_apex, 22)
        right_leg = _quad_bezier(p_apex, ((p_rb[0] + p_apex[0]) * 0.5 + random.uniform(-3, 3), cy + random.uniform(-3, 3)), p_rb, 22)
        _draw_smooth_polyline(canvas, left_leg, thick)
        _draw_smooth_polyline(canvas, right_leg, thick)
        if random.random() < 0.40:
            hook = _quad_bezier(p_lb, (p_lb[0] - 4, p_lb[1] + 1), (p_lb[0] - random.uniform(4, 8), p_lb[1] - random.uniform(2, 6)), 8)
            _draw_smooth_polyline(canvas, hook, thick * 0.85)

    elif char in ("Н", "Ң"):
        is_ng = (char == "Ң")
        w_n = s * random.uniform(0.28, 0.40)
        x_l = cx - w_n * 0.5
        x_r = cx + w_n * 0.5
        y_base = (y_t + h * random.uniform(0.74, 0.82)) if is_ng else y_b

        p_lt = (x_l + random.uniform(-3, 3), y_t + random.uniform(-2, 4))
        p_lb = (x_l + random.uniform(-3, 3), y_base + random.uniform(-2, 3))
        p_rt = (x_r + random.uniform(-3, 3), y_t + random.uniform(-2, 4))
        p_rb = (x_r + random.uniform(-3, 3), y_base + random.uniform(-2, 3))

        # Both vertical stems are always clearly drawn from top to base
        _draw_smooth_polyline(canvas, np.array([p_lt, p_lb]), thick)
        _draw_smooth_polyline(canvas, np.array([p_rt, p_rb]), thick)

        style = random.choices(["slanted", "continuous_v", "standard"], weights=[0.45, 0.40, 0.15], k=1)[0]
        h_body = y_base - y_t
        if style == "slanted":
            # Matches 01 Q4[5], 05 Q1[7]: strongly tilted diagonal crossbar
            y_c1 = y_t + h_body * random.uniform(0.46, 0.68)
            y_c2 = y_t + h_body * random.uniform(0.26, 0.46)
            if random.random() < 0.35:
                y_c1, y_c2 = y_c2, y_c1
            cbar = _quad_bezier((x_l, y_c1), (cx, (y_c1 + y_c2) * 0.5 + random.uniform(-3, 3)), (x_r, y_c2), 18)
            _draw_smooth_polyline(canvas, cbar, thick * 0.90)
        elif style == "continuous_v":
            # Matches 02 Q3[6], 08 Q2[1], 12 Q4[6]: crossbar starts near upper-left, dips to middle (36-50%), rises toward upper-right
            y_start = y_t + h_body * random.uniform(0.10, 0.30)
            y_dip = y_t + h_body * random.uniform(0.36, 0.50)
            y_end = y_t + h_body * random.uniform(0.08, 0.26)
            c1 = _quad_bezier((x_l, y_start), ((x_l + cx) * 0.5, (y_start + y_dip) * 0.5), (cx + random.uniform(-3, 3), y_dip), 14)
            c2 = _quad_bezier((cx + random.uniform(-3, 3), y_dip), ((cx + x_r) * 0.5, (y_dip + y_end) * 0.5), (x_r, y_end), 14)
            _draw_smooth_polyline(canvas, np.vstack([c1, c2]), thick * 0.88)
        else:
            y_c = y_t + h_body * random.uniform(0.42, 0.56)
            cbar = _quad_bezier((x_l, y_c), (cx, y_c + random.uniform(-4, 4)), (x_r, y_c + random.uniform(-3, 3)), 16)
            _draw_smooth_polyline(canvas, cbar, thick * 0.90)

        if is_ng:
            # Authentic Tatar Ң right descender: small step-out to the right at base, then vertical drop (matches 01 Q3[8], 02 Q2[7], 12 Q4[6])
            step_w = s * random.uniform(0.06, 0.11)
            tail_y = y_b + s * random.uniform(0.04, 0.10)
            p_step = (p_rb[0] + step_w, p_rb[1] - random.uniform(0, 5))
            p_tip = (p_step[0] + random.uniform(-3, 3), tail_y)
            tail1 = _quad_bezier((p_rb[0], p_rb[1] - 2), (p_rb[0] + step_w * 0.5, p_rb[1] - 3), p_step, 10)
            tail2 = _quad_bezier(p_step, (p_step[0], (p_step[1] + tail_y) * 0.5), p_tip, 14)
            _draw_smooth_polyline(canvas, np.vstack([tail1, tail2]), thick * 0.90)

    elif char == "М":
        # Two outer legs meeting at top vertices + central V reaching 60-95% height
        w_m = s * random.uniform(0.32, 0.46)
        x_l = cx - w_m * 0.5
        x_r = cx + w_m * 0.5
        p_lb = (x_l, y_b)
        p_lt = (x_l + w_m * random.uniform(0.02, 0.14), y_t)
        p_v = (cx + random.uniform(-3, 3), y_t + h * random.uniform(0.60, 0.95))
        p_rt = (x_r - w_m * random.uniform(0.02, 0.14), y_t + random.uniform(-3, 3))
        p_rb = (x_r, y_b)
        _draw_smooth_polyline(canvas, np.array([p_lb, p_lt, p_v, p_rt, p_rb]), thick)

    elif char == "Ә":
        # Handwritten cursive '∂' (lower loop in bottom 45-52% + tall overarching top hook to the left, matches 01 Q1[6], 02 Q1[3], 10 Q2[7], 11 Q6[3])
        w_ae = s * random.uniform(0.26, 0.36)
        x_l = cx - w_ae * 0.5
        x_r = cx + w_ae * 0.5
        y_loop_top = y_t + h * random.uniform(0.48, 0.58)

        p_rmid = (x_r, (y_loop_top + y_b) * 0.5)
        p_top_loop = (cx, y_loop_top)
        p_lmid = (x_l, (y_loop_top + y_b) * 0.52)
        p_bot_loop = (cx, y_b)

        c1 = _quad_bezier(p_rmid, (x_r, y_loop_top), p_top_loop, 14)
        c2 = _quad_bezier(p_top_loop, (x_l, y_loop_top), p_lmid, 14)
        c3 = _quad_bezier(p_lmid, (x_l, y_b), p_bot_loop, 14)
        c4 = _quad_bezier(p_bot_loop, (x_r, y_b), p_rmid, 14)

        # Tall overarching stem rising on the right and curving over the top to the left with clear open space above the loop
        p_top_peak = (cx - w_ae * random.uniform(0.0, 0.15), y_t)
        p_hook_end = (x_l + w_ae * random.uniform(0.0, 0.22), y_t + h * random.uniform(0.12, 0.25))
        arch1 = _quad_bezier(p_rmid, (x_r + random.uniform(0, 3), y_t + h * 0.12), p_top_peak, 16)
        arch2 = _quad_bezier(p_top_peak, (x_l, y_t + 1), p_hook_end, 14)

        _draw_smooth_polyline(canvas, np.vstack([c1, c2, c3, c4, arch1, arch2]), thick)

    elif char == "У":
        if random.random() < 0.70:
            # Cursive 'у' with rounded U-cup and sweeping right descender (matches 08 Q4[2], 11 Q7[8], 02 Q2[5])
            w_u = s * random.uniform(0.26, 0.38)
            x_l = cx - w_u * 0.5
            x_r = cx + w_u * 0.5
            y_cup = y_t + h * random.uniform(0.42, 0.56)
            p_lt = (x_l + random.uniform(-2, 3), y_t + random.uniform(0, 5))
            p_cup = (cx - w_u * 0.08, y_cup)
            p_rt = (x_r, y_t + random.uniform(-2, 5))
            bowl1 = _quad_bezier(p_lt, (x_l, y_cup - 2), p_cup, 14)
            bowl2 = _quad_bezier(p_cup, (x_r - 2, y_cup - 2), p_rt, 14)
            _draw_smooth_polyline(canvas, np.vstack([bowl1, bowl2]), thick)

            p_tail = (cx - w_u * random.uniform(0.05, 0.42), y_b + random.uniform(-4, 4))
            desc = _cubic_bezier(p_rt, (x_r, cy), (x_r - w_u * 0.15, y_b - 6), p_tail, 24)
            _draw_smooth_polyline(canvas, desc, thick)
        else:
            w_u = s * random.uniform(0.28, 0.40)
            x_l = cx - w_u * 0.5
            x_r = cx + w_u * 0.5
            p_lt = (x_l, y_t)
            p_mid = (cx + random.uniform(-2, 4), cy + random.uniform(-4, 4))
            p_rt = (x_r, y_t + random.uniform(-3, 3))
            p_tail = (x_l + w_u * random.uniform(0.05, 0.30), y_b)
            _draw_smooth_polyline(canvas, np.array([p_lt, p_mid]), thick)
            desc = _quad_bezier(p_rt, p_mid, p_tail, 22)
            _draw_smooth_polyline(canvas, desc, thick)

    elif char == "Ү":
        # Symmetric V-fork + straight vertical central stem (matches 04 Q4[3], 07 Q4[8])
        w_v = s * random.uniform(0.28, 0.42)
        x_l = cx - w_v * 0.5
        x_r = cx + w_v * 0.5
        y_join = y_t + h * random.uniform(0.42, 0.54)
        p_lt = (x_l + random.uniform(-3, 3), y_t + random.uniform(-2, 3))
        p_rt = (x_r + random.uniform(-3, 3), y_t + random.uniform(-2, 3))
        p_join = (cx + random.uniform(-2, 2), y_join)
        p_bot = (cx + random.uniform(-3, 3), y_b)
        _draw_smooth_polyline(canvas, np.array([p_lt, p_join, p_rt]), thick)
        _draw_smooth_polyline(canvas, np.array([p_join, p_bot]), thick)
        if random.random() < 0.30:
            _draw_smooth_polyline(canvas, np.array([(p_bot[0] - 5, p_bot[1]), (p_bot[0] + 5, p_bot[1])]), thick * 0.85)

    elif char == "Ч":
        # Shallow upper cup in top 40-52% + straight vertical right stem (matches 02 Q2[3], 04 Q2[7])
        w_ch = s * random.uniform(0.26, 0.38)
        x_l = cx - w_ch * 0.5
        x_r = cx + w_ch * 0.5
        y_cup = y_t + h * random.uniform(0.40, 0.52)
        p_lt = (x_l + random.uniform(-2, 3), y_t + random.uniform(0, 4))
        p_lcorner = (x_l + random.uniform(0, 5), y_cup)
        p_rjoin = (x_r, y_cup - random.uniform(0, 6))
        cup = _cubic_bezier(p_lt, p_lcorner, (cx, y_cup + 3), p_rjoin, 20)
        _draw_smooth_polyline(canvas, cup, thick)
        p_rt = (x_r + random.uniform(-2, 2), y_t + random.uniform(-2, 4))
        p_rb = (x_r + random.uniform(-2, 5), y_b)
        _draw_smooth_polyline(canvas, np.array([p_rt, p_rb]), thick)

    elif char == "В":
        # Vertical stem + narrow/small top loop + wider bottom loop (matches 12 Q7[8], 11 Q7[6], 04 Q1[1], 04 Q3[3])
        w_b = s * random.uniform(0.28, 0.38)
        x_l = cx - w_b * 0.42
        p_lt = (x_l, y_t)
        p_lb = (x_l + random.uniform(-3, 3), y_b)
        _draw_smooth_polyline(canvas, np.array([p_lt, p_lb]), thick)

        if random.random() < 0.28:
            lead = _cubic_bezier((x_l - w_b * 0.32, cy + 8), (x_l - w_b * 0.35, y_t + 6), (x_l - 4, y_t), p_lt, 18)
            _draw_smooth_polyline(canvas, lead, thick * 0.85)

        y_mid = y_t + h * random.uniform(0.40, 0.50)
        top_w = w_b * random.uniform(0.52, 0.82)
        bot_w = w_b * random.uniform(0.88, 1.12)
        p_mid = (x_l + random.uniform(0, 4), y_mid)

        top_loop = _cubic_bezier(
            (x_l, y_t + random.uniform(0, 4)),
            (x_l + top_w * 1.15, y_t - 2),
            (x_l + top_w * 1.15, y_mid),
            p_mid,
            22,
        )
        bot_loop = _cubic_bezier(
            p_mid,
            (x_l + bot_w * 1.20, y_mid - 2),
            (x_l + bot_w * 1.20, y_b + 3),
            (x_l + random.uniform(-2, 5), y_b - random.uniform(0, 4)),
            24,
        )
        _draw_smooth_polyline(canvas, top_loop, thick * 0.95)
        _draw_smooth_polyline(canvas, bot_loop, thick * 0.95)

    elif char == "Ь":
        w_b = s * random.uniform(0.26, 0.36)
        x_l = cx - w_b * 0.45
        p_lt = (x_l + random.uniform(-2, 3), y_t)
        p_lb = (x_l + random.uniform(-2, 2), y_b)
        _draw_smooth_polyline(canvas, np.array([p_lt, p_lb]), thick)
        y_mid = y_t + h * random.uniform(0.48, 0.58)
        bot_loop = _cubic_bezier(
            (x_l, y_mid),
            (x_l + w_b * 1.25, y_mid - 2),
            (x_l + w_b * 1.25, y_b + 2),
            (x_l, y_b - 2),
            24,
        )
        _draw_smooth_polyline(canvas, bot_loop, thick * 0.95)

    elif char in ("Ц", "Щ"):
        is_shch = (char == "Щ")
        w_c = s * random.uniform(0.34, 0.46) if is_shch else s * random.uniform(0.28, 0.38)
        x_l = cx - w_c * 0.5
        x_r = cx + w_c * 0.5
        y_base = y_t + h * random.uniform(0.72, 0.80)

        p_lt = (x_l + random.uniform(-2, 3), y_t + random.uniform(0, 5))
        p_lb = (x_l + random.uniform(-2, 3), y_base)
        p_rt = (x_r + random.uniform(-3, 2), y_t + random.uniform(0, 5))
        p_rb = (x_r + random.uniform(-2, 2), y_base)

        # Left vertical leg goes ALL the way down to p_lb, then bottom bar connects p_lb to p_rb, and right leg goes p_rt to p_rb!
        left_leg = _quad_bezier(p_lt, (x_l, (y_t + y_base) * 0.5), p_lb, 16)
        bot_bar = _quad_bezier(p_lb, (cx, y_base + random.uniform(-1, 5)), p_rb, 16)
        right_leg = _quad_bezier(p_rt, (x_r, (y_t + y_base) * 0.5), p_rb, 16)
        _draw_smooth_polyline(canvas, left_leg, thick)
        _draw_smooth_polyline(canvas, bot_bar, thick)
        _draw_smooth_polyline(canvas, right_leg, thick)

        if is_shch:
            # Middle bar of Щ (sometimes slightly shorter/thinner like 04 Q3[6])
            x_m = cx + random.uniform(-3, 3)
            y_mt = y_t + h * random.uniform(0.02, 0.16)
            _draw_smooth_polyline(canvas, np.array([(x_m, y_mt), (x_m + random.uniform(-2, 2), y_base)]), thick * random.uniform(0.80, 1.0))

        # Prominent right descender loop/hook (matches 04 Q3[6], 01 Q2[2], 01 Q2[4], 05 Q2[5])
        step_w = s * random.uniform(0.05, 0.10)
        tail_y = y_b + s * random.uniform(0.04, 0.10)
        p_step = (p_rb[0] + step_w, p_rb[1] + random.uniform(-2, 3))
        p_tip = (p_step[0] + random.uniform(-6, 3), tail_y)
        tail1 = _quad_bezier(p_rb, (p_rb[0] + step_w * 0.5, p_rb[1]), p_step, 8)
        tail2 = _quad_bezier(p_step, (p_step[0] + 2, (p_step[1] + tail_y) * 0.5), p_tip, 14)
        _draw_smooth_polyline(canvas, np.vstack([tail1, tail2]), thick * 0.90)

    elif char in ("Ж", "Җ"):
        is_zh_tail = (char == "Җ")
        w_zh = s * random.uniform(0.34, 0.46)
        x_l = cx - w_zh * 0.5
        x_r = cx + w_zh * 0.5
        y_base = (y_t + h * random.uniform(0.74, 0.82)) if is_zh_tail else y_b

        # Central vertical stem
        _draw_smooth_polyline(canvas, np.array([(cx + random.uniform(-2, 2), y_t), (cx + random.uniform(-2, 2), y_base)]), thick)

        if random.random() < 0.55:
            d1 = _quad_bezier((x_l, y_t + random.uniform(0, 4)), (cx, (y_t + y_base) * 0.5), (x_r, y_base - random.uniform(0, 4)), 20)
            d2 = _quad_bezier((x_r, y_t + random.uniform(0, 4)), (cx, (y_t + y_base) * 0.5), (x_l, y_base - random.uniform(0, 4)), 20)
            _draw_smooth_polyline(canvas, d1, thick * 0.90)
            _draw_smooth_polyline(canvas, d2, thick * 0.90)
        else:
            ym = (y_t + y_base) * 0.5 + random.uniform(-4, 4)
            _draw_smooth_polyline(canvas, np.array([(x_l, y_t), (cx - 2, ym), (x_l, y_base)]), thick * 0.90)
            _draw_smooth_polyline(canvas, np.array([(x_r, y_t), (cx + 2, ym), (x_r, y_base)]), thick * 0.90)

        if is_zh_tail:
            # Distinct descender hook/tail (matches 01 Q3[4], 02 Q2[8])
            tail_y = y_b + s * random.uniform(0.04, 0.10)
            start_x = cx if random.random() < 0.5 else x_r - 2
            p_step = (start_x + s * random.uniform(0.04, 0.08), y_base)
            p_tip = (p_step[0] + random.uniform(-2, 4), tail_y)
            tail = _quad_bezier((start_x, y_base - 2), p_step, p_tip, 16)
            _draw_smooth_polyline(canvas, tail, thick * 0.90)

    elif char == "Х":
        # Narrow or curved crossed diagonals (matches 05 Q2[7], 05 Q4[6])
        w_x = s * random.uniform(0.24, 0.44)
        x_l = cx - w_x * 0.5
        x_r = cx + w_x * 0.5
        d1 = _quad_bezier((x_l, y_t), (cx + random.uniform(-3, 3), cy + random.uniform(-4, 4)), (x_r, y_b), 22)
        d2 = _quad_bezier((x_r, y_t), (cx + random.uniform(-3, 3), cy + random.uniform(-4, 4)), (x_l, y_b), 22)
        _draw_smooth_polyline(canvas, d1, thick)
        _draw_smooth_polyline(canvas, d2, thick)

    elif char == "Е":
        w_e = s * random.uniform(0.28, 0.38)
        x_l = cx - w_e * 0.5
        x_r = cx + w_e * 0.5
        _draw_smooth_polyline(canvas, np.array([(x_l + random.uniform(-2, 3), y_t), (x_l, y_b)]), thick)
        _draw_smooth_polyline(canvas, np.array([(x_l, y_t), (x_r, y_t + random.uniform(-3, 3))]), thick * 0.92)
        ym = cy + random.uniform(-5, 5)
        _draw_smooth_polyline(canvas, np.array([(x_l, ym), (x_r - w_e * 0.12, ym + random.uniform(-3, 3))]), thick * 0.90)
        _draw_smooth_polyline(canvas, np.array([(x_l, y_b), (x_r + random.uniform(-2, 4), y_b + random.uniform(-3, 3))]), thick * 0.92)

    elif char == "Э":
        w_e = s * random.uniform(0.30, 0.42)
        x_l = cx - w_e * 0.5
        x_r = cx + w_e * 0.5
        arc = _cubic_bezier((x_l, y_t + 6), (x_r + 6, y_t - 4), (x_r + 6, y_b + 4), (x_l, y_b - 6), 28)
        _draw_smooth_polyline(canvas, arc, thick)
        _draw_smooth_polyline(canvas, np.array([(cx - w_e * 0.25, cy), (x_r - 2, cy + random.uniform(-3, 3))]), thick * 0.92)

    elif char == "Я":
        # Matches 12 Q4[4], 01 Q2[6], 04 Q2[5]
        w_ya = s * random.uniform(0.28, 0.40)
        x_l = cx - w_ya * 0.5
        x_r = cx + w_ya * 0.5
        _draw_smooth_polyline(canvas, np.array([(x_r, y_t), (x_r + random.uniform(-2, 3), y_b)]), thick)
        y_loop = y_t + h * random.uniform(0.42, 0.55)
        loop = _cubic_bezier((x_r, y_t + 2), (x_l - 6, y_t - 2), (x_l - 6, y_loop + 2), (x_r, y_loop), 22)
        _draw_smooth_polyline(canvas, loop, thick * 0.95)
        leg = _quad_bezier((x_r - w_ya * 0.25, y_loop - 2), (cx - 4, (y_loop + y_b) * 0.5), (x_l + random.uniform(-3, 4), y_b), 16)
        _draw_smooth_polyline(canvas, leg, thick * 0.95)

    elif char == "Р":
        w_r = s * random.uniform(0.28, 0.40)
        x_l = cx - w_r * 0.45
        _draw_smooth_polyline(canvas, np.array([(x_l, y_t), (x_l + random.uniform(-3, 3), y_b)]), thick)
        y_loop = y_t + h * random.uniform(0.45, 0.58)
        loop = _cubic_bezier((x_l, y_t + 2), (x_l + w_r * 1.25, y_t - 2), (x_l + w_r * 1.25, y_loop + 2), (x_l, y_loop), 24)
        _draw_smooth_polyline(canvas, loop, thick * 0.95)

    elif char == "Ф":
        w_f = s * random.uniform(0.38, 0.52)
        _draw_smooth_polyline(canvas, np.array([(cx, y_t - 4), (cx + random.uniform(-3, 3), y_b + 6)]), thick)
        y_ob = y_t + h * random.uniform(0.55, 0.72)
        axes = (int(round(w_f * 0.5)), int(round((y_ob - y_t) * 0.5)))
        cv2.ellipse(canvas, (int(round(cx)), int(round((y_t + y_ob) * 0.5))), axes, 0, 0, 360, 0, max(1, int(round(thick * 0.92))), lineType=cv2.LINE_AA)

    elif char == "А":
        # Pointed tent '/\' WITH distinct horizontal/slanted crossbar (contrasts with crossbar-free 'Л')
        w_a = s * random.uniform(0.30, 0.44)
        x_l = cx - w_a * 0.5
        x_r = cx + w_a * 0.5
        p_apex = (cx + random.uniform(-4, 4), y_t)
        p_lb = (x_l + random.uniform(-3, 3), y_b)
        p_rb = (x_r + random.uniform(-3, 3), y_b)
        _draw_smooth_polyline(canvas, np.array([p_lb, p_apex, p_rb]), thick)
        yc = y_t + h * random.uniform(0.50, 0.68)
        xl_c = x_l + w_a * 0.15
        xr_c = x_r - w_a * 0.15
        _draw_smooth_polyline(canvas, np.array([(xl_c, yc + random.uniform(-3, 3)), (xr_c, yc + random.uniform(-3, 3))]), thick * 0.92)

    elif char == "К":
        # Left vertical stem + upper-right and lower-right arms meeting near mid-stem (contrasts with 'И' and 'Ү')
        w_k = s * random.uniform(0.28, 0.40)
        x_l = cx - w_k * 0.5
        x_r = cx + w_k * 0.5
        _draw_smooth_polyline(canvas, np.array([(x_l, y_t), (x_l + random.uniform(-2, 2), y_b)]), thick)
        ym = y_t + h * random.uniform(0.44, 0.56)
        _draw_smooth_polyline(canvas, np.array([(x_r, y_t + random.uniform(0, 5)), (x_l + random.uniform(1, 5), ym), (x_r + random.uniform(-2, 4), y_b)]), thick * 0.94)

    elif char == "Ю":
        # Left vertical stem + crossbar + right oval loop (contrasts with 'О')
        w_yu = s * random.uniform(0.38, 0.50)
        x_l = cx - w_yu * 0.5
        x_r = cx + w_yu * 0.5
        _draw_smooth_polyline(canvas, np.array([(x_l, y_t + random.uniform(0, 5)), (x_l + random.uniform(-2, 2), y_b - random.uniform(0, 5))]), thick)
        x_oval_l = x_l + w_yu * random.uniform(0.28, 0.38)
        ym = cy + random.uniform(-4, 4)
        _draw_smooth_polyline(canvas, np.array([(x_l, ym), (x_oval_l + 2, ym)]), thick * 0.90)
        oval_cx = int(round((x_oval_l + x_r) * 0.5))
        oval_cy = int(round(cy))
        axes = (max(4, int(round((x_r - x_oval_l) * 0.5))), max(6, int(round(h * 0.46))))
        cv2.ellipse(canvas, (oval_cx, oval_cy), axes, 0, 0, 360, 0, max(1, int(round(thick * 0.94))), lineType=cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Augmentation Filters & Pen/Pencil Physics
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
    Simulates authentic non-uniform pen/pencil physics:
    - Wide nib width diversity: fine ballpoint (thin erosion), medium gel pen, thick marker/pencil.
    - Downstrokes (vertical/tilted) become thicker and darker than crossbars.
    - Ink pooling / darker blobs at vertices, sharp turns, and stroke intersections.
    - Ballpoint longitudinal striation or partial dry-pen skips.
    - Pencil graphite micro-grain on rough paper tooth.
    """
    binary = (arr < 180).astype(np.uint8)
    if binary.sum() == 0:
        return arr

    shape = arr.shape
    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    sdf = dist_in.astype(np.float32) - dist_out.astype(np.float32)

    smooth_glyph = scipy.ndimage.gaussian_filter(arr.astype(np.float32), sigma=1.0)
    gx = cv2.Sobel(smooth_glyph, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth_glyph, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2) + 1e-5
    downstroke_bias = np.abs(gx) / grad_mag

    # Global pen nib offset: ~70% fine-to-medium ballpoint (-1.5px to +0.3px), ~30% thicker gel/marker (+0.3px to +1.2px)
    if random.random() < 0.70:
        nib_offset = random.uniform(-1.45, 0.25)
    else:
        nib_offset = random.uniform(0.25, 1.15)

    smooth_noise = scipy.ndimage.gaussian_filter(np.random.randn(*shape), sigma=7.0)
    pressure_delta = (
        nib_offset
        + (downstroke_bias - 0.48) * (pressure_variation * 0.95)
        + smooth_noise * (pressure_variation * 0.38)
    )

    # Ink pooling at stroke intersections / corners (high local stroke density)
    local_density = scipy.ndimage.gaussian_filter(binary.astype(np.float32), sigma=3.0)
    junction_boost = np.clip((local_density - 0.42) * 2.5, 0.0, 1.2)
    new_sdf = sdf + pressure_delta + junction_boost * random.uniform(0.2, 0.8)

    soft_stroke = np.clip(new_sdf * 1.35 + 0.5, 0.0, 1.0)
    res = (255.0 * (1.0 - soft_stroke)).astype(np.float32)

    # Pencil graphite grain vs ballpoint ink striation
    writing_tool = random.choices(["ballpoint", "gel_pen", "pencil"], weights=[0.50, 0.30, 0.20], k=1)[0]
    ink_mask = (res < 225).astype(np.float32)

    if writing_tool == "pencil":
        # Graphite grain texture inside the stroke + slightly softer core darkness
        grain = np.random.normal(28.0, 18.0, shape).astype(np.float32)
        grain = scipy.ndimage.gaussian_filter(grain, sigma=0.6)
        res = np.clip(res + ink_mask * grain, 25.0, 255.0)
    elif writing_tool == "ballpoint" and random.random() < 0.40:
        # Ballpoint longitudinal striation / micro-groove
        streak_noise = scipy.ndimage.gaussian_filter(np.random.randn(*shape).astype(np.float32), sigma=(3.5, 0.7))
        streak_fade = np.clip(streak_noise * 35.0, 0.0, 65.0)
        res = np.clip(res + ink_mask * streak_fade, 0.0, 255.0)

    # Dry-pen stroke starvation / skip on a section of a stroke
    if random.random() < starvation_prob:
        center_x = random.randint(int(shape[1] * 0.2), int(shape[1] * 0.8))
        center_y = random.randint(int(shape[0] * 0.2), int(shape[0] * 0.8))
        radius = random.randint(int(shape[0] * 0.12), int(shape[0] * 0.30))
        y_grid, x_grid = np.ogrid[:shape[0], :shape[1]]
        dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        starve_mask = np.clip(1.0 - dist_from_center / (radius + 1e-5), 0.0, 1.0)
        starve_mask = scipy.ndimage.gaussian_filter(starve_mask, sigma=2.5)

        fade_intensity = random.uniform(55.0, 125.0)
        res = np.clip(res + (starve_mask * ink_mask * fade_intensity), 0.0, 255.0)

    return res.astype(np.uint8)


def apply_overwriting_correction(arr, char, font_path, canvas_size=128, prob=0.20):
    """
    Simulates real school mistake correction:
    - Double-strike / re-tracing a stroke (обводка)
    - Overwriting over a fainter underlying mistake
    """
    if random.random() > prob:
        return arr

    mode = random.choices(["double_strike", "underlying_error", "scratch_line"], weights=[0.60, 0.25, 0.15], k=1)[0]
    out = arr.copy().astype(np.float32)

    if mode == "underlying_error":
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
        err_img = render_base_char(err_char, font_path, canvas_size=canvas_size, font_scale=random.uniform(0.66, 0.74))

        shift_x = random.randint(-4, 4)
        shift_y = random.randint(-4, 4)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        err_img = cv2.warpAffine(err_img, M, (canvas_size, canvas_size), borderValue=255)

        under_alpha = (255.0 - err_img.astype(np.float32)) / 255.0 * random.uniform(0.25, 0.42)
        out = 255.0 * (1.0 - under_alpha) + 140.0 * under_alpha

        k_size = random.choice([3, 5])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated_target = cv2.erode(arr, kernel)
        dilated_target = scipy.ndimage.gaussian_filter(dilated_target.astype(np.float32), sigma=0.6)

        target_alpha = (255.0 - dilated_target) / 255.0
        out = out * (1.0 - target_alpha) + (dilated_target * target_alpha)

    elif mode == "double_strike":
        shift_x = random.randint(-3, 3)
        shift_y = random.randint(-3, 3)
        angle = random.uniform(-3.0, 3.0)
        center = (canvas_size / 2.0, canvas_size / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        second_pass = cv2.warpAffine(arr, M, (canvas_size, canvas_size), borderValue=255)
        out = np.minimum(arr, second_pass).astype(np.float32)

    elif mode == "scratch_line":
        p1 = (random.randint(int(canvas_size * 0.20), int(canvas_size * 0.40)),
              random.randint(int(canvas_size * 0.20), int(canvas_size * 0.50)))
        p2 = (p1[0] + random.randint(int(canvas_size * 0.25), int(canvas_size * 0.45)),
              p1[1] + random.randint(-int(canvas_size * 0.12), int(canvas_size * 0.15)))
        scratch = np.full_like(arr, 255)
        cv2.line(scratch, p1, p2, color=random.randint(70, 130), thickness=random.choice([1, 2]))
        scratch = scipy.ndimage.gaussian_filter(scratch.astype(np.float32), sigma=0.6)
        out = np.minimum(out, scratch)

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_flourishes_and_stray_marks(arr, prob=0.35):
    """
    Adds pen tails, starting hooks, endpoint ink pooling blobs (кляксы).
    """
    if random.random() > prob:
        return arr

    res = arr.copy()
    h, w = res.shape
    ink_ys, ink_xs = np.where(res < 180)

    if len(ink_xs) < 15:
        return res

    num_marks = random.randint(1, 2)
    for _ in range(num_marks):
        mark_type = random.choices(["ink_blob", "hook_tail", "stray_loop"], weights=[0.50, 0.35, 0.15], k=1)[0]

        if mark_type == "hook_tail":
            idx = random.choice([0, len(ink_xs) - 1, random.randint(0, len(ink_xs) - 1)])
            start_x = int(ink_xs[idx])
            start_y = int(ink_ys[idx])

            ctrl_x = start_x + random.randint(-8, 8)
            ctrl_y = start_y + random.randint(-8, 8)
            end_x = ctrl_x + random.randint(-7, 7)
            end_y = ctrl_y + random.randint(-7, 7)

            pts = []
            for t in np.linspace(0, 1, 10):
                bx = (1 - t)**2 * start_x + 2 * (1 - t) * t * ctrl_x + t**2 * end_x
                by = (1 - t)**2 * start_y + 2 * (1 - t) * t * ctrl_y + t**2 * end_y
                pts.append([int(bx), int(by)])
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(res, [pts], isClosed=False, color=random.randint(30, 100), thickness=random.choice([1, 2]), lineType=cv2.LINE_AA)

        elif mark_type == "stray_loop":
            idx = random.choice([0, len(ink_xs) - 1])
            cx = int(np.clip(ink_xs[idx] + random.randint(-4, 4), 4, w - 5))
            cy = int(np.clip(ink_ys[idx] + random.randint(-4, 4), 4, h - 5))
            axes = (random.randint(2, 5), random.randint(2, 4))
            angle = random.randint(0, 180)
            cv2.ellipse(res, (cx, cy), axes, angle, 0, random.randint(200, 340), color=random.randint(50, 120), thickness=1, lineType=cv2.LINE_AA)

        elif mark_type == "ink_blob":
            idx = random.choice([0, len(ink_xs) - 1, random.randint(0, len(ink_xs) - 1)])
            bx = int(ink_xs[idx] + random.randint(-1, 1))
            by = int(ink_ys[idx] + random.randint(-1, 1))
            cv2.circle(res, (bx, by), random.choice([2, 3, 4]), color=random.randint(10, 55), thickness=-1, lineType=cv2.LINE_AA)

    return res


def apply_cell_crop_artifacts(arr, prob=0.75):
    """Legacy cell-border artifact helper kept for backward compatibility."""
    if random.random() > prob:
        return arr

    res = arr.copy()
    h, w = res.shape
    edges = [random.choice(["top", "bottom", "left", "right"])]
    line_color = random.randint(90, 180)
    line_thickness = random.choice([1, 2])
    tilt = random.uniform(-1.0, 1.0)

    for edge in edges:
        if edge == "left":
            lx = random.randint(0, 2)
            cv2.line(res, (int(np.clip(lx - tilt, 0, w - 1)), 0), (int(np.clip(lx + tilt, 0, w - 1)), h - 1), color=line_color, thickness=line_thickness)
        elif edge == "right":
            rx = random.randint(w - 3, w - 1)
            cv2.line(res, (int(np.clip(rx - tilt, 0, w - 1)), 0), (int(np.clip(rx + tilt, 0, w - 1)), h - 1), color=line_color, thickness=line_thickness)
        elif edge == "top":
            ty = random.randint(0, 2)
            cv2.line(res, (0, int(np.clip(ty - tilt, 0, h - 1))), (w - 1, int(np.clip(ty + tilt, 0, h - 1))), color=line_color, thickness=line_thickness)
        elif edge == "bottom":
            by = random.randint(h - 3, h - 1)
            cv2.line(res, (0, int(np.clip(by - tilt, 0, h - 1))), (w - 1, int(np.clip(by + tilt, 0, h - 1))), color=line_color, thickness=line_thickness)

    return res


def apply_paper_and_lighting(arr, bg_range=(225, 252), add_gradient=True, add_faint_grid=False):
    """Renders realistic classroom paper with non-uniform phone/hand lighting and sensor grain."""
    h, w = arr.shape
    base_lum = random.randint(bg_range[0], bg_range[1])

    if add_gradient:
        angle = random.uniform(0, 2 * math.pi)
        contrast = random.uniform(12.0, 52.0)
        y_norm, x_norm = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing="ij")
        directional_shadow = (x_norm * math.cos(angle) + y_norm * math.sin(angle)) * contrast
        r2 = (x_norm**2 + y_norm**2) * random.uniform(4.0, 15.0)
        lighting_field = base_lum + directional_shadow - r2
    else:
        lighting_field = np.full((h, w), base_lum, dtype=np.float32)

    pulp_cloud = scipy.ndimage.gaussian_filter(np.random.normal(0, 3.5, (h, w)), sigma=2.8)
    pulp_tooth = np.random.normal(0, 1.8, (h, w))
    paper_bg = np.clip(lighting_field + pulp_cloud + pulp_tooth, 165, 255).astype(np.float32)

    stroke_alpha = (255.0 - arr.astype(np.float32)) / 255.0
    final = paper_bg * (1.0 - stroke_alpha) + (arr.astype(np.float32) * stroke_alpha)

    cam_noise = np.random.normal(0, 1.8, (h, w))
    final = np.clip(final + cam_noise, 0, 255).astype(np.uint8)
    return final


def apply_camera_defocus(arr, prob=0.40):
    """Simulates slight smartphone camera defocus or optical softness."""
    if random.random() > prob:
        return arr
    k = 3
    sigma = random.uniform(0.4, 0.9)
    return cv2.GaussianBlur(arr, (k, k), sigma)


def apply_affine_cell_placement(
    arr,
    max_angle=11.0,
    max_shear=12.0,
    scale_range=(0.72, 0.98),
    aspect_ratio_range=(0.76, 1.22),
):
    """
    Applies realistic handwriting structural deformations:
    - Independent horizontal vs vertical aspect ratio squeeze/stretch (narrow vs wide handwriting).
    - Natural hand tilt / rotation (-11° to +11°).
    - Italic slant / shear (-12° to +12°).
    - Positioning jitter within the cell box.
    """
    h, w = arr.shape
    angle = random.uniform(-max_angle, max_angle)
    shear = random.uniform(-max_shear, max_shear)
    base_scale = random.uniform(scale_range[0], scale_range[1])
    aspect = random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])

    scale_x = base_scale * math.sqrt(aspect)
    scale_y = base_scale / math.sqrt(aspect)

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 0] *= scale_x
    rot_mat[0, 1] *= scale_y
    rot_mat[1, 0] *= scale_x
    rot_mat[1, 1] *= scale_y

    shear_rad = np.deg2rad(shear)
    rot_mat[0, 1] += np.tan(shear_rad) * rot_mat[0, 0]

    shift_x = random.uniform(-w * 0.06, w * 0.06)
    shift_y = random.uniform(-h * 0.06, h * 0.06)
    rot_mat[0, 2] = center[0] - (rot_mat[0, 0] * center[0] + rot_mat[0, 1] * center[1]) + shift_x
    rot_mat[1, 2] = center[1] - (rot_mat[1, 0] * center[0] + rot_mat[1, 1] * center[1]) + shift_y

    return cv2.warpAffine(arr, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def simulate_camera_and_scanner_crop(
    glyph_128,
    canvas_size=64,
    enable_paper_texture=True,
    enable_camera_blur=True,
    enable_cell_borders=True,
):
    """
    Mirrors the exact physical and algorithmic pipeline of `BlankOCRScanner`:
    1. Finds tight ink bounding box on the high-res canvas so we know the true glyph bounds.
    2. Places the glyph inside an ~88x88 paper cell patch with realistic paper shading/lighting.
    3. Distant Camera Bottleneck: Downsamples the 88x88 cell to 17..38px (matching a 768x1024
       full-page phone photo from a distance), applies optical lens blur & sensor noise, and
       upscales back to 88x88 via `cv2.INTER_LANCZOS4` (matching `rectify_sheet`).
    4. `classify_cell` Normalization:
       - Extracts the tight ink bounding box (`pad = 2` in 88x88 coordinates),
       - Occasionally (~16%) includes a faint printed cell border remnant along one edge of the tight crop
         (matching `03 Q1[3]`, `03 Q4[8]`, `09 Q1[5]`),
       - Resizes `max(gh, gw)` to `target_dim = 48.0` (with minor jitter 46..50) using `INTER_AREA` / `INTER_LANCZOS4`,
       - Applies percentile contrast stretch (`p2..p98 -> 10..250`, or `CLAHE`),
       - Centers `(new_h, new_w)` onto a `64x64` canvas filled with background value `250`!
    """
    ink_ys, ink_xs = np.where(glyph_128 < 210)
    if len(ink_xs) < 8:
        return np.full((canvas_size, canvas_size), 250, dtype=np.uint8)

    # 1. Place on an 88x88 rectified cell patch
    cell_dim = 88
    cell_img = cv2.resize(glyph_128, (cell_dim, cell_dim), interpolation=cv2.INTER_AREA)

    # Compute true ink bbox in 88x88 cell coordinates before adding paper illumination
    c_ys, c_xs = np.where(cell_img < 215)
    if len(c_xs) == 0:
        return np.full((canvas_size, canvas_size), 250, dtype=np.uint8)

    x1, x2 = int(np.min(c_xs)), int(np.max(c_xs))
    y1, y2 = int(np.min(c_ys)), int(np.max(c_ys))

    # Add paper illumination & texture on the 88x88 cell
    if enable_paper_texture:
        # Vary lighting contrast: some cells are clean (contrast 6..18), some shaded/blurry (25..62 like blank_03)
        is_shaded = random.random() < 0.35
        cell_shaded = apply_paper_and_lighting(
            cell_img,
            bg_range=(215, 246) if is_shaded else (235, 252),
            add_gradient=True,
        )
    else:
        cell_shaded = cell_img.copy()

    # 2. Distant Camera Sensor Resolution Bottleneck + INTER_LANCZOS4 Rectification
    if enable_camera_blur:
        # In a 768x1024 full-page photo, each cell is ~20-34px across (or 15-22px when blurry/distant like blank_03)
        if random.random() < 0.25:
            sensor_px = random.randint(16, 23)  # heavy low-res camera blur (blank_03 style)
        else:
            sensor_px = random.randint(23, 38)  # normal distant phone photo (blank_01..blank_12 style)

        low_res = cv2.resize(cell_shaded, (sensor_px, sensor_px), interpolation=cv2.INTER_AREA)

        # Optical PSF blur at sensor level
        if random.random() < 0.65:
            sigma = random.uniform(0.35, 0.85)
            low_res = cv2.GaussianBlur(low_res, (3, 3), sigma)

        # Slight sensor shot noise at low resolution
        if random.random() < 0.50:
            noise = np.random.normal(0, random.uniform(1.0, 2.8), low_res.shape)
            low_res = np.clip(low_res.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Rectification upscale via INTER_LANCZOS4 (matching BlankOCRScanner.rectify_sheet)
        rectified_cell = cv2.warpPerspective(
            low_res,
            np.array([[cell_dim / float(sensor_px), 0, 0], [0, cell_dim / float(sensor_px), 0], [0, 0, 1]], dtype=np.float32),
            (cell_dim, cell_dim),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        rectified_cell = cell_shaded

    # 3. Tight ink bounding box extraction (matching classify_cell pad=2)
    pad = random.choice([2, 2, 3])
    bx1 = max(0, x1 - pad)
    by1 = max(0, y1 - pad)
    bx2 = min(cell_dim - 1, x2 + pad)
    by2 = min(cell_dim - 1, y2 + pad)

    glyph_crop = rectified_cell[by1:by2 + 1, bx1:bx2 + 1].copy()
    gh, gw = glyph_crop.shape[:2]

    # Occasionally (~16%) inject a faint residual border shadow on one edge of the tight crop
    # (matches 03 Q1[3], 03 Q4[8], 09 Q1[5] where a cell border touched the tight bbox)
    if enable_cell_borders and random.random() < 0.16 and gh >= 8 and gw >= 8:
        edge = random.choice(["left", "right", "top", "bottom"])
        b_col = random.randint(95, 185)
        b_thick = random.choice([1, 2])
        if edge == "left":
            glyph_crop[:, :b_thick] = np.minimum(glyph_crop[:, :b_thick], b_col)
        elif edge == "right":
            glyph_crop[:, -b_thick:] = np.minimum(glyph_crop[:, -b_thick:], b_col)
        elif edge == "top":
            glyph_crop[:b_thick, :] = np.minimum(glyph_crop[:b_thick, :], b_col)
        elif edge == "bottom":
            glyph_crop[-b_thick:, :] = np.minimum(glyph_crop[-b_thick:, :], b_col)

    # 4. Aspect-ratio preserving scaling to canonical ~48px occupancy (matching classify_cell)
    max_dim = max(gh, gw)
    target_dim = random.uniform(46.5, 49.5) * (canvas_size / 64.0)
    scale = target_dim / float(max_dim + 1e-5)
    max_allowed = max(4, int(canvas_size - 4))
    new_w = max(1, min(max_allowed, int(round(gw * scale))))
    new_h = max(1, min(max_allowed, int(round(gh * scale))))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized_glyph = cv2.resize(glyph_crop, (new_w, new_h), interpolation=interp)

    # 5. Contrast normalization (matching classify_cell percentile p2..p98 -> 10..250, or CLAHE)
    if random.random() < 0.10:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm_glyph = clahe.apply(resized_glyph)
    else:
        p_lo = np.percentile(resized_glyph, 2)
        p_hi = np.percentile(resized_glyph, 98)
        norm_glyph = np.clip((resized_glyph.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)

    # 6. Center on 64x64 canvas with background value 250 (matching classify_cell inp_64)
    inp_64 = np.full((canvas_size, canvas_size), 250, dtype=np.uint8)
    off_x = (canvas_size - new_w) // 2 + random.choice([-1, 0, 0, 0, 1])
    off_y = (canvas_size - new_h) // 2 + random.choice([-1, 0, 0, 0, 1])
    off_x = max(0, min(canvas_size - new_w, off_x))
    off_y = max(0, min(canvas_size - new_h, off_y))
    inp_64[off_y:off_y + new_h, off_x:off_x + new_w] = norm_glyph

    return inp_64


def normalize_real_cell_like_scanner(cell_gray, canvas_size=64):
    """
    Normalizes an uncropped 64x64 or 100x100 real dataset cell image (e.g. from `realdataset/`)
    using the exact `BlankOCRScanner.classify_cell` border-trimming, connected-component ink
    segmentation, tight bounding-box extraction, 48px scaling, p2..p98 contrast normalization,
    and 250-background centering pipeline!
    """
    ch, cw = cell_gray.shape[:2]
    my = max(4, int(round(ch * 0.08)))
    mx = max(4, int(round(cw * 0.08)))
    inner = cell_gray[my:ch - my, mx:cw - mx] if (ch - 2 * my >= 12 and cw - 2 * mx >= 12) else cell_gray
    ih, iw = inner.shape[:2]

    bg_val = float(np.median(inner))
    p5_val = float(np.percentile(inner, 5))
    contrast_range = bg_val - p5_val
    ink_delta = 20.0 if contrast_range > 55 else (14.0 if contrast_range > 30 else max(7.0, 0.32 * contrast_range))
    ink_mask = (inner < (bg_val - ink_delta)).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask)
    if num_labels <= 1:
        return np.full((canvas_size, canvas_size), 250, dtype=np.uint8)

    candidate_indices = []
    for i in range(1, num_labels):
        bx, by, bw, bh, area = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
            stats[i, cv2.CC_STAT_AREA],
        )
        is_edge = (
            (bw > 0.45 * iw and bh <= 6 and (by <= 2 or by + bh >= ih - 2))
            or (bh > 0.45 * ih and bw <= 6 and (bx <= 2 or bx + bw >= iw - 2))
            or ((bx <= 2 or by <= 2 or bx + bw >= iw - 2 or by + bh >= ih - 2) and area < 18)
        )
        if not is_edge:
            candidate_indices.append(i)

    if not candidate_indices:
        candidate_indices = list(range(1, num_labels))

    best_idx = max(candidate_indices, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
    max_area = stats[best_idx, cv2.CC_STAT_AREA]
    valid_ink = (labels == best_idx).astype(np.uint8)
    for i in candidate_indices:
        if i != best_idx and stats[i, cv2.CC_STAT_AREA] >= max(6, 0.12 * max_area):
            valid_ink[labels == i] = 1

    ys, xs = np.where(valid_ink > 0)
    if len(xs) == 0:
        glyph_crop = inner
    else:
        pad = 2
        x1, x2 = max(0, int(np.min(xs)) - pad), min(iw - 1, int(np.max(xs)) + pad)
        y1, y2 = max(0, int(np.min(ys)) - pad), min(ih - 1, int(np.max(ys)) + pad)
        glyph_crop = inner[y1:y2 + 1, x1:x2 + 1]

    gh, gw = glyph_crop.shape[:2]
    scale = 48.0 * (canvas_size / 64.0) / float(max(gh, gw) + 1e-5)
    new_w = max(1, min(canvas_size - 4, int(round(gw * scale))))
    new_h = max(1, min(canvas_size - 4, int(round(gh * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(glyph_crop, (new_w, new_h), interpolation=interp)

    p_lo, p_hi = np.percentile(resized, 2), np.percentile(resized, 98)
    norm_glyph = np.clip((resized.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)

    inp_64 = np.full((canvas_size, canvas_size), 250, dtype=np.uint8)
    off_x = (canvas_size - new_w) // 2
    off_y = (canvas_size - new_h) // 2
    inp_64[off_y:off_y + new_h, off_x:off_x + new_w] = norm_glyph
    return inp_64


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
    enable_handwritten_variants=True,
    align_to_scanner=True,
):
    """
    Comprehensive, realistic Tatar handwritten character generator (v2) aligned with
    `BlankOCRScanner.classify_cell` inference preprocessing and distant mobile camera optics.
    Preserves full backward compatibility with all existing dataset & training callers.
    """
    internal_size = max(128, canvas_size * 2)

    # 1. Base glyph selection: procedural handwritten allograph, lowercase allograph, or font glyph
    use_procedural = (
        enable_handwritten_variants
        and (char in HANDWRITTEN_ALLOGRAPH_CHARS)
        and (random.random() < 0.58)
    )
    use_lower_allograph = (
        not use_procedural
        and enable_handwritten_variants
        and (char in _LOWERCASE_ALLOGRAPH_PROB)
        and (random.random() < _LOWERCASE_ALLOGRAPH_PROB[char])
    )

    if use_procedural:
        raw = render_handwritten_allograph(char, canvas_size=internal_size)
    elif use_lower_allograph:
        raw = render_base_char(char.lower(), font_path, canvas_size=internal_size, font_scale=0.76)
    elif base_img is not None:
        raw = base_img.copy()
        if raw.shape != (internal_size, internal_size):
            raw = cv2.resize(raw, (internal_size, internal_size), interpolation=cv2.INTER_AREA)
    else:
        raw = render_base_char(char, font_path, canvas_size=internal_size)

    # 2. Overwriting / double-strike correction
    if enable_overwriting:
        img = apply_overwriting_correction(raw, char, font_path, canvas_size=internal_size, prob=0.18)
    else:
        img = raw.copy()

    # 3. Non-uniform stroke pressure, vertex ink pooling, ballpoint skipping & pencil grain
    img = apply_variable_thickness_and_starvation(
        img,
        pressure_variation=pressure_variation,
        starvation_prob=0.30,
    )

    # 4. Extra flourishes, hooks, endpoint ink blobs
    if enable_flourishes:
        img = apply_flourishes_and_stray_marks(img, prob=0.32)

    # 5. Elastic muscle tremor / wobbly strokes
    img = apply_elastic_wobble(img, alpha=elastic_strength * 0.75, sigma=6.0)

    # 6. Structural handwriting variations: aspect-ratio squeeze/stretch, slant/shear, rotation
    img = apply_affine_cell_placement(
        img,
        max_angle=11.0,
        max_shear=12.0,
        scale_range=(0.76, 0.98),
        aspect_ratio_range=(0.74, 1.24),
    )

    # 7. Distant camera resolution bottleneck + INTER_LANCZOS4 + classify_cell 48px tight-crop normalization
    if align_to_scanner:
        return simulate_camera_and_scanner_crop(
            img,
            canvas_size=canvas_size,
            enable_paper_texture=enable_paper_texture,
            enable_camera_blur=enable_camera_blur or True,
            enable_cell_borders=enable_cell_borders,
        )

    # Legacy fallback path if align_to_scanner=False
    if internal_size != canvas_size:
        img = cv2.resize(img, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    if enable_cell_borders:
        img = apply_cell_crop_artifacts(img, prob=0.70)
    if enable_camera_blur:
        img = apply_camera_defocus(img, prob=0.35)
    if enable_paper_texture:
        img = apply_paper_and_lighting(img, bg_range=(225, 252), add_gradient=True)
    return img

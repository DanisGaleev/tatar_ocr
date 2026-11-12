"""Diagnostic, annotation, border-suppression, and benchmark export tools for BlankOCRScanner."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = "models/tatar_ocr_v4_handwritten39_int8.onnx"

# Annotation confidence thresholds and RGBA colors
CONF_GREEN = 80.0
CONF_YELLOW = 50.0
BOX_COLOR_GREEN = (34, 197, 94, 255)
BADGE_BG_GREEN = (34, 197, 94, 230)
BOX_COLOR_YELLOW = (234, 179, 8, 255)
BADGE_BG_YELLOW = (202, 138, 4, 230)
BOX_COLOR_RED = (239, 68, 68, 255)
BADGE_BG_RED = (220, 38, 38, 230)
TEXT_WHITE = (255, 255, 255)

# Badge and banner layout
BOX_BORDER_WIDTH = 4
BADGE_W = 56
BADGE_H = 38
BADGE_OFFSET_X = 4
BADGE_OFFSET_Y = -20
BADGE_RADIUS = 6
BADGE_TEXT_PAD_X = 10
BADGE_TEXT_PAD_Y = 1
BANNER_HEIGHT = 50
BANNER_BG = (15, 23, 42, 230)
BANNER_TEXT_POS = (30, 8)

# Annotation font sizes and candidate paths (local -> system -> PIL default)
FONT_SIZE_BADGE = 36
FONT_SIZE_BANNER = 22
LOCAL_FONT_PATH = WORKSPACE_DIR / "fonts" / "Rubik-Regular.ttf"
FONT_PAIR_CANDIDATES = (
    (str(LOCAL_FONT_PATH), str(LOCAL_FONT_PATH)),
    ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)

# Diagnostic grid-border suppression parameters
BORDER_SAFE_MARGIN_PCT = 8
BORDER_SAFE_MIN_PX = 6
BORDER_ENVELOPE_PAD_PX = 2
BORDER_THRESH_BLOCK_SIZE = 21
BORDER_THRESH_C = 10
BORDER_H_KERNEL_LEN = 80
BORDER_V_KERNEL_LEN = 75
BORDER_DILATE_SIZE = 3
DEFAULT_BORDER_COLOR = (255, 255, 255)

SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _confidence_colors(conf: float):
    if conf >= CONF_GREEN:
        return BOX_COLOR_GREEN, BADGE_BG_GREEN
    if conf >= CONF_YELLOW:
        return BOX_COLOR_YELLOW, BADGE_BG_YELLOW
    return BOX_COLOR_RED, BADGE_BG_RED


def load_annotation_fonts():
    """Load badge and banner PIL fonts with local, system, and default fallbacks."""
    for badge_path, banner_path in FONT_PAIR_CANDIDATES:
        try:
            font_badge = ImageFont.truetype(badge_path, FONT_SIZE_BADGE)
            font_banner = ImageFont.truetype(banner_path, FONT_SIZE_BANNER)
            return font_badge, font_banner
        except Exception:
            continue
    default_font = ImageFont.load_default()
    return default_font, default_font


def draw_cell_badge(draw, x, y, w, h, cell, font_badge):
    """Draw a confidence-colored frame and top-right character badge for one cell."""
    ch = cell["char"]
    conf = float(cell["confidence"])
    box_col, badge_bg = _confidence_colors(conf)

    draw.rectangle([(x, y), (x + w, y + h)], outline=box_col, width=BOX_BORDER_WIDTH)
    bx = x + w - BADGE_W + BADGE_OFFSET_X
    by = y + BADGE_OFFSET_Y
    draw.rounded_rectangle([(bx, by), (bx + BADGE_W, by + BADGE_H)], radius=BADGE_RADIUS, fill=badge_bg)
    draw.text((bx + BADGE_TEXT_PAD_X, by + BADGE_TEXT_PAD_Y), ch, fill=TEXT_WHITE, font=font_badge)


def annotate_blank_result(rectified_bgr: np.ndarray, scan_result: dict) -> np.ndarray:
    """Render cell prediction badges and a summary status banner onto a rectified sheet."""
    font_badge, font_banner = load_annotation_fonts()
    annotated_pil = Image.fromarray(cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_pil, "RGBA")

    cells = list(scan_result.get("student_name_cells", []))
    for q in scan_result.get("questions", []):
        cells.extend(q.get("cells", []))

    for cell in cells:
        if not cell.get("is_empty"):
            x, y, w, h = cell["bbox"]
            draw_cell_badge(draw, x, y, w, h, cell, font_badge)

    img_w = rectified_bgr.shape[1]
    stats = scan_result.get("stats", {})
    student_name = scan_result.get("student_name") or "N/A"
    total_letters = stats.get("total_letters", 0)
    avg_conf = float(stats.get("avg_confidence", 0.0))
    method = scan_result.get("rectification_method", "unknown")

    draw.rectangle([(0, 0), (img_w, BANNER_HEIGHT)], fill=BANNER_BG)
    banner_text = (
        f"Tatar OCR Scanner • Name: '{student_name}' • "
        f"Recognized {total_letters} letters • Avg Confidence: {avg_conf:.1f}% • Alignment: {method}"
    )
    draw.text(BANNER_TEXT_POS, banner_text, fill=TEXT_WHITE, font=font_banner)
    return cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)


def suppress_grid_borders(
    rectified_bgr,
    scanner=None,
    name_cells=None,
    question_rows=None,
    border_color=DEFAULT_BORDER_COLOR,
    safe_margin_pct=BORDER_SAFE_MARGIN_PCT,
):
    """Non-destructively suppress printed grid borders outside each cell's inner safe core."""
    if name_cells is None or question_rows is None:
        if scanner is None:
            from blank_pipeline import BlankOCRScanner
            scanner = BlankOCRScanner()
        name_cells, question_rows = scanner.extract_cells(rectified_bgr)

    ih, iw = rectified_bgr.shape[:2]
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY) if rectified_bgr.ndim == 3 else rectified_bgr.copy()
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        BORDER_THRESH_BLOCK_SIZE,
        BORDER_THRESH_C,
    )

    perimeter_mask = np.zeros((ih, iw), dtype=np.uint8)
    all_cells = list(name_cells)
    for q in question_rows:
        all_cells.extend(q.get("cells", []))

    for cell in all_cells:
        cx, cy, cw, ch = cell["bbox"]
        my = max(BORDER_SAFE_MIN_PX, int(round(ch * (safe_margin_pct / 100.0))))
        mx = max(BORDER_SAFE_MIN_PX, int(round(cw * (safe_margin_pct / 100.0))))

        x1_out = max(0, cx - BORDER_ENVELOPE_PAD_PX)
        y1_out = max(0, cy - BORDER_ENVELOPE_PAD_PX)
        x2_out = min(iw, cx + cw + BORDER_ENVELOPE_PAD_PX)
        y2_out = min(ih, cy + ch + BORDER_ENVELOPE_PAD_PX)
        cv2.rectangle(perimeter_mask, (x1_out, y1_out), (x2_out, y2_out), 255, -1)

        x1_in, y1_in = cx + mx, cy + my
        x2_in, y2_in = cx + cw - mx, cy + ch - my
        if x2_in > x1_in and y2_in > y1_in:
            cv2.rectangle(perimeter_mask, (x1_in, y1_in), (x2_in, y2_in), 0, -1)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (BORDER_H_KERNEL_LEN, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, BORDER_V_KERNEL_LEN))
    detected_lines = cv2.bitwise_or(
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel),
    )

    safe_border_mask = cv2.bitwise_and(detected_lines, perimeter_mask)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (BORDER_DILATE_SIZE, BORDER_DILATE_SIZE))
    final_border_mask = cv2.bitwise_and(cv2.dilate(safe_border_mask, dilate_kernel), perimeter_mask)

    suppressed_bgr = rectified_bgr.copy()
    if isinstance(border_color, (int, float)):
        fill_val = np.array([border_color, border_color, border_color], dtype=np.uint8)
    else:
        fill_val = np.array(border_color, dtype=np.uint8)

    suppressed_bgr[final_border_mask > 0] = fill_val
    suppressed_count = int(np.count_nonzero(final_border_mask))
    return suppressed_bgr, final_border_mask, suppressed_count


def inspect_cell(scanner, cell_bgr, **kwargs):
    """Classify a single cell crop with diagnostic debug tensors and ink stats enabled."""
    kwargs["debug"] = True
    return scanner.classify_cell(cell_bgr, **kwargs)


def sanitize_json(obj):
    """Recursively convert NumPy scalars, arrays, and tuples into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_benchmark_predictions(images_dir, output_path, scanner=None):
    """Run OCR across a directory of blank images and write per-image and combined JSON."""
    if scanner is None:
        from blank_pipeline import BlankOCRScanner
        scanner = BlankOCRScanner()

    images_path = Path(images_dir)
    images = sorted(p for p in images_path.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    output_path = Path(output_path)

    if output_path.suffix.lower() == ".json":
        out_dir = output_path.parent / output_path.stem
        out_file = output_path
    else:
        out_dir = output_path
        out_file = output_path / "predictions.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    predictions_map = {}
    for img_p in images:
        image_id = img_p.stem
        print(f"Processing {img_p.name} ...")
        raw = scanner.process_image(img_p, annotate=False)
        payload = sanitize_json({
            "image_id": image_id,
            "pipeline": "python_blank_pipeline",
            "rectification_method": raw.get("rectification_method", "unknown"),
            "student_name": {
                "text": raw.get("student_name", ""),
                "cells": raw.get("student_name_cells", []),
            },
            "questions": raw.get("questions", []),
            "stats": raw.get("stats", {}),
        })
        predictions_map[image_id] = payload

        with open(out_dir / f"{image_id}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions_map, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(images)} predictions to {out_file} and directory {out_dir}")
    return predictions_map


def main():
    """CLI entry point for exporting benchmark OCR predictions."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Tatar Blank OCR Scanner Benchmark Exporter")
    parser.add_argument("--images-dir", type=str, default="benchmark/images", help="Path to images directory")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/predictions/python_baseline.json",
        help="Path to output json file or directory",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to model file (.onnx or .pth)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, cuda)")
    args = parser.parse_args()

    from blank_pipeline import BlankOCRScanner
    scanner = BlankOCRScanner(model_path=args.model, device=args.device)
    export_benchmark_predictions(args.images_dir, args.output, scanner=scanner)


if __name__ == "__main__":
    main()

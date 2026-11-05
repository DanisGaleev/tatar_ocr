"""
Tatar OCR - Interactive Pipeline Visualizer & Step-by-Step Inspector.

Generates visual stage-by-stage outputs and a standalone interactive HTML inspector:
  Step 1: Marker Detection (ArUco corners & question anchors)
  Step 2: Perspective Rectification (Canonical 2100x2970 canvas)
  Step 3: Grid Snapping & Localization (Name & question cells)
  Step 4: Printed Border Suppression (Paper white ink isolation)
  Step 5: Character Recognition & Scoring (TatarOCRNet & confidence badges)
"""

import os
import sys
import json
import time
import base64
import argparse
import shutil
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from blank_pipeline import BlankOCRScanner, CANVAS_W, CANVAS_H, DST_CORNERS, CANONICAL_ANCHORS
from tatar_ocr_augmentation import TATAR_UPPERCASE

TATAR_SPECIFIC_LETTERS = set("ӘӨҮҖҢҺ")


def load_font(size: int):
    font_paths = [
        WORKSPACE_DIR / "fonts" / "Rubik-Regular.ttf",
        WORKSPACE_DIR / "fonts" / "Comfortaa.ttf",
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for p in font_paths:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


def json_serializable(obj):
    """Serialize numpy data types cleanly for json.dump."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def cv2_to_b64(img_bgr, quality=80, max_dim=None) -> str:
    """Encode OpenCV BGR image to Base64 JPEG data URL."""
    if max_dim is not None:
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


class PipelineVisualizer:
    def __init__(self, scanner: BlankOCRScanner = None):
        self.scanner = scanner or BlankOCRScanner()
        self.font_title = load_font(40)
        self.font_body = load_font(26)
        self.font_badge = load_font(20)
        self.font_small = load_font(16)

    def process_image(self, image_path: Path, output_dir: Path) -> dict:
        image_path = Path(image_path)
        blank_id = image_path.stem
        blank_out_dir = output_dir / blank_id
        blank_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==========================================")
        print(f"[*] Processing Blank: {image_path.name}")
        print(f"==========================================")

        raw_bgr = cv2.imread(str(image_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"Cannot load image at {image_path}")
        h_orig, w_orig = raw_bgr.shape[:2]

        timings = {}

        # ----------------------------------------------------
        # STEP 1: Marker Detection
        # ----------------------------------------------------
        t0 = time.perf_counter()
        marker_map = self.scanner.detect_markers_padded(raw_bgr, pad=60)
        t_markers = (time.perf_counter() - t0) * 1000.0
        timings["markers"] = round(t_markers, 1)

        vis_step1 = raw_bgr.copy()
        # Draw detected ArUco markers with PIL for crisp Cyrillic badges
        vis_step1_pil = Image.fromarray(cv2.cvtColor(vis_step1, cv2.COLOR_BGR2RGB))
        draw1 = ImageDraw.Draw(vis_step1_pil, "RGBA")

        # Color schemes:
        # Corner markers (0..3): Vibrant Gold / Cyan
        # Question anchors (11..18): Vibrant Emerald Green
        corner_colors = {
            0: (255, 215, 0, 220),   # TL Gold
            1: (0, 229, 255, 220),   # TR Cyan
            2: (255, 64, 129, 220),  # BL Pink
            3: (105, 240, 174, 220), # BR Mint
        }
        detected_corners = []
        detected_anchors = []

        for mid, pts in marker_map.items():
            pts_int = pts.astype(np.int32)
            pts_poly = [(int(p[0]), int(p[1])) for p in pts_int]
            center = pts.mean(axis=0)
            cx, cy = int(center[0]), int(center[1])

            if mid in [0, 1, 2, 3]:
                detected_corners.append(mid)
                fill_color = corner_colors.get(mid, (255, 215, 0, 180))
                # Outline polygon
                draw1.polygon(pts_poly, outline=(fill_color[0], fill_color[1], fill_color[2], 255), width=4)
                # Corner vertices circles
                for p in pts_poly:
                    draw1.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
                # Label badge
                lbl = f"УГОЛ #{mid}"
                tw = len(lbl) * 11
                draw1.rectangle([cx - tw//2 - 6, cy - 20, cx + tw//2 + 6, cy + 12], fill=(15, 23, 42, 230), outline=(fill_color[0], fill_color[1], fill_color[2], 255), width=2)
                draw1.text((cx - tw//2, cy - 16), lbl, font=self.font_badge, fill=(255, 255, 255, 255))
            elif mid >= 11:
                q_num = mid - 10
                detected_anchors.append(mid)
                draw1.polygon(pts_poly, outline=(16, 185, 129, 255), width=3)
                for p in pts_poly:
                    draw1.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(255, 255, 255, 255), outline=(16, 185, 129, 255))
                lbl = f"Вопрос #{q_num} (ID {mid})"
                tw = len(lbl) * 10
                draw1.rectangle([cx - tw//2 - 6, cy - 18, cx + tw//2 + 6, cy + 12], fill=(6, 78, 59, 230), outline=(52, 211, 153, 255), width=2)
                draw1.text((cx - tw//2, cy - 15), lbl, font=self.font_small, fill=(255, 255, 255, 255))

        # Top banner with detection stats
        banner_h = 70
        draw1.rectangle([0, 0, w_orig, banner_h], fill=(15, 23, 42, 240))
        banner_text = f"Шаг 1: Детекция ArUco маркеров • Найдено {len(marker_map)} маркеров (Углы: {len(detected_corners)}/4, Вопросы: {len(detected_anchors)}) • Время: {timings['markers']}мс"
        draw1.text((20, 18), banner_text, font=self.font_body, fill=(248, 250, 252, 255))

        vis_step1 = cv2.cvtColor(np.array(vis_step1_pil), cv2.COLOR_RGB2BGR)
        step1_path = blank_out_dir / "step1_markers.jpg"
        cv2.imwrite(str(step1_path), vis_step1, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # ----------------------------------------------------
        # STEP 2: Perspective Rectification
        # ----------------------------------------------------
        t0 = time.perf_counter()
        rectified_bgr, rect_success, rect_method = self.scanner.rectify_sheet(raw_bgr)
        t_rect = (time.perf_counter() - t0) * 1000.0
        timings["rectification"] = round(t_rect, 1)

        vis_step2 = rectified_bgr.copy()
        vis_step2_pil = Image.fromarray(cv2.cvtColor(vis_step2, cv2.COLOR_BGR2RGB))
        draw2 = ImageDraw.Draw(vis_step2_pil, "RGBA")

        # Overlay canonical bounding box guides
        draw2.rectangle([60, 60, 2040, 2910], outline=(59, 130, 246, 200), width=4)
        # Corner markers canonical targets
        corner_boxes = [
            (60, 60, 200, 200, "M0 TL"),
            (1900, 60, 2040, 200, "M1 TR"),
            (60, 2770, 200, 2910, "M2 BL"),
            (1900, 2770, 2040, 2910, "M3 BR"),
        ]
        for x1, y1, x2, y2, tag in corner_boxes:
            draw2.rectangle([x1, y1, x2, y2], outline=(234, 179, 8, 220), width=3)
            draw2.text((x1 + 8, y1 + 8), tag, font=self.font_badge, fill=(234, 179, 8, 255))

        # Top banner
        draw2.rectangle([0, 0, CANVAS_W, 90], fill=(15, 23, 42, 230))
        banner_text2 = f"Шаг 2: Выравнивание перспективы • Канонический размер 2100×2970 px • Метод: {rect_method} • Время: {timings['rectification']}мс"
        draw2.text((40, 25), banner_text2, font=self.font_title, fill=(248, 250, 252, 255))

        vis_step2 = cv2.cvtColor(np.array(vis_step2_pil), cv2.COLOR_RGB2BGR)
        step2_path = blank_out_dir / "step2_rectified.jpg"
        cv2.imwrite(str(step2_path), vis_step2, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # ----------------------------------------------------
        # STEP 3: Grid Snapping & Localization
        # ----------------------------------------------------
        t0 = time.perf_counter()
        # Note: extract_cells modifies rectified_bgr for border suppression in-place,
        # so we pass a clean copy to observe border suppression cleanly at Step 4!
        clean_rectified = rectified_bgr.copy()
        name_cells, question_rows = self.scanner.extract_cells(clean_rectified)
        t_grid = (time.perf_counter() - t0) * 1000.0
        timings["grid_snapping"] = round(t_grid, 1)

        vis_step3 = rectified_bgr.copy()
        vis_step3_pil = Image.fromarray(cv2.cvtColor(vis_step3, cv2.COLOR_BGR2RGB))
        draw3 = ImageDraw.Draw(vis_step3_pil, "RGBA")

        # Draw student name cells
        for cell in name_cells:
            x, y, w, h = cell["bbox"]
            draw3.rectangle([x, y, x + w, y + h], outline=(6, 182, 212, 255), width=3)
            draw3.rectangle([x, y, x + 34, y + 26], fill=(6, 182, 212, 230))
            draw3.text((x + 4, y + 3), f"N{cell['cell_idx']}", font=self.font_small, fill=(255, 255, 255, 255))

        total_q_cells = 0
        for q in question_rows:
            q_num = q["q_num"]
            for cell in q["cells"]:
                total_q_cells += 1
                x, y, w, h = cell["bbox"]
                draw3.rectangle([x, y, x + w, y + h], outline=(34, 197, 94, 255), width=3)
                draw3.rectangle([x, y, x + 44, y + 26], fill=(34, 197, 94, 230))
                draw3.text((x + 3, y + 3), f"Q{q_num}-{cell['cell_idx']}", font=self.font_small, fill=(255, 255, 255, 255))

        draw3.rectangle([0, 0, CANVAS_W, 90], fill=(15, 23, 42, 230))
        banner_text3 = f"Шаг 3: Локализация сетки шакмаков • 15 ячеек ФИО + {total_q_cells} ячеек вопросов ({len(question_rows)} строк) • Время: {timings['grid_snapping']}мс"
        draw3.text((40, 25), banner_text3, font=self.font_title, fill=(248, 250, 252, 255))

        vis_step3 = cv2.cvtColor(np.array(vis_step3_pil), cv2.COLOR_RGB2BGR)
        step3_path = blank_out_dir / "step3_grid_cells.jpg"
        cv2.imwrite(str(step3_path), vis_step3, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # ----------------------------------------------------
        # STEP 4: Printed Border Suppression (Paper White Ink Isolation)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        suppressed_bgr, final_border_mask, suppressed_pixels_count = self.scanner.suppress_grid_borders(
            clean_rectified,
            name_cells=name_cells,
            question_rows=question_rows,
            border_color=(255, 255, 255),
            safe_margin_pct=8,
        )
        t_border = (time.perf_counter() - t0) * 1000.0
        timings["border_suppression"] = round(t_border, 1)

        vis_step4 = suppressed_bgr.copy()
        vis_step4_pil = Image.fromarray(cv2.cvtColor(vis_step4, cv2.COLOR_BGR2RGB))
        draw4 = ImageDraw.Draw(vis_step4_pil, "RGBA")
        draw4.rectangle([0, 0, CANVAS_W, 90], fill=(15, 23, 42, 230))
        banner_text4 = f"Шаг 4: Очистка типографских рамок • Подавлено {suppressed_pixels_count:,} пикселей линий • Время: {timings['border_suppression']}мс"
        draw4.text((40, 25), banner_text4, font=self.font_title, fill=(248, 250, 252, 255))

        vis_step4 = cv2.cvtColor(np.array(vis_step4_pil), cv2.COLOR_RGB2BGR)
        step4_path = blank_out_dir / "step4_border_suppressed.jpg"
        cv2.imwrite(str(step4_path), vis_step4, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # ----------------------------------------------------
        # STEP 5: Character Recognition & Scoring
        # ----------------------------------------------------
        t0 = time.perf_counter()

        # Load Ground Truth if available
        gt_path = WORKSPACE_DIR / "benchmark" / "ground_truth" / f"{blank_id}.json"
        gt_data = None
        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

        gt_student_name = gt_data.get("student_name", {}).get("text", "") if gt_data else ""
        gt_questions_map = {}
        if gt_data:
            for gq in gt_data.get("questions", []):
                gt_questions_map[gq["q_num"]] = gq

        all_cells_payload = []
        recognized_student_chars = []
        name_results = []

        vis_step5 = suppressed_bgr.copy()
        vis_step5_pil = Image.fromarray(cv2.cvtColor(vis_step5, cv2.COLOR_BGR2RGB))
        draw5 = ImageDraw.Draw(vis_step5_pil, "RGBA")

        # Classify Student Name Cells
        for cell in name_cells:
            x, y, w, h = cell["bbox"]
            raw_cell_crop = rectified_bgr[y:y+h, x:x+w]
            cell_crop = suppressed_bgr[y:y+h, x:x+w]
            res = self.scanner.classify_cell(raw_cell_crop)

            pred_char = res["char"]
            conf = res["confidence"]
            is_empty = res["is_empty"]

            # Ground truth match check
            gt_cell_char = ""
            gt_is_empty = True
            if gt_data and "cells" in gt_data.get("student_name", {}):
                idx = cell["cell_idx"] - 1
                if idx < len(gt_data["student_name"]["cells"]):
                    gt_c = gt_data["student_name"]["cells"][idx]
                    gt_cell_char = gt_c.get("char", "")
                    gt_is_empty = gt_c.get("is_empty", True)

            char_match = (pred_char == gt_cell_char) if gt_data else None

            # Crop thumbnail base64
            crop_b64 = cv2_to_b64(raw_cell_crop, quality=85, max_dim=80)
            suppressed_crop_b64 = cv2_to_b64(cell_crop, quality=85, max_dim=80)

            cell_info = {
                "section": "student_name",
                "label": f"ФИО-{cell['cell_idx']}",
                "q_num": 0,
                "cell_idx": cell["cell_idx"],
                "bbox": [int(x), int(y), int(w), int(h)],
                "char": pred_char,
                "confidence": conf,
                "is_empty": is_empty,
                "top3": res["top3"],
                "top_items": res.get("top_items", []),
                "gt_char": gt_cell_char,
                "gt_is_empty": gt_is_empty,
                "match": char_match,
                "is_tatar_special": pred_char in TATAR_SPECIFIC_LETTERS,
                "crop_b64": crop_b64,
                "suppressed_crop_b64": suppressed_crop_b64,
            }
            all_cells_payload.append(cell_info)
            name_results.append(cell_info)

            if not is_empty:
                recognized_student_chars.append(pred_char)
                self.scanner._draw_cell_badge(draw5, x, y, w, h, res, self.font_title)
            else:
                # Thin outline for empty cells
                draw5.rectangle([x, y, x + w, y + h], outline=(148, 163, 184, 120), width=1)

        recognized_student_name = "".join(recognized_student_chars).strip()

        # Classify Question Cells
        questions_payload = []
        total_recognized_letters = 0
        total_conf_sum = 0.0
        total_correct_chars = 0
        total_evaluated_chars = 0

        for q in question_rows:
            q_num = q["q_num"]
            q_chars = []
            q_cells_res = []
            gt_q_obj = gt_questions_map.get(q_num, {})
            gt_q_text = gt_q_obj.get("text", "")
            gt_q_cells = gt_q_obj.get("cells", [])

            for cell in q["cells"]:
                x, y, w, h = cell["bbox"]
                raw_cell_crop = rectified_bgr[y:y+h, x:x+w]
                cell_crop = suppressed_bgr[y:y+h, x:x+w]
                res = self.scanner.classify_cell(raw_cell_crop)

                pred_char = res["char"]
                conf = res["confidence"]
                is_empty = res["is_empty"]

                # GT check
                gt_cell_char = ""
                gt_is_empty = True
                c_idx = cell["cell_idx"] - 1
                if c_idx < len(gt_q_cells):
                    gt_c = gt_q_cells[c_idx]
                    gt_cell_char = gt_c.get("char", "")
                    gt_is_empty = gt_c.get("is_empty", True)

                char_match = (pred_char == gt_cell_char) if gt_data else None
                if gt_data and not gt_is_empty:
                    total_evaluated_chars += 1
                    if char_match:
                        total_correct_chars += 1

                crop_b64 = cv2_to_b64(raw_cell_crop, quality=85, max_dim=80)
                suppressed_crop_b64 = cv2_to_b64(cell_crop, quality=85, max_dim=80)

                cell_info = {
                    "section": f"question_{q_num}",
                    "label": f"Q{q_num}-{cell['cell_idx']}",
                    "q_num": q_num,
                    "cell_idx": cell["cell_idx"],
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "char": pred_char,
                    "confidence": conf,
                    "is_empty": is_empty,
                    "top3": res["top3"],
                    "top_items": res.get("top_items", []),
                    "gt_char": gt_cell_char,
                    "gt_is_empty": gt_is_empty,
                    "match": char_match,
                    "is_tatar_special": pred_char in TATAR_SPECIFIC_LETTERS,
                    "crop_b64": crop_b64,
                    "suppressed_crop_b64": suppressed_crop_b64,
                }
                all_cells_payload.append(cell_info)
                q_cells_res.append(cell_info)

                if not is_empty:
                    q_chars.append(pred_char)
                    total_recognized_letters += 1
                    total_conf_sum += conf
                    self.scanner._draw_cell_badge(draw5, x, y, w, h, res, self.font_title)
                else:
                    draw5.rectangle([x, y, x + w, y + h], outline=(148, 163, 184, 120), width=1)

            q_text = "".join(q_chars)
            q_conf = round(total_conf_sum / max(1, total_recognized_letters), 1)
            q_exact_match = (q_text == gt_q_text) if gt_data else None
            q_cer = round(levenshtein_distance(q_text, gt_q_text) / max(1, len(gt_q_text)), 3) if gt_data and gt_q_text else 0.0

            questions_payload.append({
                "q_num": q_num,
                "text": q_text,
                "gt_text": gt_q_text,
                "exact_match": q_exact_match,
                "cer": q_cer,
                "confidence": q_conf,
                "cells": q_cells_res
            })

        t_ocr = (time.perf_counter() - t0) * 1000.0
        timings["recognition"] = round(t_ocr, 1)
        timings["total"] = round(sum(timings.values()), 1)

        # Form-level metrics
        num_questions = len(questions_payload)
        exact_matches = sum(1 for q in questions_payload if q["exact_match"] is True) if gt_data else 0
        fem_rate = round(exact_matches / max(1, num_questions) * 100.0, 1) if gt_data else None
        char_acc = round(total_correct_chars / max(1, total_evaluated_chars) * 100.0, 1) if total_evaluated_chars > 0 else 100.0
        avg_confidence = round(total_conf_sum / max(1, total_recognized_letters), 1) if total_recognized_letters > 0 else 100.0

        draw5.rectangle([0, 0, CANVAS_W, 90], fill=(15, 23, 42, 230))
        banner_text5 = f"Шаг 5: Распознавание букв • ФИО: '{recognized_student_name}' • Букв: {total_recognized_letters} • Точность: {char_acc}% • Время: {timings['recognition']}мс"
        draw5.text((40, 25), banner_text5, font=self.font_title, fill=(248, 250, 252, 255))

        vis_step5 = cv2.cvtColor(np.array(vis_step5_pil), cv2.COLOR_RGB2BGR)
        step5_path = blank_out_dir / "step5_annotated.jpg"
        cv2.imwrite(str(step5_path), vis_step5, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # Step 5 Recognition payload JSON
        recognition_payload = {
            "image_id": blank_id,
            "filename": image_path.name,
            "timings_ms": timings,
            "student_name": {
                "text": recognized_student_name,
                "gt_text": gt_student_name,
                "exact_match": recognized_student_name == gt_student_name if gt_data else None,
                "cells": name_results,
            },
            "questions": questions_payload,
            "metrics": {
                "field_exact_match_pct": fem_rate,
                "character_accuracy_pct": char_acc,
                "total_recognized_letters": total_recognized_letters,
                "average_confidence": avg_confidence,
                "evaluated_chars_count": total_evaluated_chars,
            },
            "rectification": {
                "method": rect_method,
                "success": rect_success,
                "canvas_size": [CANVAS_W, CANVAS_H],
            },
            "markers": {
                "total_detected": len(marker_map),
                "corners": detected_corners,
                "anchors": detected_anchors,
            }
        }
        with open(blank_out_dir / "step5_recognition.json", "w", encoding="utf-8") as f:
            json.dump(recognition_payload, f, ensure_ascii=False, indent=2, default=json_serializable)

        print(f"[+] Blank {blank_id} successfully processed in {timings['total']} ms!")
        print(f"    - Student Name: '{recognized_student_name}'")
        for q in questions_payload:
            gt_tag = f" (GT: '{q['gt_text']}' {'✓' if q['exact_match'] else '✗'})" if gt_data else ""
            print(f"    - Question {q['q_num']}: '{q['text']}' [conf: {q['confidence']}%]{gt_tag}")

        # Assemble high-quality base64 web assets for pipeline_inspector.html
        web_payload = {
            "id": blank_id,
            "filename": image_path.name,
            "title": f"{blank_id} ({num_questions} вопр.)",
            "timings_ms": timings,
            "metrics": recognition_payload["metrics"],
            "rectification_method": rect_method,
            "markers_summary": {
                "count": len(marker_map),
                "corners": detected_corners,
                "anchors": detected_anchors,
            },
            "student_name": recognition_payload["student_name"],
            "questions": questions_payload,
            "cells": all_cells_payload,
            "steps": {
                "step1": {
                    "title": "1. Детекция маркеров ArUco",
                    "subtitle": "Поиск 4 угловых маркеров листа и направляющих меток строк вопросов",
                    "image": cv2_to_b64(vis_step1, quality=78, max_dim=1400),
                    "raw_image": cv2_to_b64(raw_bgr, quality=75, max_dim=1400),
                    "time_ms": timings["markers"],
                    "stats": [
                        {"label": "Всего маркеров", "val": str(len(marker_map))},
                        {"label": "Угловые метки", "val": f"{len(detected_corners)}/4"},
                        {"label": "Метки вопросов", "val": f"{len(detected_anchors)}"},
                        {"label": "Размер фото", "val": f"{w_orig}×{h_orig}"},
                    ],
                    "explanation": (
                        "OpenCV <code>cv2.aruco.ArucoDetector</code> со словарем <code>DICT_4X4_50</code>. "
                        "Алгоритм выполняет субпиксельное уточнение углов (<code>CORNER_REFINE_SUBPIX</code>) с адаптивным порогом. "
                        "Если освещение неоднородно или углы обрезаны, пайплайн активирует многопроходный fallback с CLAHE-контрастированием."
                    )
                },
                "step2": {
                    "title": "2. Выравнивание перспективы (Rectification)",
                    "subtitle": "Геометрическая трансформация к каноническому холсту 2100×2970 px (A4 @ 254 DPI)",
                    "image": cv2_to_b64(vis_step2, quality=78, max_dim=1400),
                    "time_ms": timings["rectification"],
                    "stats": [
                        {"label": "Метод выравнивания", "val": rect_method},
                        {"label": "Канонический размер", "val": "2100×2970 px"},
                        {"label": "Интерполяция", "val": "INTER_LANCZOS4"},
                        {"label": "Статус геометрии", "val": "OK ✓" if rect_success else "Warning"},
                    ],
                    "explanation": (
                        f"Использован метод <b>{rect_method}</b>. "
                        "При обнаружении всех 4 угловых маркеров рассчитывается точная 3×3 матрица проективного преобразования <code>getPerspectiveTransform</code>. "
                        "В случае частичного закрытия углов используется устойчивый RANSAC с объединением маркеров вопросов и канонических координат."
                    )
                },
                "step3": {
                    "title": "3. Локализация сетки шакмаков (Grid Snapping)",
                    "subtitle": "Морфологический анализ печатных линий и адаптивная сегментация ячеек",
                    "image": cv2_to_b64(vis_step3, quality=78, max_dim=1400),
                    "time_ms": timings["grid_snapping"],
                    "stats": [
                        {"label": "Ячеек ФИО", "val": "15"},
                        {"label": "Ячеек ответов", "val": str(total_q_cells)},
                        {"label": "Строк вопросов", "val": str(len(question_rows))},
                        {"label": "Шаг сетки (Pitch)", "val": "~100 px"},
                    ],
                    "explanation": (
                        "Адаптивная бинаризация и морфологические ядра (40×1) и (1×40) выделяют горизонтальные и вертикальные печатные линии. "
                        "Кластеризация цепочек пиков (1D-профили проекций) динамически определяет границы шакмаков, компенсируя возможную усадку бумаги и микродеформации сканирования."
                    )
                },
                "step4": {
                    "title": "4. Очистка типографских рамок (Border Suppression)",
                    "subtitle": "Подавление черных печатных линий с сохранением рукописных штрихов",
                    "image": cv2_to_b64(vis_step4, quality=78, max_dim=1400),
                    "time_ms": timings["border_suppression"],
                    "stats": [
                        {"label": "Подавлено пикселей", "val": f"{suppressed_pixels_count:,}"},
                        {"label": "Дилатация маски", "val": "3×3 Rect"},
                        {"label": "Защита полей", "val": "Active (Margined)"},
                        {"label": "Цвет фона", "val": "RGB(255,255,255)"},
                    ],
                    "explanation": (
                        "Геометрически ограниченная очистка рамок: внутреннее ядро каждой клетки (84% площади) строго защищено от удаления. "
                        "Морфологические ядра увеличенного масштаба (80×1 и 1×75) детектируют исключительно многоячеечные печатные линии бланка. "
                        "Вертикальные и горизонтальные штрихи букв (Ш, Щ, И, П, Н, перекладины) никогда не стираются и сохраняют целостность на 100%."
                    )
                },
                "step5": {
                    "title": "5. Распознавание букв и скоринг (OCR & Confidence)",
                    "subtitle": "Нейросетевая классификация татарского алфавита (TatarOCRNet) и скоринг уверенности",
                    "image": cv2_to_b64(vis_step5, quality=78, max_dim=1400),
                    "time_ms": timings["recognition"],
                    "stats": [
                        {"label": "Распознано букв", "val": str(total_recognized_letters)},
                        {"label": "Точность (Accuracy)", "val": f"{char_acc}%" if gt_data else "N/A"},
                        {"label": "Средняя уверенность", "val": f"{avg_confidence}%"},
                        {"label": "Exact Match (FEM)", "val": f"{fem_rate}%" if fem_rate is not None else "N/A"},
                    ],
                    "explanation": (
                        "Каждая ячейка нормализуется (инвариантность к масштабу 72% заполнения, центрирование на 64×64, контрастирование перцентилями). "
                        "Сверточная сеть <code>TatarOCRNet</code> (39 классов татарского верхнего регистра) вычисляет распределение Softmax. "
                        "Шкала уверенности: <span class='text-emerald-500 font-bold'>Зеленый ≥80%</span>, "
                        "<span class='text-amber-500 font-bold'>Желтый 50–79%</span>, "
                        "<span class='text-rose-500 font-bold'>Красный <50%</span>."
                    )
                }
            }
        }

        return web_payload


def generate_inspector_html(all_blanks_data: dict, output_html_path: Path):
    """Generate self-contained modern HTML interactive inspector."""
    json_data_str = json.dumps(all_blanks_data, ensure_ascii=False, default=json_serializable)

    html_content = f"""<!DOCTYPE html>
<html lang="ru" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tatar OCR • Интерактивный инспектор пайплайна</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card: #111827;
      --card-border: #1f293d;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
    }}
    html.light {{
      --bg: #f8fafc;
      --card: #ffffff;
      --card-border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --accent: #2563eb;
    }}
    body {{
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    .custom-scrollbar::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    .custom-scrollbar::-webkit-scrollbar-track {{
      background: rgba(15, 23, 42, 0.4);
    }}
    .custom-scrollbar::-webkit-scrollbar-thumb {{
      background: rgba(148, 163, 184, 0.3);
      border-radius: 3px;
    }}
    .custom-scrollbar::-webkit-scrollbar-thumb:hover {{
      background: rgba(148, 163, 184, 0.5);
    }}
    .lens-container {{
      position: relative;
      overflow: hidden;
      cursor: crosshair;
    }}
  </style>
</head>
<body class="min-h-screen flex flex-col selection:bg-blue-600 selection:text-white">

  <!-- Header -->
  <header class="border-b border-[var(--card-border)] bg-[var(--card)]/90 backdrop-blur sticky top-0 z-50 px-4 lg:px-8 py-3">
    <div class="max-w-[1920px] mx-auto flex flex-wrap items-center justify-between gap-4">
      
      <!-- Logo & Title -->
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-emerald-400 flex items-center justify-center text-white font-black text-xl shadow-lg shadow-blue-500/20">
          Ә
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="font-bold text-lg text-[var(--text)] tracking-tight">Tatar OCR Pipeline Inspector</h1>
            <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">v2.4 Blank Engine</span>
          </div>
          <p class="text-xs text-[var(--text-muted)]">Пошаговый визуализатор детекции, ректификации и OCR распознавания</p>
        </div>
      </div>

      <!-- Blank Selector Pills -->
      <div class="flex items-center gap-2 bg-[var(--bg)] p-1.5 rounded-xl border border-[var(--card-border)]">
        <span class="text-xs font-medium text-[var(--text-muted)] px-2">Бланк:</span>
        <div id="blank-selector" class="flex gap-1.5">
          <!-- Populated by JS -->
        </div>
      </div>

      <!-- Telemetry Badge & Theme Switcher -->
      <div class="flex items-center gap-3">
        <div id="telemetry-badge" class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--bg)] border border-[var(--card-border)] text-xs">
          <span class="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span class="text-[var(--text-muted)]">Общее время:</span>
          <span id="header-total-time" class="font-bold text-emerald-400">0 мс</span>
        </div>

        <button id="theme-toggle" class="p-2 rounded-lg bg-[var(--bg)] border border-[var(--card-border)] text-[var(--text-muted)] hover:text-[var(--text)] transition" title="Переключить тему">
          <svg id="theme-icon-moon" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>
      </div>

    </div>
  </header>

  <!-- Interactive 5-Step Progress Bar -->
  <nav class="bg-[var(--card)]/60 border-b border-[var(--card-border)] px-4 lg:px-8 py-2.5">
    <div class="max-w-[1920px] mx-auto flex items-center justify-between gap-2 overflow-x-auto custom-scrollbar pb-1">
      <div id="stepper-buttons" class="flex items-center gap-2">
        <!-- 5 Steps populated by JS -->
      </div>
      
      <!-- Stepper Prev/Next controls -->
      <div class="flex items-center gap-2 shrink-0">
        <button id="prev-step-btn" class="px-3 py-1.5 rounded-lg border border-[var(--card-border)] bg-[var(--card)] hover:bg-slate-800 text-xs font-medium flex items-center gap-1 transition disabled:opacity-40 disabled:cursor-not-allowed">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
          <span>Назад</span>
        </button>
        <button id="next-step-btn" class="px-3 py-1.5 rounded-lg border border-blue-500/30 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold flex items-center gap-1 transition shadow-lg shadow-blue-500/20 disabled:opacity-40 disabled:cursor-not-allowed">
          <span>Далее</span>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
        </button>
      </div>
    </div>
  </nav>

  <!-- Main Work Area -->
  <main class="flex-1 max-w-[1920px] w-full mx-auto p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

    <!-- Left / Center: Interactive Visual Display (7 cols on large screens) -->
    <section class="lg:col-span-7 flex flex-col gap-4">
      
      <!-- Viewer Controls Toolbar -->
      <div class="flex flex-wrap items-center justify-between gap-3 bg-[var(--card)] p-3 rounded-xl border border-[var(--card-border)]">
        <div class="flex items-center gap-2">
          <span id="step-number-pill" class="px-2.5 py-0.5 rounded-md bg-blue-500/20 text-blue-400 font-bold text-xs border border-blue-500/30">ШАГ 1</span>
          <h2 id="step-title-display" class="font-bold text-sm lg:text-base text-[var(--text)]">Загрузка...</h2>
        </div>

        <!-- Zoom / View options -->
        <div class="flex items-center gap-2 text-xs">
          <!-- Step 1 Overlay Toggle -->
          <button id="toggle-raw-btn" class="hidden px-2.5 py-1 rounded bg-[var(--bg)] border border-[var(--card-border)] text-[var(--text-muted)] hover:text-white transition">
            Исходное фото
          </button>
          
          <button id="zoom-in-btn" class="p-1.5 rounded bg-[var(--bg)] border border-[var(--card-border)] hover:bg-slate-800 transition" title="Увеличить">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v12m6-6H6"/></svg>
          </button>
          <button id="zoom-out-btn" class="p-1.5 rounded bg-[var(--bg)] border border-[var(--card-border)] hover:bg-slate-800 transition" title="Уменьшить">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 12H6"/></svg>
          </button>
          <button id="zoom-reset-btn" class="px-2.5 py-1 rounded bg-[var(--bg)] border border-[var(--card-border)] text-[var(--text-muted)] hover:text-white transition">
            Сброс
          </button>
          <span id="zoom-level-text" class="text-[var(--text-muted)] w-12 text-center">100%</span>
        </div>
      </div>

      <!-- Main Canvas Container -->
      <div class="relative bg-slate-950 rounded-2xl border border-[var(--card-border)] overflow-hidden min-h-[580px] max-h-[820px] flex items-center justify-center p-2 group shadow-2xl">
        <div id="viewport-stage" class="relative overflow-auto max-w-full max-h-[800px] w-full flex items-center justify-center custom-scrollbar">
          <img id="main-step-img" src="" alt="Текущий этап пайплайна" class="max-w-full h-auto object-contain transition-transform duration-150 rounded shadow-lg select-none" />
        </div>
        
        <!-- Loading spinner overlay -->
        <div id="img-loader" class="absolute inset-0 bg-slate-950/80 flex items-center justify-center hidden z-20 backdrop-blur-sm">
          <div class="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        </div>
      </div>

      <!-- Step Subtitle & Meta -->
      <p id="step-subtitle-display" class="text-xs text-[var(--text-muted)] px-1">Подзаголовок этапа</p>
    </section>

    <!-- Right: Computer Vision Telemetry & Interactive Cards (5 cols) -->
    <section class="lg:col-span-5 flex flex-col gap-5">

      <!-- Stage Telemetry & Metric Cards -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3" id="step-stats-grid">
        <!-- 4 stats cards populated by JS -->
      </div>

      <!-- Stage CV Explanation Card -->
      <div class="bg-[var(--card)] rounded-xl border border-[var(--card-border)] p-5 flex flex-col gap-3 shadow-lg">
        <div class="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-400">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          <span>Механика компьютерного зрения</span>
        </div>
        <p id="step-explanation-text" class="text-xs lg:text-sm text-[var(--text)] leading-relaxed">
          Описание алгоритма...
        </p>
      </div>

      <!-- OCR Results Summary Box (Always visible or on Step 5) -->
      <div class="bg-[var(--card)] rounded-xl border border-[var(--card-border)] p-5 flex flex-col gap-3 shadow-lg">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-400">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            <span>Распознанный текст бланка</span>
          </div>
          <span id="ocr-acc-badge" class="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/30">
            Точность: 100%
          </span>
        </div>

        <!-- Student Name Row -->
        <div class="bg-[var(--bg)] p-3 rounded-lg border border-[var(--card-border)] flex items-center justify-between text-xs">
          <div>
            <span class="text-[var(--text-muted)] block text-[10px] uppercase font-bold">Фамилия / Имя учащегося:</span>
            <span id="summary-student-name" class="font-mono text-sm font-bold text-white">—</span>
          </div>
          <div id="summary-name-gt-badge" class="text-[11px] text-[var(--text-muted)]"></div>
        </div>

        <!-- Questions List -->
        <div id="summary-questions-list" class="flex flex-col gap-2 max-h-[180px] overflow-y-auto custom-scrollbar pr-1">
          <!-- Populated by JS -->
        </div>
      </div>

      <!-- Gallery of Cells Explorer (Visible especially on Step 5, switchable) -->
      <div class="bg-[var(--card)] rounded-xl border border-[var(--card-border)] p-4 flex flex-col gap-3 shadow-lg">
        <div class="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--card-border)] pb-2.5">
          <div class="flex items-center gap-2">
            <h3 class="font-bold text-sm text-[var(--text)]">Галерея ячеек</h3>
            <span id="cells-total-count" class="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-[var(--text-muted)] font-mono">0</span>
          </div>

          <!-- Cell Filters -->
          <div class="flex items-center gap-1 text-[11px]" id="cell-filters">
            <button data-filter="all" class="px-2 py-1 rounded bg-blue-600 text-white font-medium filter-btn">Все</button>
            <button data-filter="chars" class="px-2 py-1 rounded bg-[var(--bg)] text-[var(--text-muted)] hover:text-white filter-btn">С буквами</button>
            <button data-filter="tatar" class="px-2 py-1 rounded bg-[var(--bg)] text-[var(--text-muted)] hover:text-white filter-btn">Татарские (Ә Ө Ү Җ Ң Һ)</button>
            <button data-filter="low_conf" class="px-2 py-1 rounded bg-[var(--bg)] text-[var(--text-muted)] hover:text-white filter-btn">&lt;80% Уверенность</button>
          </div>
        </div>

        <!-- Cells Grid Gallery Container -->
        <div id="cells-gallery-grid" class="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-2.5 max-h-[360px] overflow-y-auto custom-scrollbar pr-1">
          <!-- Cell cards populated by JS -->
        </div>
      </div>

    </section>

  </main>

  <!-- Embedded Pipeline Data -->
  <script id="pipeline-data" type="application/json">
{json_data_str}
  </script>

  <!-- Inspector Interactive Script -->
  <script>
    (function() {{
      const dataScript = document.getElementById('pipeline-data');
      let pipelineData = {{}};
      try {{
        pipelineData = JSON.parse(dataScript.textContent);
      }} catch (e) {{
        console.error('Failed to parse pipeline data', e);
      }}

      const blankKeys = Object.keys(pipelineData);
      if (blankKeys.length === 0) {{
        document.body.innerHTML = "<div class='p-8 text-center text-red-500'>Нет данных для отображения!</div>";
        return;
      }}

      let currentBlankId = blankKeys[0];
      let currentStepKey = 'step1';
      let currentZoom = 1.0;
      let showingRawStep1 = false;
      let activeCellFilter = 'all';

      const stepOrder = ['step1', 'step2', 'step3', 'step4', 'step5'];
      const stepLabels = [
        '1. Маркеры ArUco',
        '2. Выравнивание',
        '3. Локализация сетки',
        '4. Очистка рамок',
        '5. Распознавание букв'
      ];

      // DOM Elements
      const blankSelectorContainer = document.getElementById('blank-selector');
      const stepperContainer = document.getElementById('stepper-buttons');
      const prevStepBtn = document.getElementById('prev-step-btn');
      const nextStepBtn = document.getElementById('next-step-btn');
      const mainStepImg = document.getElementById('main-step-img');
      const stepNumberPill = document.getElementById('step-number-pill');
      const stepTitleDisplay = document.getElementById('step-title-display');
      const stepSubtitleDisplay = document.getElementById('step-subtitle-display');
      const stepStatsGrid = document.getElementById('step-stats-grid');
      const stepExplanationText = document.getElementById('step-explanation-text');
      const headerTotalTime = document.getElementById('header-total-time');
      const toggleRawBtn = document.getElementById('toggle-raw-btn');
      const zoomInBtn = document.getElementById('zoom-in-btn');
      const zoomOutBtn = document.getElementById('zoom-out-btn');
      const zoomResetBtn = document.getElementById('zoom-reset-btn');
      const zoomLevelText = document.getElementById('zoom-level-text');
      const summaryStudentName = document.getElementById('summary-student-name');
      const summaryNameGtBadge = document.getElementById('summary-name-gt-badge');
      const summaryQuestionsList = document.getElementById('summary-questions-list');
      const ocrAccBadge = document.getElementById('ocr-acc-badge');
      const cellsGalleryGrid = document.getElementById('cells-gallery-grid');
      const cellsTotalCount = document.getElementById('cells-total-count');
      const filterBtns = document.querySelectorAll('.filter-btn');
      const themeToggle = document.getElementById('theme-toggle');

      // Theme toggle handler
      themeToggle.addEventListener('click', () => {{
        document.documentElement.classList.toggle('dark');
        document.documentElement.classList.toggle('light');
      }});

      // Populate Blank Switcher
      function initBlankSelector() {{
        blankSelectorContainer.innerHTML = '';
        blankKeys.forEach(bId => {{
          const item = pipelineData[bId];
          const btn = document.createElement('button');
          btn.className = `px-3 py-1 rounded-lg text-xs font-semibold transition border ${{
            bId === currentBlankId
              ? 'bg-blue-600 text-white border-blue-500 shadow-md shadow-blue-500/20'
              : 'bg-transparent text-[var(--text-muted)] border-transparent hover:text-white hover:bg-slate-800'
          }}`;
          btn.textContent = bId;
          btn.title = item.title || bId;
          btn.addEventListener('click', () => {{
            currentBlankId = bId;
            currentZoom = 1.0;
            showingRawStep1 = false;
            initBlankSelector();
            renderStep();
            renderSummaryCards();
            renderCellsGallery();
          }});
          blankSelectorContainer.appendChild(btn);
        }});
      }}

      // Populate 5 Stepper Buttons
      function initStepper() {{
        stepperContainer.innerHTML = '';
        stepOrder.forEach((sKey, idx) => {{
          const btn = document.createElement('button');
          const isActive = sKey === currentStepKey;
          btn.className = `px-3.5 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition flex items-center gap-1.5 border ${{
            isActive
              ? 'bg-blue-600/20 text-blue-400 border-blue-500/50 shadow-sm'
              : 'bg-[var(--card)] text-[var(--text-muted)] border-[var(--card-border)] hover:bg-slate-800 hover:text-[var(--text)]'
          }}`;
          btn.innerHTML = `
            <span class="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${{isActive ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300'}}">${{idx + 1}}</span>
            <span>${{stepLabels[idx]}}</span>
          `;
          btn.addEventListener('click', () => {{
            currentStepKey = sKey;
            showingRawStep1 = false;
            initStepper();
            renderStep();
          }});
          stepperContainer.appendChild(btn);
        }});

        const currentIndex = stepOrder.indexOf(currentStepKey);
        prevStepBtn.disabled = currentIndex === 0;
        nextStepBtn.disabled = currentIndex === stepOrder.length - 1;
      }}

      prevStepBtn.addEventListener('click', () => {{
        const idx = stepOrder.indexOf(currentStepKey);
        if (idx > 0) {{
          currentStepKey = stepOrder[idx - 1];
          showingRawStep1 = false;
          initStepper();
          renderStep();
        }}
      }});

      nextStepBtn.addEventListener('click', () => {{
        const idx = stepOrder.indexOf(currentStepKey);
        if (idx < stepOrder.length - 1) {{
          currentStepKey = stepOrder[idx + 1];
          showingRawStep1 = false;
          initStepper();
          renderStep();
        }}
      }});

      // Keyboard navigation
      window.addEventListener('keydown', (e) => {{
        if (e.key === 'ArrowRight' || e.key === 'PageDown') {{
          nextStepBtn.click();
        }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
          prevStepBtn.click();
        }}
      }});

      // Zoom Controls
      function updateZoom() {{
        mainStepImg.style.transform = `scale(${{currentZoom}})`;
        zoomLevelText.textContent = `${{Math.round(currentZoom * 100)}}%`;
      }}
      zoomInBtn.addEventListener('click', () => {{
        currentZoom = Math.min(3.0, currentZoom + 0.25);
        updateZoom();
      }});
      zoomOutBtn.addEventListener('click', () => {{
        currentZoom = Math.max(0.5, currentZoom - 0.25);
        updateZoom();
      }});
      zoomResetBtn.addEventListener('click', () => {{
        currentZoom = 1.0;
        updateZoom();
      }});

      // Toggle raw image on step 1
      toggleRawBtn.addEventListener('click', () => {{
        const blank = pipelineData[currentBlankId];
        const step = blank.steps.step1;
        showingRawStep1 = !showingRawStep1;
        if (showingRawStep1) {{
          mainStepImg.src = step.raw_image || step.image;
          toggleRawBtn.textContent = "Показать маркеры";
          toggleRawBtn.classList.add('bg-blue-600', 'text-white');
        }} else {{
          mainStepImg.src = step.image;
          toggleRawBtn.textContent = "Исходное фото";
          toggleRawBtn.classList.remove('bg-blue-600', 'text-white');
        }}
      }});

      // Render Current Step View
      function renderStep() {{
        const blank = pipelineData[currentBlankId];
        const step = blank.steps[currentStepKey];
        if (!step) return;

        const stepIdx = stepOrder.indexOf(currentStepKey) + 1;
        stepNumberPill.textContent = `ШАГ ${{stepIdx}} ИЗ 5`;
        stepTitleDisplay.textContent = step.title;
        stepSubtitleDisplay.textContent = step.subtitle;

        if (currentStepKey === 'step1') {{
          toggleRawBtn.classList.remove('hidden');
          toggleRawBtn.textContent = showingRawStep1 ? "Показать маркеры" : "Исходное фото";
          mainStepImg.src = showingRawStep1 ? (step.raw_image || step.image) : step.image;
        }} else {{
          toggleRawBtn.classList.add('hidden');
          mainStepImg.src = step.image;
        }}

        // Stats grid
        stepStatsGrid.innerHTML = '';
        (step.stats || []).forEach(st => {{
          const card = document.createElement('div');
          card.className = 'bg-[var(--card)] p-3 rounded-xl border border-[var(--card-border)] flex flex-col justify-between';
          card.innerHTML = `
            <span class="text-[10px] text-[var(--text-muted)] uppercase font-semibold tracking-wider">${{st.label}}</span>
            <span class="text-sm font-bold text-[var(--text)] mt-1 truncate" title="${{st.val}}">${{st.val}}</span>
          `;
          stepStatsGrid.appendChild(card);
        }});

        // Time stat badge
        const timeCard = document.createElement('div');
        timeCard.className = 'bg-blue-500/10 p-3 rounded-xl border border-blue-500/20 flex flex-col justify-between';
        timeCard.innerHTML = `
          <span class="text-[10px] text-blue-400 uppercase font-semibold tracking-wider">Время шага</span>
          <span class="text-sm font-bold text-blue-400 mt-1">${{step.time_ms}} мс</span>
        `;
        stepStatsGrid.appendChild(timeCard);

        // Explanation text
        stepExplanationText.innerHTML = step.explanation || '';

        // Reset zoom
        currentZoom = 1.0;
        updateZoom();
      }}

      // Render Summary Text & Question cards
      function renderSummaryCards() {{
        const blank = pipelineData[currentBlankId];
        headerTotalTime.textContent = `${{blank.timings_ms.total || 0}} мс`;

        // Student name
        const sName = blank.student_name.text || '—';
        summaryStudentName.textContent = sName;
        if (blank.student_name.gt_text) {{
          const isMatch = sName === blank.student_name.gt_text;
          summaryNameGtBadge.innerHTML = `<span class="${{isMatch ? 'text-emerald-400' : 'text-rose-400'}} font-semibold">GT: '${{blank.student_name.gt_text}}' ${{isMatch ? '✓' : '✗'}}</span>`;
        }} else {{
          summaryNameGtBadge.textContent = '';
        }}

        // Questions list
        summaryQuestionsList.innerHTML = '';
        (blank.questions || []).forEach(q => {{
          const row = document.createElement('div');
          row.className = 'bg-[var(--bg)] p-2.5 rounded-lg border border-[var(--card-border)] flex items-center justify-between text-xs';
          
          let gtBadge = '';
          if (q.gt_text) {{
            const isMatch = q.exact_match;
            gtBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold ${{isMatch ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}}">
              ${{isMatch ? 'Точно ✓' : `GT: '${{q.gt_text}}' ✗`}}
            </span>`;
          }}

          row.innerHTML = `
            <div class="flex items-center gap-2">
              <span class="font-bold text-blue-400 w-8">В${{q.q_num}}:</span>
              <span class="font-mono text-sm font-bold tracking-wider text-white">${{q.text || '—'}}</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-[11px] text-[var(--text-muted)] font-mono">${{q.confidence}}%</span>
              ${{gtBadge}}
            </div>
          `;
          summaryQuestionsList.appendChild(row);
        }});

        // Overall accuracy badge
        const acc = blank.metrics.character_accuracy_pct;
        if (acc !== null && acc !== undefined) {{
          ocrAccBadge.textContent = `Точность: ${{acc}}%`;
          ocrAccBadge.className = `text-xs px-2 py-0.5 rounded font-bold border ${{
            acc >= 90 ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' :
            acc >= 75 ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' :
            'bg-rose-500/20 text-rose-400 border-rose-500/30'
          }}`;
        }} else {{
          ocrAccBadge.textContent = `Уверенность: ${{blank.metrics.average_confidence}}%`;
        }}
      }}

      // Filter buttons logic
      filterBtns.forEach(btn => {{
        btn.addEventListener('click', () => {{
          filterBtns.forEach(b => {{
            b.classList.remove('bg-blue-600', 'text-white');
            b.classList.add('bg-[var(--bg)]', 'text-[var(--text-muted)]');
          }});
          btn.classList.add('bg-blue-600', 'text-white');
          btn.classList.remove('bg-[var(--bg)]', 'text-[var(--text-muted)]');
          activeCellFilter = btn.dataset.filter;
          renderCellsGallery();
        }});
      }});

      // Render Step 5 Cells Explorer Gallery
      function renderCellsGallery() {{
        const blank = pipelineData[currentBlankId];
        const cells = blank.cells || [];
        cellsTotalCount.textContent = cells.length;

        const filtered = cells.filter(c => {{
          if (activeCellFilter === 'chars') return !c.is_empty;
          if (activeCellFilter === 'tatar') return c.is_tatar_special;
          if (activeCellFilter === 'low_conf') return !c.is_empty && c.confidence < 80;
          return true;
        }});

        cellsGalleryGrid.innerHTML = '';
        if (filtered.length === 0) {{
          cellsGalleryGrid.innerHTML = "<div class='col-span-full py-8 text-center text-xs text-[var(--text-muted)]'>Нет подходящих ячеек по выбранному фильтру</div>";
          return;
        }}

        filtered.forEach(cell => {{
          const card = document.createElement('div');
          
          let confColor = 'text-slate-400 bg-slate-800 border-slate-700';
          if (!cell.is_empty) {{
            if (cell.confidence >= 80) confColor = 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
            else if (cell.confidence >= 50) confColor = 'text-amber-400 bg-amber-500/10 border-amber-500/30';
            else confColor = 'text-rose-400 bg-rose-500/10 border-rose-500/30';
          }}

          let gtStatusHtml = '';
          if (cell.match !== null && cell.match !== undefined && !cell.gt_is_empty) {{
            gtStatusHtml = cell.match
              ? `<span class="text-emerald-400 font-bold" title="Совпало с Ground Truth">✓</span>`
              : `<span class="text-rose-400 font-bold" title="GT: '${{cell.gt_char}}'">✗ '${{cell.gt_char}}'</span>`;
          }}

          const topCandidatesHtml = (cell.top3 || []).map(([ch, cf]) => `
            <div class="flex items-center justify-between text-[10px]">
              <span class="font-bold text-slate-300 font-mono">${{ch === ' ' ? '␣' : ch}}</span>
              <span class="text-[var(--text-muted)]">${{cf}}%</span>
            </div>
          `).join('');

          card.className = 'bg-[var(--bg)] p-2 rounded-lg border border-[var(--card-border)] flex flex-col gap-1.5 hover:border-blue-500/40 transition group';
          card.innerHTML = `
            <div class="flex items-center justify-between text-[10px] text-[var(--text-muted)]">
              <span class="font-mono font-semibold">${{cell.label}}</span>
              ${{gtStatusHtml}}
            </div>

            <div class="flex items-center gap-2">
              <div class="w-11 h-11 bg-white rounded border border-slate-700 flex items-center justify-center overflow-hidden shrink-0 group-hover:scale-105 transition-transform">
                <img src="${{cell.crop_b64}}" alt="Crop" class="max-w-full max-h-full object-contain" />
              </div>
              <div class="flex flex-col">
                <span class="text-base font-black font-mono ${{cell.is_empty ? 'text-slate-500' : 'text-white'}} leading-none">
                  ${{cell.is_empty ? '␣' : cell.char}}
                </span>
                <span class="text-[10px] px-1.5 py-0.5 rounded border ${{confColor}} font-mono font-bold mt-1">
                  ${{cell.is_empty ? 'Пусто' : `${{cell.confidence}}%`}}
                </span>
              </div>
            </div>

            <div class="border-t border-[var(--card-border)] pt-1 flex flex-col gap-0.5">
              ${{topCandidatesHtml}}
            </div>
          `;
          cellsGalleryGrid.appendChild(card);
        }});
      }}

      // Init on load
      initBlankSelector();
      initStepper();
      renderStep();
      renderSummaryCards();
      renderCellsGallery();
    }})();
  </script>

</body>
</html>
"""
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[✓] Generated interactive inspector HTML at: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(description="Tatar Blank OCR Pipeline Visualizer")
    parser.add_argument("--image", nargs="*", help="Path to input blank photo(s)")
    parser.add_argument("--all", action="store_true", help="Process all blanks in benchmark/images")
    parser.add_argument("--model", default="models/tatar_ocr_v2_handwritten39_int8.onnx", help="Path to OCR model weights (.pth or .onnx)")
    parser.add_argument("--output_dir", default="benchmark/reports/pipeline_inspector", help="Output directory for visual assets")
    parser.add_argument("--html", default="benchmark/reports/pipeline_inspector.html", help="Path for generated HTML inspector")
    args = parser.parse_args()

    # Determine which images to process
    if args.all:
        image_paths = sorted(list(WORKSPACE_DIR.glob("benchmark/images/blank_*.jpg")))
    elif args.image:
        image_paths = [Path(p) for p in args.image]
    else:
        # Default to the 3 canonical evaluation samples: blank_01, blank_03, blank_10
        candidates = ["blank_01.jpg", "blank_03.jpg", "blank_10.jpg"]
        image_paths = [WORKSPACE_DIR / "benchmark" / "images" / c for c in candidates]
        image_paths = [p for p in image_paths if p.exists()]

    if not image_paths:
        print("[!] No images found to process. Exiting.")
        sys.exit(1)

    print("==================================================================")
    print("Tatar OCR • Pipeline Visualizer & Step-by-Step Inspector")
    print(f"Model: {args.model}")
    print(f"Target Blanks ({len(image_paths)}): {[p.name for p in image_paths]}")
    print("==================================================================")

    scanner = BlankOCRScanner(model_path=args.model)
    visualizer = PipelineVisualizer(scanner=scanner)

    output_dir = WORKSPACE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_blanks_data = {}
    for img_p in image_paths:
        payload = visualizer.process_image(img_p, output_dir=output_dir)
        all_blanks_data[payload["id"]] = payload

    # Generate the main interactive HTML inspector in benchmark/reports/
    html_out = WORKSPACE_DIR / args.html
    generate_inspector_html(all_blanks_data, html_out)

    # Also export to caller's brain artifact directory if accessible:
    caller_brain_dir = Path(r"C:\Users\galee\.gemini\antigravity\brain\c5675fc0-8738-46fa-adcc-3b1ab2f5c3c2")
    try:
        if caller_brain_dir.exists():
            caller_html_out = caller_brain_dir / "pipeline_inspector.html"
            shutil.copy2(html_out, caller_html_out)
            print(f"[✓] Exported inspector to caller brain artifact: {caller_html_out}")
    except Exception as e:
        print(f"[-] Note: Could not copy to caller brain dir ({e}), saving to workspace and conversation artifact.")

    # Also export to current conversation brain artifact directory:
    current_brain_dir = Path(r"C:\Users\galee\.gemini\antigravity\brain\9c650274-add5-43e1-ba08-d81b65ad76b3")
    try:
        if current_brain_dir.exists():
            current_html_out = current_brain_dir / "pipeline_inspector.html"
            shutil.copy2(html_out, current_html_out)
            print(f"[✓] Exported inspector to conversation artifact: {current_html_out}")
    except Exception as e:
        print(f"[-] Note: Could not copy to conversation artifact ({e}).")


if __name__ == "__main__":
    main()

import os
import sys
import json
import base64
from pathlib import Path

WORKSPACE_DIR = Path(r"c:\Users\galee\PycharmProjects\tatar_ocr")
sys.path.insert(0, str(WORKSPACE_DIR))
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from tatar_ocr_dataset import TatarOCRNet
from tatar_ocr_augmentation import TATAR_UPPERCASE

CANVAS_W = 2100
CANVAS_H = 2970

DST_CORNERS = np.array([
    [60.0, 60.0],       # Marker 0 TL
    [2040.0, 60.0],     # Marker 1 TR
    [2040.0, 2910.0],   # Marker 3 BR
    [60.0, 2910.0],     # Marker 2 BL
], dtype=np.float32)


class BlankOCRScanner:
    def __init__(self, model_path="models/finetuned_uppercase39.pth", device="cpu"):
        self.device = torch.device(device)
        self.model = TatarOCRNet(num_classes=len(TATAR_UPPERCASE)).to(self.device)
        weights = WORKSPACE_DIR / model_path
        if not weights.exists():
            weights = WORKSPACE_DIR / "models" / "baseline_synthetic_uppercase39.pth"
        if weights.exists():
            self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f"BlankOCRScanner: Loaded OCR model from {weights}")
        else:
            print("BlankOCRScanner: Warning - model weights not found!")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

    def rectify_sheet(self, img_bgr):
        h_orig, w_orig = img_bgr.shape[:2]
        corners, ids, _ = self.detector.detectMarkers(img_bgr)

        marker_map = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                marker_map[int(mid)] = corners[i][0]

        has_all_corners = all(m in marker_map for m in [0, 1, 2, 3])

        if has_all_corners:
            src_pts = np.array([
                marker_map[0][0],  # TL of marker 0
                marker_map[1][1],  # TR of marker 1
                marker_map[3][2],  # BR of marker 3
                marker_map[2][3],  # BL of marker 2
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_pts, DST_CORNERS)
            warped = cv2.warpPerspective(img_bgr, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LANCZOS4)
            return warped, True, "ArUco 4-Point Geometry"

        # Fallback: Detect paper rectangle
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 180)
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > (h_orig * w_orig * 0.4):
                pts = approx.reshape(4, 2).astype(np.float32)
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                rect_pts = np.array([
                    pts[np.argmin(s)],
                    pts[np.argmin(diff)],
                    pts[np.argmax(s)],
                    pts[np.argmax(diff)]
                ], dtype=np.float32)
                dst = np.array([[0, 0], [CANVAS_W, 0], [CANVAS_W, CANVAS_H], [0, CANVAS_H]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(rect_pts, dst)
                warped = cv2.warpPerspective(img_bgr, M, (CANVAS_W, CANVAS_H))
                return warped, True, "Document Contour Fallback"

        warped = cv2.resize(img_bgr, (CANVAS_W, CANVAS_H))
        return warped, False, "Direct Scaling (No markers detected)"

    def extract_cells(self, rectified_bgr):
        gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        table_mask = cv2.bitwise_or(h_lines, v_lines)

        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 70 <= w <= 130 and 70 <= h <= 130 and (0.75 <= float(w)/h <= 1.3):
                if x >= 200:
                    detected_boxes.append((x, y, w, h))

        # 1. Student Name row: pitch 90px, 16 cells
        # Snap horizontal line around y=216
        name_cells = []
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        name_profile = sobel_y[200:245, 470:1910].mean(axis=1)
        name_top_y = 200 + int(np.argmax(name_profile))

        # Vertical dividers with ~90px pitch
        name_x_start = 474
        for i in range(16):
            cx = int(name_x_start + i * 89.9)
            name_cells.append({
                "section": "student_name",
                "cell_idx": i + 1,
                "bbox": (cx, name_top_y, 90, 96)
            })

        # 2. Questions: anchor to detected question ArUco markers (ID 11..18)
        corners_w, ids_w, _ = self.detector.detectMarkers(rectified_bgr)
        q_markers = {}
        if ids_w is not None:
            for i, mid in enumerate(ids_w.flatten()):
                if int(mid) >= 11:
                    q_markers[int(mid)] = corners_w[i][0]

        question_rows = []
        if q_markers:
            sorted_mids = sorted(q_markers.keys())
            for q_idx, mid in enumerate(sorted_mids):
                q_num = q_idx + 1
                my = float(q_markers[mid][:, 1].mean())
                # Find exact top line of answer strip near marker
                y_min, y_max = int(my - 40), int(my + 40)
                strip_sobel = sobel_y[y_min:y_max, 220:1020].mean(axis=1)
                snapped_q_y = y_min + int(np.argmax(strip_sobel))

                # Detect actual number of cells in this row (from 1 up to 12) based on vertical cell borders
                row_cell_count = 0
                for c_test in range(1, 13):
                    x_div = 220 + c_test * 100
                    div_strip = v_lines[snapped_q_y + 15 : snapped_q_y + 80, x_div - 10 : x_div + 10]
                    if np.sum(div_strip > 0) > 100:
                        row_cell_count = c_test
                    else:
                        break
                row_cell_count = max(1, min(row_cell_count if row_cell_count > 0 else 8, 12))

                cells = []
                for c_idx in range(row_cell_count):
                    cx = int(220 + c_idx * 100)
                    cells.append({
                        "section": f"question_{q_num}",
                        "q_num": q_num,
                        "cell_idx": c_idx + 1,
                        "bbox": (cx, snapped_q_y, 100, 95)
                    })
                question_rows.append({"q_num": q_num, "cells": cells, "marker_id": mid})
        else:
            # Fallback to estimated anchors if no question markers
            q_y_anchors = [(1, 474), (2, 753), (3, 1031), (4, 1310), (5, 1590), (6, 1870), (7, 2150), (8, 2430)]
            for q_num, est_y in q_y_anchors:
                cells = []
                for c_idx in range(8):
                    cx = int(220 + c_idx * 100)
                    cells.append({
                        "section": f"question_{q_num}",
                        "q_num": q_num,
                        "cell_idx": c_idx + 1,
                        "bbox": (cx, est_y, 100, 95)
                    })
                question_rows.append({"q_num": q_num, "cells": cells})

        return name_cells, question_rows

    def classify_cell(self, cell_bgr, margin_trim_pct=4):
        ch, cw = cell_bgr.shape[:2]
        if ch < 10 or cw < 10:
            return {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)]}

        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

        # Empty cell check: inspect central 70% region
        cx1, cx2 = int(cw * 0.15), int(cw * 0.85)
        cy1, cy2 = int(ch * 0.15), int(ch * 0.85)
        center_area = gray[cy1:cy2, cx1:cx2]

        bg_val = float(np.median(center_area))
        ink_mask = (center_area < (bg_val - 25)).astype(np.uint8)
        ink_pixels = np.sum(ink_mask)
        total_center_pixels = center_area.size
        ink_ratio = ink_pixels / float(total_center_pixels + 1e-5)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask)
        max_comp = max([s[cv2.CC_STAT_AREA] for s in stats[1:]], default=0)

        # Mark empty if stroke component is tiny or ink ratio is negligible
        if ink_ratio < 0.02 or max_comp < 35:
            return {
                "is_empty": True,
                "char": " ",
                "confidence": 100.0,
                "top3": [(" ", 100.0)]
            }

        # 1. Margin trim
        trim_y = int(ch * (margin_trim_pct / 100.0))
        trim_x = int(cw * (margin_trim_pct / 100.0))
        trimmed = cell_bgr[trim_y:ch - trim_y, trim_x:cw - trim_x]

        # 2. Square padding
        th, tw = trimmed.shape[:2]
        sq_size = max(th, tw)
        edge_pix = np.concatenate([
            trimmed[0, :, :], trimmed[-1, :, :],
            trimmed[:, 0, :], trimmed[:, -1, :]
        ], axis=0)
        bg_col = np.median(edge_pix, axis=0).astype(np.uint8)

        padded = np.full((sq_size, sq_size, 3), bg_col, dtype=np.uint8)
        py = (sq_size - th) // 2
        px = (sq_size - tw) // 2
        padded[py:py + th, px:px + tw] = trimmed

        # 3. Grayscale + Percentile contrast normalization
        gray_p = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        p_lo = np.percentile(gray_p, 2)
        p_hi = np.percentile(gray_p, 98)
        norm_gray = np.clip((gray_p.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)

        # 4. Resize to 64x64
        inp_64 = cv2.resize(norm_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # 5. Tensor inference
        t = ((inp_64.astype(np.float32) / 255.0) - 0.5) / 0.5
        tensor = torch.from_numpy(t).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, k=3)
        top_chars = [TATAR_UPPERCASE[i] for i in top_indices.cpu().numpy()]
        top_confs = (top_probs.cpu().numpy() * 100.0).tolist()

        return {
            "is_empty": False,
            "char": top_chars[0],
            "confidence": round(float(top_confs[0]), 1),
            "top3": list(zip(top_chars, [round(c, 1) for c in top_confs]))
        }

    def process_blank(self, img_bgr, annotate=True):
        rectified, success, method = self.rectify_sheet(img_bgr)
        name_cells, question_rows = self.extract_cells(rectified)

        annotated_img = rectified.copy() if annotate else None

        try:
            font_badge = ImageFont.truetype(str(WORKSPACE_DIR / "fonts" / "Rubik-Regular.ttf"), 36)
            font_badge_small = ImageFont.truetype(str(WORKSPACE_DIR / "fonts" / "Rubik-Regular.ttf"), 22)
        except Exception:
            try:
                font_badge = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
                font_badge_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
            except Exception:
                font_badge = font_badge_small = ImageFont.load_default()

        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(annotated_pil, "RGBA")

        # 1. Process Student Name
        student_name_chars = []
        name_results = []
        for cell in name_cells:
            x, y, w, h = cell["bbox"]
            cell_crop = rectified[y:y+h, x:x+w]
            res = self.classify_cell(cell_crop)
            cell["result"] = res
            name_results.append({
                "cell_idx": cell["cell_idx"],
                "char": res["char"],
                "confidence": res["confidence"],
                "is_empty": res["is_empty"],
                "bbox": cell["bbox"]
            })

            if not res["is_empty"]:
                student_name_chars.append(res["char"])
                self._draw_cell_badge(draw, x, y, w, h, res, font_badge)

        student_name_str = "".join(student_name_chars).strip()

        # 2. Process Questions
        questions_results = []
        total_recognized_letters = 0
        conf_sum = 0.0

        for q in question_rows:
            q_chars = []
            q_cells_res = []
            for cell in q["cells"]:
                x, y, w, h = cell["bbox"]
                cell_crop = rectified[y:y+h, x:x+w]
                res = self.classify_cell(cell_crop)
                cell["result"] = res
                q_cells_res.append({
                    "cell_idx": cell["cell_idx"],
                    "char": res["char"],
                    "confidence": res["confidence"],
                    "is_empty": res["is_empty"],
                    "bbox": cell["bbox"]
                })

                if not res["is_empty"]:
                    q_chars.append(res["char"])
                    total_recognized_letters += 1
                    conf_sum += res["confidence"]
                    self._draw_cell_badge(draw, x, y, w, h, res, font_badge)

            q_text = "".join(q_chars)
            questions_results.append({
                "q_num": q["q_num"],
                "text": q_text,
                "cells": q_cells_res
            })

        avg_conf = (conf_sum / max(1, total_recognized_letters)) if total_recognized_letters > 0 else 0.0
        draw.rectangle([(0, 0), (CANVAS_W, 50)], fill=(15, 23, 42, 230))
        banner_text = f"Tatar OCR Scanner • Name: '{student_name_str or 'N/A'}' • Recognized {total_recognized_letters} letters • Avg Confidence: {avg_conf:.1f}% • Alignment: {method}"
        draw.text((30, 8), banner_text, fill=(255, 255, 255), font=font_badge_small)

        final_annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

        return {
            "status": "success",
            "rectification_method": method,
            "student_name": student_name_str,
            "questions": questions_results,
            "stats": {
                "total_letters": total_recognized_letters,
                "avg_confidence": round(avg_conf, 1),
                "questions_count": len(questions_results)
            },
            "rectified_bgr": rectified,
            "annotated_bgr": final_annotated_bgr
        }

    def _draw_cell_badge(self, draw, x, y, w, h, res, font_badge):
        ch = res["char"]
        conf = res["confidence"]

        if conf >= 80.0:
            box_col = (34, 197, 94, 255)
            badge_bg = (34, 197, 94, 230)
        elif conf >= 50.0:
            box_col = (234, 179, 8, 255)
            badge_bg = (202, 138, 4, 230)
        else:
            box_col = (239, 68, 68, 255)
            badge_bg = (220, 38, 38, 230)

        draw.rectangle([(x, y), (x + w, y + h)], outline=box_col, width=4)
        badge_w, badge_h = 56, 38
        bx, by = x + w - badge_w + 4, y - 20
        draw.rounded_rectangle([(bx, by), (bx + badge_w, by + badge_h)], radius=6, fill=badge_bg)
        draw.text((bx + 10, by + 1), ch, fill=(255, 255, 255), font=font_badge)

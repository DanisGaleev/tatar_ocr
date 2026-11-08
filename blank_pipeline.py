import os
import sys
import json
import base64
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.stdout.reconfigure(encoding='utf-8')

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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

CANONICAL_ANCHORS = {
    11: np.array([130.0, 494.0], dtype=np.float32),
    12: np.array([130.0, 770.0], dtype=np.float32),
    13: np.array([128.0, 1045.0], dtype=np.float32),
    14: np.array([125.0, 1320.0], dtype=np.float32),
    15: np.array([125.0, 1595.0], dtype=np.float32),
    16: np.array([125.0, 1872.0], dtype=np.float32),
    17: np.array([125.0, 2150.0], dtype=np.float32),
    18: np.array([125.0, 2428.0], dtype=np.float32),
}


class BlankOCRScanner:
    def __init__(self, model_path="models/tatar_ocr_v2_handwritten39_int8.onnx", device="cpu"):
        self.device = torch.device(device)
        self.model_path = str(model_path)
        self.is_onnx = self.model_path.lower().endswith(".onnx")
        self.session = None
        self.model = None

        weights = Path(model_path)
        if not weights.is_absolute():
            weights = WORKSPACE_DIR / model_path

        if not weights.exists():
            if self.is_onnx:
                weights = WORKSPACE_DIR / "models" / "tatar_ocr_uppercase39_fp32.onnx"
            else:
                weights = WORKSPACE_DIR / "models" / "baseline_synthetic_uppercase39.pth"

        if self.is_onnx:
            if ort is None:
                raise ImportError("onnxruntime is not installed, but an .onnx model was specified.")
            if weights.exists():
                self.session = ort.InferenceSession(str(weights), providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                print(f"BlankOCRScanner: Loaded ONNX model from {weights}")
            else:
                print(f"BlankOCRScanner: Warning - ONNX model weights not found at {weights}!")
        else:
            if weights.exists():
                state_dict = torch.load(weights, map_location=self.device, weights_only=True)
                is_v3_ckpt = any(".se." in k or ".shortcut." in k for k in state_dict.keys())
                if is_v3_ckpt:
                    from v3.architecture import TatarOCRNetV3
                    self.model = TatarOCRNetV3(num_classes=len(TATAR_UPPERCASE)).to(self.device)
                else:
                    self.model = TatarOCRNet(num_classes=len(TATAR_UPPERCASE)).to(self.device)
                self.model.load_state_dict(state_dict)
                print(f"BlankOCRScanner: Loaded OCR model ({self.model.__class__.__name__}) from {weights}")
            else:
                self.model = TatarOCRNet(num_classes=len(TATAR_UPPERCASE)).to(self.device)
                print("BlankOCRScanner: Warning - model weights not found!")
            self.model.eval()

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 45
        self.detector_params.adaptiveThreshWinSizeStep = 4
        self.detector_params.adaptiveThreshConstant = 7
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def detect_markers_padded(self, img_bgr, pad=60):
        padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        corners, ids, _ = self.detector.detectMarkers(padded)
        marker_map = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                marker_map[int(mid)] = corners[i][0] - float(pad)

        # Multi-pass fallback for shadows/tilt/low contrast
        if len(marker_map) < 8:
            p2 = cv2.aruco.DetectorParameters()
            p2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            p2.adaptiveThreshWinSizeMin = 3
            p2.adaptiveThreshWinSizeMax = 45
            p2.adaptiveThreshWinSizeStep = 4
            p2.adaptiveThreshConstant = 12
            corners2, ids2, _ = cv2.aruco.ArucoDetector(self.aruco_dict, p2).detectMarkers(padded)
            if ids2 is not None:
                for i, mid in enumerate(ids2.flatten()):
                    mid_val = int(mid)
                    if mid_val not in marker_map:
                        marker_map[mid_val] = corners2[i][0] - float(pad)

        if len(marker_map) < 8:
            gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            corners_c, ids_c, _ = self.detector.detectMarkers(enhanced)
            if ids_c is not None:
                for i, mid in enumerate(ids_c.flatten()):
                    mid_val = int(mid)
                    if mid_val not in marker_map:
                        marker_map[mid_val] = corners_c[i][0] - float(pad)
        return marker_map

    def rectify_sheet(self, img_bgr):
        h_orig, w_orig = img_bgr.shape[:2]
        marker_map = self.detect_markers_padded(img_bgr, pad=60)

        # 1. Exact 4-Corner Rectification if all corners detected
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

        # 2. Multi-Marker Homography with RANSAC
        src_pts = []
        dst_pts = []
        if 0 in marker_map:
            src_pts.append(marker_map[0][0])
            dst_pts.append([60.0, 60.0])
        if 1 in marker_map:
            src_pts.append(marker_map[1][1])
            dst_pts.append([2040.0, 60.0])
        if 2 in marker_map:
            src_pts.append(marker_map[2][3])
            dst_pts.append([60.0, 2910.0])
        if 3 in marker_map:
            src_pts.append(marker_map[3][2])
            dst_pts.append([2040.0, 2910.0])

        for mid in sorted(marker_map.keys()):
            if mid in CANONICAL_ANCHORS:
                src_pts.append(marker_map[mid].mean(axis=0))
                dst_pts.append(CANONICAL_ANCHORS[mid])

        if len(src_pts) >= 4:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)

            quads = set()
            for pt in dst_arr:
                q = (0 if pt[0] < 1050 else 1) + (0 if pt[1] < 1485 else 2)
                quads.add(q)

            missing_both_top = (0 not in marker_map and 1 not in marker_map)
            missing_both_bot = (2 not in marker_map and 3 not in marker_map)
            unbalanced = len(quads) < 3 or missing_both_top or missing_both_bot

            if unbalanced:
                M_aff, _ = cv2.estimateAffinePartial2D(src_arr, dst_arr, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                warped = cv2.warpAffine(img_bgr, M_aff, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LANCZOS4)
                return warped, True, f"ArUco Similarity Transform ({len(src_pts)} anchors, unbalanced)"
            else:
                H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
                warped = cv2.warpPerspective(img_bgr, H, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LANCZOS4)
                return warped, True, f"ArUco Multi-Marker Homography ({len(src_pts)} anchors)"

        # 3. Fallback: Detect paper rectangle
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
        # 1. Detect question anchors FIRST before any image modification
        marker_map = self.detect_markers_padded(rectified_bgr, pad=40)

        gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)

        # Bridge small 1-2px breaks in printed lines
        h_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
        v_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))

        # Long horizontal morphological kernel to eliminate prompt text completely
        h_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (180, 1))
        v_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        h_lines = cv2.morphologyEx(h_closed, cv2.MORPH_OPEN, h_kernel_long)
        v_lines = cv2.morphologyEx(v_closed, cv2.MORPH_OPEN, v_kernel_long)

        # Grid lines detected from morphological kernels are used strictly for row/column grid snapping.
        # Handwritten ink and rectified_bgr remain 100% intact and untouched.

        # 2. Student Name row active snapping
        name_cells = []
        name_strip = h_lines[200:360, 450:1800]
        h_prof = name_strip.sum(axis=1)
        name_peaks = []
        if h_prof.max() > 0:
            for dy in range(2, len(h_prof) - 2):
                if h_prof[dy] > 0.15 * h_prof.max() and h_prof[dy] == max(h_prof[dy-2:dy+3]):
                    name_peaks.append(200 + dy)

        name_top_y, name_bot_y = 214, 306
        for i in range(len(name_peaks)):
            for j in range(i + 1, len(name_peaks)):
                if 75 <= (name_peaks[j] - name_peaks[i]) <= 110:
                    name_top_y, name_bot_y = name_peaks[i], name_peaks[j]
                    break
            if (name_bot_y - name_top_y) >= 75:
                break
        name_cell_h = name_bot_y - name_top_y

        v_name_strip = v_lines[name_top_y + 10 : name_bot_y - 10, 420:1850]
        v_prof = v_name_strip.sum(axis=0)
        v_peaks = []
        if v_prof.max() > 0:
            for dx in range(2, len(v_prof) - 2):
                if v_prof[dx] > 0.15 * v_prof.max() and v_prof[dx] == max(v_prof[dx-2:dx+3]):
                    v_peaks.append(420 + dx)

        clustered_name_v = []
        for p in v_peaks:
            if not clustered_name_v or (p - clustered_name_v[-1]) > 10:
                clustered_name_v.append(p)
            else:
                clustered_name_v[-1] = int(round((clustered_name_v[-1] + p) / 2.0))

        best_name_chain = []
        for i in range(len(clustered_name_v)):
            for j in range(i + 1, len(clustered_name_v)):
                pitch_cand = clustered_name_v[j] - clustered_name_v[i]
                if 75 <= pitch_cand <= 98:
                    chain = [clustered_name_v[i], clustered_name_v[j]]
                    expected_next = clustered_name_v[j] + pitch_cand
                    for k in range(j + 1, len(clustered_name_v)):
                        if abs(clustered_name_v[k] - expected_next) <= 12:
                            chain.append(clustered_name_v[k])
                            pitch_cand = (clustered_name_v[k] - chain[0]) / float(len(chain) - 1)
                            expected_next = clustered_name_v[k] + pitch_cand
                    if len(chain) > len(best_name_chain):
                        best_name_chain = chain

        if len(best_name_chain) >= 2:
            name_start_x = best_name_chain[0]
            name_pitch = (best_name_chain[-1] - best_name_chain[0]) / float(len(best_name_chain) - 1)
        else:
            name_start_x = 465
            name_pitch = 88.0

        for idx in range(15):
            cx = int(round(name_start_x + idx * name_pitch))
            name_cells.append({
                "section": "student_name",
                "cell_idx": idx + 1,
                "bbox": (cx, name_top_y, int(round(name_pitch)), int(name_cell_h))
            })

        # 3. Question rows active snapping
        detected_qs = [mid - 10 for mid in marker_map if mid >= 11]
        max_q = max(detected_qs) if detected_qs else 4
        max_q = max(4, min(8, max_q))

        question_rows = []
        for q_num in range(1, max_q + 1):
            mid = q_num + 10
            est_my = float(marker_map[mid][:, 1].mean()) if mid in marker_map else (495.0 + (q_num - 1) * 280.0)

            # Snap top and bottom horizontal lines of the row
            y_search_min = max(0, int(est_my - 35))
            y_search_max = min(gray.shape[0], int(est_my + 140))
            row_h_strip = h_lines[y_search_min:y_search_max, 220:1020]
            row_h_prof = row_h_strip.sum(axis=1)

            peaks = []
            for dy in range(2, len(row_h_prof) - 2):
                if row_h_prof[dy] > 0.20 * row_h_prof.max() and row_h_prof[dy] == max(row_h_prof[dy-2:dy+3]):
                    peaks.append((dy, row_h_prof[dy]))
            peaks.sort(key=lambda p: p[1], reverse=True)

            row_top = int(round(est_my - 5))
            row_h = 100
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    dist = abs(peaks[i][0] - peaks[j][0])
                    if 85 <= dist <= 115:
                        row_top = y_search_min + min(peaks[i][0], peaks[j][0])
                        row_h = dist
                        break
                if row_h != 100:
                    break

            # Snap vertical divider lines
            v_strip = v_lines[row_top + 10 : row_top + row_h - 10, 160:1500]
            v_prof = v_strip.sum(axis=0)
            v_pks = []
            if v_prof.max() > 0:
                for dx in range(2, len(v_prof) - 2):
                    if v_prof[dx] > 0.20 * v_prof.max() and v_prof[dx] == max(v_prof[dx-2:dx+3]):
                        v_pks.append(160 + dx)

            clustered_v = []
            for p in v_pks:
                if not clustered_v or (p - clustered_v[-1]) > 10:
                    clustered_v.append(p)
                else:
                    clustered_v[-1] = int(round((clustered_v[-1] + p) / 2.0))
            clustered_v = [p for p in clustered_v if p >= 185]

            best_chain = []
            for i in range(len(clustered_v)):
                for j in range(i + 1, len(clustered_v)):
                    pitch_cand = clustered_v[j] - clustered_v[i]
                    if 85 <= pitch_cand <= 115:
                        chain = [clustered_v[i], clustered_v[j]]
                        expected_next = clustered_v[j] + pitch_cand
                        for k in range(j + 1, len(clustered_v)):
                            if abs(clustered_v[k] - expected_next) <= 15:
                                chain.append(clustered_v[k])
                                pitch_cand = (clustered_v[k] - chain[0]) / float(len(chain) - 1)
                                expected_next = clustered_v[k] + pitch_cand
                        if len(chain) > len(best_chain):
                            best_chain = chain

            # Determine row cell count (e.g. 8, 9, 10 cells)
            if len(best_chain) >= 2:
                start_x = best_chain[0]
                detected_cell_count = len(best_chain) - 1
                row_cell_count = max(8, min(12, detected_cell_count))
                pitch = (best_chain[row_cell_count] - start_x) / float(row_cell_count) if len(best_chain) > row_cell_count else float(best_chain[-1] - best_chain[0]) / float(len(best_chain) - 1)
            else:
                start_x = 224
                pitch = 100.0
                row_cell_count = 8

            cells = []
            for c in range(row_cell_count):
                cx = int(round(start_x + c * pitch))
                cells.append({
                    "section": f"question_{q_num}",
                    "q_num": q_num,
                    "cell_idx": c + 1,
                    "bbox": (cx, row_top, int(round(pitch)), int(row_h))
                })
            question_rows.append({"q_num": q_num, "cells": cells, "marker_id": mid})

        return name_cells, question_rows

    def suppress_grid_borders(
        self,
        rectified_bgr,
        name_cells=None,
        question_rows=None,
        border_color=(255, 255, 255),
        safe_margin_pct=8,
    ):
        """
        Non-destructive, geometrically-bounded printed grid border suppression.

        Mathematical and structural guarantees:
        1. Safe Core Invariant: An inner safe zone of every cell (core 84% area) is strictly
           masked out from any border erasure. Character strokes inside cells can NEVER be erased.
        2. Multi-Cell Kernel Constraints: Horizontal lines require >= 80px continuous width,
           preventing handwritten horizontal bars from matching. Vertical lines require >= 75px
           continuous height, preventing strokes in Ш, Щ, И, П, Н from matching.
        3. Dilation Containment: Dilation (3x3) is re-constrained by the perimeter envelope,
           ensuring the mask never bleeds into the cell interior or clips descenders/ascenders.
        """
        if name_cells is None or question_rows is None:
            name_cells, question_rows = self.extract_cells(rectified_bgr)

        ih, iw = rectified_bgr.shape[:2]
        gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY) if rectified_bgr.ndim == 3 else rectified_bgr.copy()
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        # 1. Construct strict geometric perimeter mask
        perimeter_mask = np.zeros((ih, iw), dtype=np.uint8)
        all_cells = list(name_cells)
        for q in question_rows:
            all_cells.extend(q.get("cells", []))

        for cell in all_cells:
            cx, cy, cw, ch = cell["bbox"]
            my = max(6, int(round(ch * (safe_margin_pct / 100.0))))
            mx = max(6, int(round(cw * (safe_margin_pct / 100.0))))

            # Outer bounding box envelope (+2px padding)
            x1_out, y1_out = max(0, cx - 2), max(0, cy - 2)
            x2_out, y2_out = min(iw, cx + cw + 2), min(ih, cy + ch + 2)
            cv2.rectangle(perimeter_mask, (x1_out, y1_out), (x2_out, y2_out), 255, -1)

            # Cut out protected safe core where character strokes live
            x1_in, y1_in = cx + mx, cy + my
            x2_in, y2_in = cx + cw - mx, cy + ch - my
            if x2_in > x1_in and y2_in > y1_in:
                cv2.rectangle(perimeter_mask, (x1_in, y1_in), (x2_in, y2_in), 0, -1)

        # 2. Multi-cell morphological kernels:
        # Long horizontal kernel (80px) ensures single-character horizontal strokes cannot match.
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        # Tall vertical kernel (75px) ensures ~50-60px vertical character strokes (Ш, Щ, И, etc.) cannot match.
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 75))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        detected_lines = cv2.bitwise_or(h_lines, v_lines)

        # 3. Restrict suppression strictly to the perimeter zone of detected cells
        safe_border_mask = cv2.bitwise_and(detected_lines, perimeter_mask)

        # Dilate slightly (3x3) to cover anti-aliased edge fringes, but re-mask to preserve cell interior
        dilated_borders = cv2.dilate(safe_border_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        final_border_mask = cv2.bitwise_and(dilated_borders, perimeter_mask)

        # 4. Apply suppression cleanly
        suppressed_bgr = rectified_bgr.copy()
        if isinstance(border_color, (int, float)):
            fill_val = np.array([border_color, border_color, border_color], dtype=np.uint8)
        else:
            fill_val = np.array(border_color, dtype=np.uint8)

        suppressed_bgr[final_border_mask > 0] = fill_val
        suppressed_count = int(np.count_nonzero(final_border_mask))

        return suppressed_bgr, final_border_mask, suppressed_count

    def classify_cell(
        self,
        cell_bgr,
        margin_trim_pct=4,
        is_header: bool = False,
        model=None,
        char_list=None,
        contrast_mode: str = "percentile",
        top_k: int = 3,
        force_predict: bool = False,
    ) -> dict:
        active_session = None
        active_model = None

        if model is not None:
            if ort is not None and isinstance(model, ort.InferenceSession):
                active_session = model
            elif hasattr(model, "run"):
                active_session = model
            else:
                active_model = model
        else:
            if self.is_onnx:
                active_session = self.session
            else:
                active_model = self.model

        active_chars = char_list if char_list is not None else TATAR_UPPERCASE

        ch, cw = cell_bgr.shape[:2]
        if ch < 10 or cw < 10:
            return {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)], "top_items": []}

        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY) if cell_bgr.ndim == 3 else cell_bgr

        # 1. Clean geometric inner margin to avoid outer printed grid borders
        my = max(5, int(round(ch * (margin_trim_pct / 100.0)))) if margin_trim_pct > 0 else 0
        mx = max(5, int(round(cw * (margin_trim_pct / 100.0)))) if margin_trim_pct > 0 else 0
        inner_gray = gray[my:ch - my, mx:cw - mx] if (my > 0 and mx > 0 and (ch - 2*my) >= 8 and (cw - 2*mx) >= 8) else gray
        ih, iw = inner_gray.shape[:2]

        if ih < 8 or iw < 8:
            return {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)], "top_items": []}

        # 2. Ink segmentation & connected components
        bg_val = float(np.median(inner_gray))
        p5_val = float(np.percentile(inner_gray, 5))
        contrast_range = bg_val - p5_val

        if contrast_range > 60:
            ink_delta = 22.0
        elif contrast_range > 35:
            ink_delta = 16.0
        else:
            ink_delta = max(7.0, 0.32 * contrast_range)

        ink_mask = (inner_gray < (bg_val - ink_delta)).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask)
        if num_labels <= 1 and not force_predict:
            return {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)], "top_items": []}

        # Pre-filter out printed border lines, perimeter frames, and corner border specks
        candidate_indices = []
        for i in range(1, num_labels):
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Detect full-box printed perimeter frames (hollow outer rectangle enclosing the cell)
            is_hollow_box = (
                (bx <= 3 and by <= 3 and bx + bw >= iw - 3 and by + bh >= ih - 3) and
                (area < 0.12 * (bw * bh))
            )

            is_edge = (
                is_hollow_box or
                (bw > 0.35 * iw and bh <= 8 and (by <= 4 or by + bh >= ih - 4)) or
                (bh > 0.35 * ih and bw <= 8 and (bx <= 4 or bx + bw >= iw - 4)) or
                (bh > 0.60 * ih and bw <= 12 and (bx <= 2 or bx + bw >= iw - 2)) or
                (bw > 0.60 * iw and bh <= 12 and (by <= 2 or by + bh >= ih - 2)) or
                ((bx <= 3 or by <= 3 or bx + bw >= iw - 3 or by + bh >= ih - 3) and (bw <= 5 or bh <= 5) and area < 40)
            )
            if not is_edge:
                candidate_indices.append(i)

        if not candidate_indices and not force_predict:
            return {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)], "top_items": []}

        # Find the main character body among candidate components
        if candidate_indices:
            best_candidate_idx = max(candidate_indices, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
        else:
            best_candidate_idx = 1
        max_comp_area = stats[best_candidate_idx, cv2.CC_STAT_AREA]
        mbx = stats[best_candidate_idx, cv2.CC_STAT_LEFT]
        mby = stats[best_candidate_idx, cv2.CC_STAT_TOP]
        mbw = stats[best_candidate_idx, cv2.CC_STAT_WIDTH]
        mbh = stats[best_candidate_idx, cv2.CC_STAT_HEIGHT]

        if max_comp_area < 20 and not force_predict:
            return {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)], "top_items": []}

        valid_ink = np.zeros_like(ink_mask)
        valid_ink[labels == best_candidate_idx] = 1
        total_ink_pixels = max_comp_area

        for i in candidate_indices:
            if i == best_candidate_idx:
                continue
            area = stats[i, cv2.CC_STAT_AREA]
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # Check if component is a top diacritic (dots for Ё, breve for Й)
            is_top_diacritic = (
                (by + bh <= mby + 5) and
                (mby - (by + bh) <= 22) and
                (bx + bw >= mbx - 6) and
                (bx <= mbx + mbw + 6) and
                (bw < 0.60 * iw)
            )

            near_border = (bx <= 2 or by <= 2 or (bx + bw) >= iw - 2 or (by + bh) >= ih - 2)
            if near_border and not is_top_diacritic and area < 0.35 * max_comp_area:
                continue

            dist_x = max(0, bx - (mbx + mbw), mbx - (bx + bw))
            dist_y = max(0, by - (mby + mbh), mby - (by + bh))
            if not is_top_diacritic:
                if (dist_x > 10 or dist_y > 10) and area < 0.25 * max_comp_area:
                    continue

            if area < 8 and not is_top_diacritic:
                continue

            valid_ink[labels == i] = 1
            total_ink_pixels += area

        # Scale-free ink statistics (relative to the inner cell) used by the emptiness decision
        vy, vx = np.nonzero(valid_ink)
        cy, cx = int(round(0.15 * ih)), int(round(0.15 * iw))
        y_40 = int(round(0.40 * ih))
        core = valid_ink[cy:ih - cy, cx:iw - cx] > 0
        low_core = valid_ink[y_40:ih - cy, cx:iw - cx] > 0
        bh_k = max(9, int(round(0.25 * min(ih, iw))) | 1)
        blackhat = cv2.morphologyEx(inner_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (bh_k, bh_k)))
        ink_stats = {
            "ink_frac": total_ink_pixels / float(ih * iw),
            "ink_h": (int(vy.max() - vy.min()) + 1) / float(ih) if len(vy) else 0.0,
            "ink_w": (int(vx.max() - vx.min()) + 1) / float(iw) if len(vx) else 0.0,
            "ink_contrast": bg_val - float(np.median(inner_gray[vy, vx])) if len(vy) else 0.0,
            "core_frac": float(core.sum()) / float(ih * iw),
            "low_core_frac": float(low_core.sum()) / float(ih * iw),
            "top_ratio": float((valid_ink[:y_40, :] > 0).sum()) / float(max(1, total_ink_pixels)),
            "core_contrast": float(np.median(blackhat[cy:ih - cy, cx:iw - cx][core])) if core.any() else 0.0,
            "stroke_contrast": float(np.median(blackhat[vy, vx])) if len(vy) else 0.0,
            "core_bh_p90": float(np.percentile(blackhat[cy:ih - cy, cx:iw - cx], 90)),
            "core_bh_p95": float(np.percentile(blackhat[cy:ih - cy, cx:iw - cx], 95)),
            "core_bh_p98": float(np.percentile(blackhat[cy:ih - cy, cx:iw - cx], 98)),
        }

        is_empty_cell = (
            total_ink_pixels < 35
            or ink_stats["core_frac"] < 0.005
            or (ink_stats["core_bh_p98"] < 10.0 and ink_stats["ink_contrast"] < 25.0)
            or (ink_stats["ink_w"] >= 0.92 and ink_stats["low_core_frac"] < 0.010 and ink_stats["top_ratio"] > 0.45)
        )

        # Mark empty if stroke component is tiny, border-clinging, diffuse shadow gradient, or top-band intrusion
        if is_empty_cell and not force_predict:
            return {
                "is_empty": True,
                "char": " ",
                "confidence": 100.0,
                "top3": [(" ", 100.0)],
                "top_items": [],
                "ink_stats": ink_stats,
                "ink_mask": valid_ink,
            }

        # 3. Ink Bounding-Box Extraction
        ys, xs = np.where(valid_ink > 0)
        if (len(xs) == 0 or len(ys) == 0) and not force_predict:
            return {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)], "top_items": []}

        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(np.min(xs)), int(np.max(xs))
            y1, y2 = int(np.min(ys)), int(np.max(ys))

            # Add small cushion padding
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(iw - 1, x2 + pad)
            y2 = min(ih - 1, y2 + pad)

            glyph_crop = inner_gray[y1:y2 + 1, x1:x2 + 1]
        else:
            glyph_crop = inner_gray

        gh, gw = glyph_crop.shape[:2]

        # 4. Aspect-ratio preserving scaling to canonical ~75% occupancy (48px out of 64px)
        max_dim = max(gh, gw)
        target_dim = 48.0
        scale = target_dim / float(max_dim + 1e-5)
        new_w = max(1, min(60, int(round(gw * scale))))
        new_h = max(1, min(60, int(round(gh * scale))))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized_glyph = cv2.resize(glyph_crop, (new_w, new_h), interpolation=interp)

        # 5. Contrast normalization
        if contrast_mode == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            norm_glyph = clahe.apply(resized_glyph)
        else:
            p_lo = np.percentile(resized_glyph, 2)
            p_hi = np.percentile(resized_glyph, 98)
            norm_glyph = np.clip((resized_glyph.astype(np.float32) - p_lo) / (p_hi - p_lo + 1e-5) * 240.0 + 10.0, 0, 255).astype(np.uint8)

        # 6. Center on 64x64 canvas
        inp_64 = np.full((64, 64), 250, dtype=np.uint8)
        off_x = (64 - new_w) // 2
        off_y = (64 - new_h) // 2
        inp_64[off_y:off_y + new_h, off_x:off_x + new_w] = norm_glyph

        # 7. Tensor inference
        t = ((inp_64.astype(np.float32) / 255.0) - 0.5) / 0.5
        k = min(max(3, top_k), len(active_chars))

        if active_session is not None:
            inp_array = np.expand_dims(np.expand_dims(t, axis=0), axis=0).astype(np.float32)
            input_name = active_session.get_inputs()[0].name
            raw_logits = active_session.run(None, {input_name: inp_array})[0][0]
            # Numerically stable softmax
            exp_l = np.exp(raw_logits - np.max(raw_logits))
            probs = exp_l / np.sum(exp_l)
            top_indices = np.argsort(probs)[::-1][:k]
            top_chars = [active_chars[i] for i in top_indices]
            top_confs = (probs[top_indices] * 100.0).tolist()
        else:
            tensor = torch.from_numpy(t).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = active_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, k=k)
            top_chars = [active_chars[i] for i in top_indices.cpu().numpy()]
            top_confs = (top_probs.cpu().numpy() * 100.0).tolist()

        if top_chars and top_chars[0] in ("Ё", "Й") and not is_empty_cell:
            n_lbl, _, st, _ = cv2.connectedComponentsWithStats(valid_ink)
            has_top_diacritic = False
            if n_lbl > 2:
                areas = [st[i, cv2.CC_STAT_AREA] for i in range(1, n_lbl)]
                main_c_idx = int(np.argmax(areas)) + 1
                main_top = st[main_c_idx, cv2.CC_STAT_TOP]
                for i in range(1, n_lbl):
                    if i == main_c_idx:
                        continue
                    c_top = st[i, cv2.CC_STAT_TOP]
                    c_h = st[i, cv2.CC_STAT_HEIGHT]
                    c_area = st[i, cv2.CC_STAT_AREA]
                    if (c_top + c_h) <= main_top + 2 and c_area >= 10:
                        has_top_diacritic = True
                        break
            if not has_top_diacritic:
                target_alt = "Е" if top_chars[0] == "Ё" else ("О" if "О" in top_chars else "И")
                if target_alt in top_chars:
                    alt_idx = top_chars.index(target_alt)
                    top_chars.insert(0, top_chars.pop(alt_idx))
                    top_confs.insert(0, top_confs.pop(alt_idx))

        top_items = [
            {
                "char": ch_val,
                "unicode": f"U+{ord(ch_val):04X}",
                "confidence": round(float(conf_val), 2),
            }
            for ch_val, conf_val in zip(top_chars[:top_k], top_confs[:top_k])
        ]

        is_empty = is_empty_cell
        return {
            "is_empty": is_empty,
            "char": " " if is_empty else top_chars[0],
            "confidence": 100.0 if is_empty else round(float(top_confs[0]), 1),
            "top3": [(" ", 100.0)] if is_empty else list(zip(top_chars[:3], [round(c, 1) for c in top_confs[:3]])),
            "top_items": top_items,
            "resized_glyph": resized_glyph,
            "inp_64": inp_64,
            "ink_stats": ink_stats,
            "ink_mask": valid_ink,
        }

    def process_blank(self, img_bgr, annotate=True):
        rectified, success, method = self.rectify_sheet(img_bgr)
        name_cells, question_rows = self.extract_cells(rectified)

        annotated_img = rectified.copy() if annotate else None

        draw = None
        annotated_pil = None
        font_badge = None
        font_badge_small = None
        if annotate:
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
                if draw:
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
                    if draw:
                        self._draw_cell_badge(draw, x, y, w, h, res, font_badge)

            q_text = "".join(q_chars)
            questions_results.append({
                "q_num": q["q_num"],
                "text": q_text,
                "cells": q_cells_res
            })

        avg_conf = (conf_sum / max(1, total_recognized_letters)) if total_recognized_letters > 0 else 0.0
        final_annotated_bgr = None
        if draw and annotated_pil:
            draw.rectangle([(0, 0), (CANVAS_W, 50)], fill=(15, 23, 42, 230))
            banner_text = f"Tatar OCR Scanner • Name: '{student_name_str or 'N/A'}' • Recognized {total_recognized_letters} letters • Avg Confidence: {avg_conf:.1f}% • Alignment: {method}"
            draw.text((30, 8), banner_text, fill=(255, 255, 255), font=font_badge_small)
            final_annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

        return {
            "status": "success",
            "rectification_method": method,
            "student_name": student_name_str,
            "student_name_cells": name_results,
            "questions": questions_results,
            "stats": {
                "total_letters": total_recognized_letters,
                "avg_confidence": round(avg_conf, 1),
                "questions_count": len(questions_results)
            },
            "rectified_bgr": rectified,
            "annotated_bgr": final_annotated_bgr
        }

    def process_image(self, image_path, annotate=False):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.process_blank(img_bgr, annotate=annotate)

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


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json(x) for x in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_benchmark_predictions(images_dir, output_path, scanner=None):
    if scanner is None:
        scanner = BlankOCRScanner()
    images_path = Path(images_dir)
    images = sorted([p for p in images_path.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    output_path = Path(output_path)

    is_json_file = output_path.suffix.lower() == '.json'
    predictions_map = {}

    if is_json_file:
        out_dir = output_path.parent / output_path.stem
        out_file = output_path
    else:
        out_dir = output_path
        out_file = output_path / "predictions.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)

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
                "cells": raw.get("student_name_cells", [])
            },
            "questions": raw.get("questions", []),
            "stats": raw.get("stats", {})
        })
        predictions_map[image_id] = payload

        with open(out_dir / f"{image_id}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions_map, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(images)} predictions to {out_file} and directory {out_dir}")
    return predictions_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tatar Blank OCR Scanner Runner")
    parser.add_argument("--images-dir", type=str, default="benchmark/images", help="Path to images directory")
    parser.add_argument("--output", type=str, default="benchmark/predictions/python_baseline.json", help="Path to output json file or directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, cuda)")
    args = parser.parse_args()

    scanner = BlankOCRScanner(device=args.device)
    export_benchmark_predictions(args.images_dir, args.output, scanner=scanner)


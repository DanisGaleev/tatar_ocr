"""
v3.evaluate_blanks
Unified evaluation suite for Tatar Blank OCR models on the 12 canonical blanks.
Supports PyTorch models (.pth / nn.Module) and ONNX models (FP32 / INT8).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Union, Dict, Any, List

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
import cv2
import torch
import torch.nn as nn

from blank_pipeline import BlankOCRScanner
from tatar_ocr_augmentation import TATAR_UPPERCASE

TATAR_SPECIFIC = ["Ә", "Ө", "Ү", "Җ", "Ң", "Һ"]


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def load_canonical_ground_truth(gt_dir: Union[str, Path] = None) -> Dict[str, Any]:
    if gt_dir is None:
        gt_dir = WORKSPACE_DIR / "benchmark" / "ground_truth"
    gt_dir = Path(gt_dir)
    gt_all_file = gt_dir / "ground_truth_all.json"
    if gt_all_file.exists():
        with open(gt_all_file, "r", encoding="utf-8") as f:
            return json.load(f)

    gt_map = {}
    for jf in sorted(gt_dir.glob("blank_*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            gt_map[jf.stem] = data
    return gt_map


def evaluate_model_on_blanks(
    model_or_path: Union[str, Path, nn.Module],
    images_dir: Union[str, Path] = None,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluates a model (pth path, onnx path, or torch nn.Module) on the 12 canonical test blanks.
    """
    if images_dir is None:
        images_dir = WORKSPACE_DIR / "benchmark" / "images"
    images_dir = Path(images_dir)

    # Instantiate scanner
    if isinstance(model_or_path, (str, Path)):
        model_str = str(model_or_path)
        scanner = BlankOCRScanner(model_path=model_str, device=device)
    elif isinstance(model_or_path, nn.Module):
        scanner = BlankOCRScanner(device=device)
        scanner.is_onnx = False
        scanner.session = None
        scanner.model = model_or_path.to(torch.device(device))
        scanner.model.eval()
    else:
        raise ValueError(f"Unsupported model type: {type(model_or_path)}")

    gt_data = load_canonical_ground_truth()
    blank_images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    total_chars = 0
    correct_chars = 0
    total_lev_edits = 0

    tatar_totals = {ch: 0 for ch in TATAR_SPECIFIC}
    tatar_correct = {ch: 0 for ch in TATAR_SPECIFIC}

    total_fields = 0
    exact_fields = 0
    fuzzy_fields_le1 = 0

    empty_total = 0
    empty_correct = 0
    fp_empty = 0
    fn_empty = 0

    high_conf_errors = 0
    errors = []

    zero_touch_sheets = 0
    total_sheets = 0

    t0_all = time.perf_counter()

    for img_p in blank_images:
        blank_id = img_p.stem
        if blank_id not in gt_data:
            continue
        total_sheets += 1
        sheet_edits = 0
        gt_blank = gt_data[blank_id]
        gt_questions = {str(q["q_num"]): q for q in gt_blank.get("questions", [])}

        pred_res = scanner.process_image(img_p, annotate=False)
        pred_questions = {str(q["q_num"]): q for q in pred_res.get("questions", [])}

        for q_num, gt_q in gt_questions.items():
            total_fields += 1
            gt_text = gt_q.get("text", "")
            pred_q = pred_questions.get(q_num, {})
            pred_text = pred_q.get("text", "")

            # Exact & Fuzzy field match
            if pred_text == gt_text:
                exact_fields += 1
            lev_dist = levenshtein(pred_text, gt_text)
            total_lev_edits += lev_dist
            sheet_edits += lev_dist
            if lev_dist <= 1:
                fuzzy_fields_le1 += 1

            # Cell-level evaluation
            gt_cells = gt_q.get("cells", [])
            pred_cells = pred_q.get("cells", [])
            for idx, gt_cell in enumerate(gt_cells):
                is_gt_empty = gt_cell.get("is_empty", False)
                gt_char = gt_cell.get("char", "")

                pred_cell = pred_cells[idx] if idx < len(pred_cells) else {"is_empty": True, "char": " ", "confidence": 0.0}
                is_pred_empty = pred_cell.get("is_empty", False)
                pred_char = pred_cell.get("char", " ").strip()
                pred_conf = pred_cell.get("confidence", 0.0)

                if is_gt_empty:
                    empty_total += 1
                    if is_pred_empty:
                        empty_correct += 1
                    else:
                        fn_empty += 1
                else:
                    total_chars += 1
                    if is_pred_empty:
                        fp_empty += 1

                    if gt_char in TATAR_SPECIFIC:
                        tatar_totals[gt_char] += 1

                    if not is_pred_empty and pred_char == gt_char:
                        correct_chars += 1
                        if gt_char in TATAR_SPECIFIC:
                            tatar_correct[gt_char] += 1
                    else:
                        err_entry = {
                            "blank": blank_id,
                            "q": q_num,
                            "pos": idx + 1,
                            "gt": gt_char,
                            "pred": pred_char if not is_pred_empty else "<EMPTY>",
                            "conf": round(float(pred_conf), 1)
                        }
                        errors.append(err_entry)
                        if pred_conf >= 80.0:
                            high_conf_errors += 1

        if sheet_edits == 0:
            zero_touch_sheets += 1

    total_time_sec = time.perf_counter() - t0_all

    acc_char = round(correct_chars / max(1, total_chars) * 100.0, 2)
    cer = round(total_lev_edits / max(1, total_chars) * 100.0, 2)
    fem = round(exact_fields / max(1, total_fields) * 100.0, 2)
    ffm_le1 = round(fuzzy_fields_le1 / max(1, total_fields) * 100.0, 2)
    wer = round((total_fields - exact_fields) / max(1, total_fields) * 100.0, 2)
    drr = round((fuzzy_fields_le1 - exact_fields) / max(1, total_fields - exact_fields) * 100.0, 2)

    prec_empty = round(empty_correct / max(1, empty_correct + fp_empty) * 100.0, 2)
    rec_empty = round(empty_correct / max(1, empty_total) * 100.0, 2)
    f1_empty = round(2 * prec_empty * rec_empty / max(1e-5, prec_empty + rec_empty), 2) if (prec_empty + rec_empty) > 0 else 0.0

    ztsr = round(zero_touch_sheets / max(1, total_sheets) * 100.0, 2)
    mcps = round(total_lev_edits / max(1, total_sheets), 2)
    hcer_80 = round(high_conf_errors / max(1, total_chars) * 100.0, 2)
    sfp_80 = round(high_conf_errors / max(1, len(errors)) * 100.0, 2) if errors else 0.0
    t_review = round(1.5 + mcps * 1.2, 2)
    l_blank_ms = round(total_time_sec / max(1, total_sheets) * 100.0, 1)

    total_tatar_chars = sum(tatar_totals.values())
    total_tatar_corr = sum(tatar_correct.values())
    r_tatar = round(total_tatar_corr / max(1, total_tatar_chars) * 100.0, 2)

    tatar_breakdown = {
        ch: f"{tatar_correct[ch]}/{tatar_totals[ch]} ({(tatar_correct[ch]/max(1, tatar_totals[ch])*100):.1f}%)"
        for ch in TATAR_SPECIFIC
    }

    results = {
        "Acc_char": acc_char,
        "correct_chars": f"{correct_chars}/{total_chars}",
        "CER": cer,
        "total_lev_edits": total_lev_edits,
        "R_Tatar": r_tatar,
        "tatar_count": f"{total_tatar_corr}/{total_tatar_chars}",
        "Tatar_breakdown": tatar_breakdown,
        "Prec_empty": prec_empty,
        "Rec_empty": rec_empty,
        "F1_empty": f1_empty,
        "FEM": fem,
        "exact_fields": f"{exact_fields}/{total_fields}",
        "FFM_le1": ffm_le1,
        "fuzzy_fields": f"{fuzzy_fields_le1}/{total_fields}",
        "WER": wer,
        "DRR": drr,
        "ZTSR": ztsr,
        "zero_touch_sheets": f"{zero_touch_sheets}/{total_sheets}",
        "MCPS": mcps,
        "HCER_80": hcer_80,
        "high_conf_errors_80": f"{high_conf_errors}/{total_chars}",
        "SFP_80": sfp_80,
        "T_review_sec": t_review,
        "L_blank_ms": l_blank_ms,
        "errors_count": len(errors),
        "errors": errors,
        "eval_time_sec": round(total_time_sec, 2),
    }

    if verbose:
        print("=" * 65)
        print(f"BENCHMARK REPORT: 12 Real Blanks ({total_chars} letters, {total_fields} fields)")
        print("=" * 65)
        print(f"  • Acc_char:     {acc_char:6.2f}% ({correct_chars}/{total_chars})")
        print(f"  • CER:          {cer:6.2f}% ({total_lev_edits} edits)")
        print(f"  • R_Tatar:      {r_tatar:6.2f}% ({total_tatar_corr}/{total_tatar_chars})")
        print(f"  • FEM:          {fem:6.2f}% ({exact_fields}/{total_fields})")
        print(f"  • FFM (Lev<=1): {ffm_le1:6.2f}% ({fuzzy_fields_le1}/{total_fields})")
        print(f"  • HCER_80%:     {results['HCER_80']:6.2f}% ({high_conf_errors} errors with conf >= 80%)")
        print(f"  • Total Errors: {len(errors)}")
        print("  • Tatar Letters Breakdown:")
        for ch, stats_str in tatar_breakdown.items():
            print(f"      {ch}: {stats_str}")
        print(f"  • Elapsed Time: {total_time_sec:.2f}s")
        print("=" * 65)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate OCR model on 12 Blanks")
    parser.add_argument("--model", type=str, default="models/tatar_ocr_v2_handwritten39_int8.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    res = evaluate_model_on_blanks(args.model, device=args.device)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

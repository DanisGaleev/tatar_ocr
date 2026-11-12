"""Thin Python facade over the portable C++20 OCR core (tatar_ocr_native)."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC_PATH = WORKSPACE_DIR / "pipeline_spec.json"

_DLL_DIR_HANDLES = []
_REGISTERED_DLL_DIRS: set[Path] = set()
_NATIVE_MODULE = None


def _resolve_spec_path(spec_path: Optional[str | Path] = None) -> Path:
    if spec_path is None:
        return DEFAULT_SPEC_PATH
    path = Path(spec_path)
    if not path.is_absolute() and not path.exists():
        candidate = WORKSPACE_DIR / path
        if candidate.exists():
            return candidate
    return path


def load_pipeline_spec(spec_path: Optional[str | Path] = None) -> dict:
    """Load declarative pipeline constants and blank geometry from JSON."""
    with open(_resolve_spec_path(spec_path), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_native_module():
    global _NATIVE_MODULE
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE

    candidate_sys_paths = [
        WORKSPACE_DIR / "native" / "build" / "Release",
        WORKSPACE_DIR / "native" / "build",
        WORKSPACE_DIR,
    ]
    for path in candidate_sys_paths:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    if hasattr(os, "add_dll_directory"):
        dll_dirs: list[Path] = [p for p in candidate_sys_paths[:2] if p.is_dir()]
        native_deps = WORKSPACE_DIR / ".native_deps"
        if native_deps.is_dir():
            for dll_file in native_deps.rglob("*.dll"):
                if dll_file.parent not in dll_dirs:
                    dll_dirs.append(dll_file.parent)
        for dll_dir in dll_dirs:
            if dll_dir not in _REGISTERED_DLL_DIRS:
                _REGISTERED_DLL_DIRS.add(dll_dir)
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))

    try:
        import tatar_ocr_native
    except ImportError as exc:
        raise ImportError(
            "tatar_ocr_native is not built. Run: cmake -S native -B native/build && cmake --build native/build --config Release"
        ) from exc

    _NATIVE_MODULE = tatar_ocr_native
    return _NATIVE_MODULE


# Backwards-compatible constants populated from default spec
_DEFAULT_SPEC = load_pipeline_spec()
_MODEL_SPEC = _DEFAULT_SPEC["model"]
_CANVAS_SPEC = _DEFAULT_SPEC["canvas"]
_ARUCO_SPEC = _DEFAULT_SPEC["aruco"]

DEFAULT_MODEL_PATH = _MODEL_SPEC["default_onnx"]
FALLBACK_ONNX_MODEL = Path(_MODEL_SPEC["fallback_onnx"])
FALLBACK_PTH_MODEL = Path(_MODEL_SPEC["fallback_pth"])
TATAR_UPPERCASE = tuple(_MODEL_SPEC["alphabet"])

CANVAS_W = int(_CANVAS_SPEC["width"])
CANVAS_H = int(_CANVAS_SPEC["height"])
DST_CORNERS = np.array(_CANVAS_SPEC["dst_corners"], dtype=np.float32)
CANONICAL_ANCHORS = {
    int(k): np.array(v, dtype=np.float32)
    for k, v in _ARUCO_SPEC["canonical_anchors"].items()
}


def _apply_diacritic_guard(
    probs: np.ndarray,
    active_chars,
    top_chars: list[str],
    top_confs: list[float],
    has_top_diacritic: bool,
    is_empty_cell: bool,
    guard_chars: tuple[str, ...],
    k: int,
) -> tuple[list[str], list[float]]:
    if not top_chars or top_chars[0] not in guard_chars or is_empty_cell or has_top_diacritic:
        return top_chars, top_confs

    target_alt = "Е" if top_chars[0] == "Ё" else ("О" if "О" in top_chars else "И")
    if target_alt in top_chars:
        alt_idx = top_chars.index(target_alt)
        top_chars.insert(0, top_chars.pop(alt_idx))
        top_confs.insert(0, top_confs.pop(alt_idx))
    elif target_alt in active_chars:
        char_idx = list(active_chars).index(target_alt)
        if char_idx < len(probs):
            top_chars = [target_alt] + top_chars[: k - 1]
            top_confs = [float(probs[char_idx]) * 100.0] + top_confs[: k - 1]
    return top_chars, top_confs


class BlankOCRScanner:
    """Thin wrapper routing blank OCR through C++ tatar_ocr_native (+ PyTorch eval support)."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: str = "cpu",
        spec_path: Optional[str | Path] = None,
        backend: str = "auto",
    ):
        self._spec_path = _resolve_spec_path(spec_path)
        self._spec = load_pipeline_spec(self._spec_path)
        self._model_spec = self._spec["model"]
        self._aruco_spec = self._spec["aruco"]
        self._cell_spec = self._spec["cell"]
        self.alphabet = tuple(self._model_spec["alphabet"])

        self.device = device
        self.requested_backend = backend
        self.model_path = str(model_path if model_path is not None else self._model_spec["default_onnx"])
        self.is_onnx = self.model_path.lower().endswith(".onnx")
        self.session = None
        self.model = None

        weights = self._resolve_weights_path(self.model_path, self.is_onnx)
        default_onnx = self._resolve_weights_path(self._model_spec["default_onnx"], True)
        native_mod = _load_native_module()

        native_weights = weights if (self.is_onnx and weights.exists()) else default_onnx
        if not native_weights.exists():
            raise FileNotFoundError(f"ONNX model weights not found: {native_weights}")

        self._native = native_mod.NativeBlankScanner(str(native_weights), str(self._spec_path))
        if self.is_onnx:
            print(f"BlankOCRScanner: Loaded native ONNX backend from {native_weights}")
        else:
            self._load_torch_model(weights)

    def _resolve_weights_path(self, raw_path: str, is_onnx: bool) -> Path:
        weights = Path(raw_path)
        if not weights.is_absolute():
            weights = WORKSPACE_DIR / weights
        if not weights.exists():
            fallback_rel = self._model_spec["fallback_onnx"] if is_onnx else self._model_spec["fallback_pth"]
            weights = WORKSPACE_DIR / fallback_rel
        return weights

    def _load_torch_model(self, weights: Path) -> None:
        import torch
        from tatar_ocr_dataset import TatarOCRNet

        num_classes = len(self.alphabet)
        v3_markers = tuple(self._model_spec["v3_ckpt_markers"])
        if weights.exists():
            state_dict = torch.load(weights, map_location=self.device, weights_only=True)
            if any(marker in k for k in state_dict.keys() for marker in v3_markers):
                from v3.architecture import TatarOCRNetV3

                self.model = TatarOCRNetV3(num_classes=num_classes).to(self.device)
            else:
                self.model = TatarOCRNet(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(state_dict)
            print(f"BlankOCRScanner: Loaded PyTorch model ({self.model.__class__.__name__}) from {weights}")
        else:
            self.model = TatarOCRNet(num_classes=num_classes).to(self.device)
            print("BlankOCRScanner: Warning - PyTorch model weights not found!")
        self.model.eval()

    @property
    def backend(self) -> str:
        return "native" if (self.is_onnx and str(self.device) == "cpu") else "pytorch"

    def detect_markers_padded(self, img_bgr: np.ndarray, pad: Optional[int] = None) -> dict[int, np.ndarray]:
        pad_px = int(self._aruco_spec["rectify_pad"] if pad is None else pad)
        return self._native.detect_markers_padded(np.ascontiguousarray(img_bgr, dtype=np.uint8), pad_px)

    def rectify_sheet(self, img_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
        return self._native.rectify_sheet(np.ascontiguousarray(img_bgr, dtype=np.uint8))

    def extract_cells(self, rectified_bgr: np.ndarray) -> tuple[list[dict], list[dict]]:
        return self._native.extract_cells(np.ascontiguousarray(rectified_bgr, dtype=np.uint8))

    def _infer_custom_batch(self, inp_64_list: list[np.ndarray], model) -> np.ndarray:
        tensors = np.stack([((arr.astype(np.float32) / 255.0) - 0.5) / 0.5 for arr in inp_64_list], axis=0)
        inp_4d = tensors[:, np.newaxis, :, :]
        if hasattr(model, "run"):
            input_name = model.get_inputs()[0].name
            logits = model.run(None, {input_name: inp_4d.astype(np.float32)})[0]
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_l / np.sum(exp_l, axis=1, keepdims=True)

        import torch

        with torch.no_grad():
            logits = model(torch.from_numpy(inp_4d).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def _classify_custom_batch(
        self,
        crops: list[np.ndarray],
        model,
        char_list,
        margin_trim_pct: float,
        top_k: int,
        force_predict: bool,
        debug: bool,
    ) -> list[dict]:
        """Run C++ CV preprocessing in native core, then score tensors with custom/PyTorch model."""
        active_chars = char_list if char_list is not None else self.alphabet
        guard_chars = tuple(self._cell_spec["diacritic_guard_chars"])
        k = min(max(3, top_k), len(active_chars))

        preps = [
            {"is_empty": True, "char": " ", "confidence": 100.0, "top3": [(" ", 100.0)], "top_items": []}
            if crop.size == 0
            else self._native.classify_cell(
                np.ascontiguousarray(crop, dtype=np.uint8), margin_trim_pct, k, force_predict, True
            )
            for crop in crops
        ]
        pending = [i for i, r in enumerate(preps) if "inp_64" in r and (not r["is_empty"] or force_predict)]
        probs_batch = self._infer_custom_batch([preps[i]["inp_64"] for i in pending], model) if pending else []

        for idx, probs in zip(pending, probs_batch):
            p = preps[idx]
            top_indices = np.argsort(probs)[::-1][:k]
            top_chars = [active_chars[j] for j in top_indices]
            top_confs = (probs[top_indices] * 100.0).tolist()
            top_chars, top_confs = _apply_diacritic_guard(
                probs, active_chars, top_chars, top_confs,
                bool(p.get("has_top_diacritic")), bool(p["is_empty"]), guard_chars, k,
            )
            p["top_items"] = [
                {"char": ch, "unicode": f"U+{ord(ch):04X}", "confidence": round(float(cf), 2)}
                for ch, cf in zip(top_chars[:top_k], top_confs[:top_k])
            ]
            if p["is_empty"]:
                p["char"], p["confidence"], p["top3"] = " ", 100.0, [(" ", 100.0)]
            else:
                p["char"] = top_chars[0]
                p["confidence"] = round(float(top_confs[0]), 1)
                p["top3"] = list(zip(top_chars[:3], [round(c, 1) for c in top_confs[:3]]))

        if not debug:
            for p in preps:
                for dbg_key in ("resized_glyph", "inp_64", "ink_stats", "ink_mask", "has_top_diacritic"):
                    p.pop(dbg_key, None)
        return preps

    def classify_cell(
        self,
        cell_bgr: np.ndarray,
        margin_trim_pct: Optional[float] = None,
        model=None,
        char_list=None,
        top_k: int = 3,
        force_predict: bool = False,
        debug: bool = False,
    ) -> dict:
        trim = float(self._cell_spec["default_margin_trim_pct"] if margin_trim_pct is None else margin_trim_pct)
        if self.is_onnx and model is None and char_list is None and str(self.device) == "cpu":
            return self._native.classify_cell(
                np.ascontiguousarray(cell_bgr, dtype=np.uint8), trim, int(top_k), bool(force_predict), bool(debug)
            )
        active_model = model if model is not None else self.model
        return self._classify_custom_batch([cell_bgr], active_model, char_list, trim, top_k, force_predict, debug)[0]

    def process_blank(self, img_bgr: np.ndarray, annotate: bool = False) -> dict:
        if self.is_onnx and str(self.device) == "cpu":
            result = self._native.process_blank(np.ascontiguousarray(img_bgr, dtype=np.uint8))
        else:
            result = self._process_blank_custom(img_bgr)

        if annotate:
            from pipeline_debug import annotate_blank_result

            result["annotated_bgr"] = annotate_blank_result(result["rectified_bgr"], result)
        return result

    def _process_blank_custom(self, img_bgr: np.ndarray) -> dict:
        rectified, _, method = self.rectify_sheet(img_bgr)
        name_cells, question_rows = self.extract_cells(rectified)
        img_h, img_w = rectified.shape[:2]

        all_cells = list(name_cells)
        for q in question_rows:
            all_cells.extend(q["cells"])

        crops = []
        for cell in all_cells:
            x, y, w, h = cell["bbox"]
            x1, y1 = max(0, min(img_w, int(x))), max(0, min(img_h, int(y)))
            x2, y2 = max(x1, min(img_w, int(x + w))), max(y1, min(img_h, int(y + h)))
            crops.append(rectified[y1:y2, x1:x2])

        trim = float(self._cell_spec["default_margin_trim_pct"])
        preds = self._classify_custom_batch(crops, self.model, self.alphabet, trim, 3, False, False)

        def _summarize(group_cells: list[dict], group_preds: list[dict]):
            chars, out_cells, letters, conf_sum = [], [], 0, 0.0
            for c, r in zip(group_cells, group_preds):
                out_cells.append({
                    "cell_idx": c["cell_idx"], "char": r["char"], "confidence": r["confidence"],
                    "is_empty": r["is_empty"], "top3": r["top3"], "bbox": c["bbox"],
                })
                if not r["is_empty"]:
                    chars.append(r["char"])
                    letters += 1
                    conf_sum += r["confidence"]
            return "".join(chars), out_cells, letters, conf_sum

        offset = len(name_cells)
        name_raw, name_results, _, _ = _summarize(name_cells, preds[:offset])
        questions_results, total_letters, conf_sum = [], 0, 0.0
        for q in question_rows:
            q_len = len(q["cells"])
            q_text, q_cells, q_letters, q_conf = _summarize(q["cells"], preds[offset : offset + q_len])
            offset += q_len
            total_letters += q_letters
            conf_sum += q_conf
            questions_results.append({"q_num": q["q_num"], "text": q_text, "cells": q_cells})

        avg_conf = (conf_sum / max(1, total_letters)) if total_letters > 0 else 0.0
        return {
            "status": "success",
            "rectification_method": method,
            "student_name": name_raw.strip(),
            "student_name_cells": name_results,
            "questions": questions_results,
            "stats": {
                "total_letters": total_letters,
                "avg_confidence": round(avg_conf, 1),
                "questions_count": len(questions_results),
            },
            "rectified_bgr": rectified,
            "annotated_bgr": None,
        }

    def process_image(self, image_path, annotate: bool = False) -> dict:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.process_blank(img_bgr, annotate=annotate)

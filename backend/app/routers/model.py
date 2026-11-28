import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("tatar_ocr.model")

router = APIRouter(prefix="/model", tags=["Model & Manifest"])

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "pipeline_spec.json"

_DEFAULT_ALPHABET: List[str] = list("АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯ")


def _load_pipeline_spec() -> Dict[str, Any]:
    with open(SPEC_PATH, "r", encoding="utf-8") as spec_file:
        return json.load(spec_file)


try:
    TATAR_UPPERCASE_39: List[str] = list(_load_pipeline_spec()["model"]["alphabet"])
except Exception:
    TATAR_UPPERCASE_39 = _DEFAULT_ALPHABET


class PreprocessingSpec(BaseModel):
    input_resolution: List[int] = [64, 64]
    channels: int = 1
    color_mode: str = "grayscale"
    margin_trim_pct: float = 4.0
    connected_component_segmentation: bool = True
    ink_threshold_offset: float = 22.0
    ink_min_area_pixels: int = 35
    bounding_box_padding_px: int = 2
    target_occupancy_px: float = 48.0
    canvas_background_val: int = 250
    normalization_formula: str = "((pixel / 255.0) - 0.5) / 0.5"


class ModelManifestResponse(BaseModel):
    model_name: str = "tatar_ocr_v4_handwritten39_int8"
    model_version: str = "1.0"
    num_classes: int = 39
    alphabet_classes: List[str]
    sha256: str
    preprocessing: PreprocessingSpec


def _compute_file_sha256(file_path: Path) -> str:
    if not file_path.exists():
        logger.warning("ONNX model file not found for SHA-256 computation: %s", file_path)
        return ""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_manifest_data() -> ModelManifestResponse:
    spec = _load_pipeline_spec()
    model_spec = spec["model"]
    cell_spec = spec["cell"]

    rel_onnx = model_spec["default_onnx"]
    onnx_path = REPO_ROOT / rel_onnx
    alphabet_classes = list(model_spec["alphabet"])
    canvas_sz = int(cell_spec["canonical_canvas_size"])

    preprocessing = PreprocessingSpec(
        input_resolution=[canvas_sz, canvas_sz],
        margin_trim_pct=float(cell_spec["default_margin_trim_pct"]),
        ink_threshold_offset=float(cell_spec["ink_delta_high"]),
        ink_min_area_pixels=int(cell_spec["empty_min_ink_pixels"]),
        bounding_box_padding_px=int(cell_spec["glyph_crop_pad"]),
        target_occupancy_px=float(cell_spec["canonical_target_dim"]),
        canvas_background_val=int(cell_spec["canonical_bg_val"]),
    )

    return ModelManifestResponse(
        model_name=onnx_path.stem,
        model_version=str(spec.get("version", "1.0")),
        num_classes=len(alphabet_classes),
        alphabet_classes=alphabet_classes,
        sha256=_compute_file_sha256(onnx_path),
        preprocessing=preprocessing,
    )


@router.get("/manifest", response_model=ModelManifestResponse, summary="Get OCR Model Manifest and Preprocessing Spec")
async def get_model_manifest():
    return _load_manifest_data()

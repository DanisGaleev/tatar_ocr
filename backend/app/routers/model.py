from typing import List, Dict
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/model", tags=["Model & Manifest"])

class PreprocessingSpec(BaseModel):
    input_resolution: List[int] = [64, 64]
    channels: int = 1
    color_mode: str = "grayscale"
    margin_trim_pct: float = 11.0
    connected_component_segmentation: bool = True
    ink_threshold_offset: float = 22.0
    ink_min_area_pixels: int = 35
    bounding_box_padding_px: int = 2
    target_occupancy_px: float = 46.0
    canvas_background_val: int = 250
    normalization_formula: str = "((pixel / 255.0) - 0.5) / 0.5"

# Definitive 39 uppercase Tatar Cyrillic classes matching trained model weights
TATAR_UPPERCASE_39: List[str] = [
    "А", "Ә", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "Җ",
    "З", "И", "Й", "К", "Л", "М", "Н", "Ң", "О", "Ө",
    "П", "Р", "С", "Т", "У", "Ү", "Ф", "Х", "Һ", "Ц",
    "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"
]

RUSSIAN_UPPERCASE_33: List[str] = [
    "А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И",
    "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т",
    "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь",
    "Э", "Ю", "Я"
]

TATAR_ALLOWED_INDICES: List[int] = list(range(len(TATAR_UPPERCASE_39)))
RUSSIAN_ALLOWED_INDICES: List[int] = [
    idx for idx, ch in enumerate(TATAR_UPPERCASE_39) if ch in set(RUSSIAN_UPPERCASE_33)
]

class ModelManifestResponse(BaseModel):
    model_name: str = "finetuned_uppercase39"
    model_version: str = "1.0.0"
    num_classes: int = 39
    alphabet_classes: List[str]
    language_alphabets: Dict[str, List[str]] = Field(default_factory=dict)
    language_masks: Dict[str, List[int]] = Field(default_factory=dict)
    sha256: str = "b55481b407751ece8237c77bbfa6270cc3a271d73d04b3ef58a4c25ee01361d3"
    preprocessing: PreprocessingSpec

@router.get("/manifest", response_model=ModelManifestResponse, summary="Get OCR Model Manifest and Preprocessing Spec")
async def get_model_manifest():
    return ModelManifestResponse(
        model_name="finetuned_uppercase39",
        model_version="1.0.0",
        num_classes=39,
        alphabet_classes=TATAR_UPPERCASE_39,
        language_alphabets={
            "tt": TATAR_UPPERCASE_39,
            "ru": RUSSIAN_UPPERCASE_33,
        },
        language_masks={
            "tt": TATAR_ALLOWED_INDICES,
            "ru": RUSSIAN_ALLOWED_INDICES,
        },
        sha256="b55481b407751ece8237c77bbfa6270cc3a271d73d04b3ef58a4c25ee01361d3",
        preprocessing=PreprocessingSpec(),
    )

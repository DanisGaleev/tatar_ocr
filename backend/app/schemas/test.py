from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ConstructorTaskRef(BaseModel):
    task_id: str
    order: int

class AssembleTestRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=256)
    grade_level: int = Field(default=7, ge=5, le=9)
    generate_variants_count: int = Field(default=1, ge=1, le=4)
    task_items: Optional[List[ConstructorTaskRef]] = None
    shuffle_questions_in_variants: bool = True
    seed: Optional[int] = None

class ReadyTestItem(BaseModel):
    test_id: str
    title: str
    grade_level: int
    questions_count: int

class TemplateCellDimensions(BaseModel):
    width: float = 10.0
    height: float = 10.0

class TemplateGeometry(BaseModel):
    format: str = "A4"
    corner_aruco_dict: str = "DICT_4X4_50"
    corner_aruco_ids: List[int] = [0, 1, 2, 3]
    cell_dimensions_mm: TemplateCellDimensions = Field(default_factory=TemplateCellDimensions)

class ExpectedCell(BaseModel):
    index: int
    char: str
    unicode: str
    is_empty_allowed: bool = False

class OfflineBundleQuestion(BaseModel):
    question_number: int
    marker_id: int
    prompt: str
    topic_tag: str
    topic_name_tt: str
    cell_count: int
    expected_answer: str
    expected_cells: List[ExpectedCell]

class OfflineBundleVariant(BaseModel):
    variant_id: int
    qr_signature: str
    template_geometry: TemplateGeometry
    questions: List[OfflineBundleQuestion]

class OfflineBundleResponse(BaseModel):
    assignment_id: str
    title: str
    total_variants: int
    variants: List[OfflineBundleVariant]

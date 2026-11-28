from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class GradingScale(BaseModel):
    grade_5_min_pct: int = Field(default=85, ge=0, le=100, description="Min % for Grade 5")
    grade_4_min_pct: int = Field(default=70, ge=0, le=100, description="Min % for Grade 4")
    grade_3_min_pct: int = Field(default=50, ge=0, le=100, description="Min % for Grade 3")

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> "GradingScale":
        if not (self.grade_3_min_pct < self.grade_4_min_pct < self.grade_5_min_pct):
            raise ValueError(
                "Grading scale thresholds must be strictly increasing: "
                "grade_3_min_pct < grade_4_min_pct < grade_5_min_pct."
            )
        return self


class DeviceHandshakeRequest(BaseModel):
    device_os: str = Field(default="android", examples=["android"])
    app_version: str = Field(default="1.0.4", examples=["1.0.4"])
    teacher_name: str = Field(..., min_length=1, max_length=256, examples=["Каримова Гөлнара Илдар кызы"])
    school_name: str = Field(..., min_length=1, max_length=256, examples=["Гимназия №2 им. Ш. Марджани"])
    grading_scale: GradingScale = Field(default_factory=GradingScale)
    confidence_flag_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("teacher_name", "school_name")
    @classmethod
    def _strip_and_require_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace only.")
        return cleaned


class TeacherPreferences(BaseModel):
    grading_scale: GradingScale = Field(default_factory=GradingScale)
    confidence_flag_threshold: float = Field(default=0.65, examples=[0.65])


class DeviceHandshakeResponse(BaseModel):
    status: str = "active"
    teacher_uuid: str
    teacher_name: str
    preferences: TeacherPreferences

from pydantic import BaseModel, Field

class GradingScale(BaseModel):
    grade_5_min_pct: int = Field(default=85, description="Min % for Grade 5")
    grade_4_min_pct: int = Field(default=70, description="Min % for Grade 4")
    grade_3_min_pct: int = Field(default=50, description="Min % for Grade 3")

class DeviceHandshakeRequest(BaseModel):
    device_os: str = Field(default="android", examples=["android"])
    app_version: str = Field(default="1.0.4", examples=["1.0.4"])
    teacher_name: str = Field(..., examples=["Каримова Гөлнара Илдар кызы"])
    school_name: str = Field(..., examples=["Гимназия №2 им. Ш. Марджани"])
    grading_scale: GradingScale = Field(default_factory=GradingScale)

class TeacherPreferences(BaseModel):
    grading_scale: GradingScale = Field(default_factory=GradingScale)
    confidence_flag_threshold: float = Field(default=0.65, examples=[0.65])

class DeviceHandshakeResponse(BaseModel):
    status: str = "active"
    teacher_uuid: str
    teacher_name: str
    preferences: TeacherPreferences

from typing import List, Optional
from pydantic import BaseModel, Field

class ClassItem(BaseModel):
    class_id: str
    name: str
    subject: str = "tatar_language"
    academic_year: str = "2026-2027"
    student_count: int = 0

class ClassesListResponse(BaseModel):
    classes: List[ClassItem]

class StudentRosterItem(BaseModel):
    student_id: str
    last_name: str
    first_name: str
    middle_name: Optional[str] = ""
    full_name: str
    short_name: str

class ClassStudentsResponse(BaseModel):
    class_id: str
    class_name: str
    students: List[StudentRosterItem]

class BulkImportStudentsRequest(BaseModel):
    raw_text: str = Field(..., description="Pasted student names list, one per line")

class ImportedStudentItem(BaseModel):
    student_id: str
    full_name: str

class BulkImportStudentsResponse(BaseModel):
    added_count: int
    students: List[ImportedStudentItem]

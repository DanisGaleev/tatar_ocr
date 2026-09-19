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


class AssignTestToClassRequest(BaseModel):
    assignment_id: str
    due_date: Optional[str] = None
    status: Optional[str] = "active"


class ClassAssignmentItem(BaseModel):
    id: str
    class_id: str
    assignment_id: str
    assignment_title: str
    total_variants: int = 1
    assigned_at: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "active"
    total_students: int = 0
    checked_submissions_count: int = 0
    pending_submissions_count: int = 0
    average_score_pct: float = 0.0


class ClassAssignmentsResponse(BaseModel):
    class_id: str
    class_name: str
    assignments: List[ClassAssignmentItem]


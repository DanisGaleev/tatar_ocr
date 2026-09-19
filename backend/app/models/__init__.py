from app.core.database import Base
from app.models.task import TaskBankModel
from app.models.test import AssembledTestModel
from app.models.teacher import TeacherModel
from app.models.school import ClassModel, StudentModel
from app.models.submission import SubmissionModel

__all__ = [
    "Base",
    "TaskBankModel",
    "AssembledTestModel",
    "TeacherModel",
    "ClassModel",
    "StudentModel",
    "SubmissionModel",
]

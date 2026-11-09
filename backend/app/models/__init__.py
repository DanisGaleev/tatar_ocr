from app.core.database import Base
from app.models.task import TaskBankModel
from app.models.test import AssembledTestModel
from app.models.teacher import TeacherModel
from app.models.school import ClassModel, StudentModel, ClassAssignmentModel
from app.models.submission import SubmissionModel
from app.models.user import UserModel
from app.models.session import UserSessionModel
from app.models.oauth import OAuthAccountModel
from app.models.verification import VerificationCodeModel
from app.models.audit import AuditLogModel

__all__ = [
    "Base",
    "TaskBankModel",
    "AssembledTestModel",
    "TeacherModel",
    "ClassModel",
    "StudentModel",
    "ClassAssignmentModel",
    "SubmissionModel",
    "UserModel",
    "UserSessionModel",
    "OAuthAccountModel",
    "VerificationCodeModel",
    "AuditLogModel",
]

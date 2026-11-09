from app.repositories.base import BaseRepository
from app.repositories.user_repo import UserRepository
from app.repositories.session_repo import SessionRepository
from app.repositories.verification_repo import VerificationCodeRepository
from app.repositories.class_repo import ClassRepository
from app.repositories.student_repo import StudentRepository
from app.repositories.submission_repo import SubmissionRepository
from app.repositories.assignment_repo import AssignmentRepository
from app.repositories.audit_repo import AuditRepository
from app.repositories.oauth_repo import OAuthRepository
from app.repositories.teacher_repo import TeacherRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "SessionRepository",
    "VerificationCodeRepository",
    "ClassRepository",
    "StudentRepository",
    "SubmissionRepository",
    "AssignmentRepository",
    "AuditRepository",
    "OAuthRepository",
    "TeacherRepository",
]

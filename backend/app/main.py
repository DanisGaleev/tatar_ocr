import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import select

from app.core.config import settings
from app.core.database import init_db, AsyncSessionLocal
from app.models.task import TaskBankModel
from app.models.test import AssembledTestModel
from app.models.teacher import TeacherModel
from app.models.school import ClassModel, StudentModel
from app.routers import constructor, assignments, auth, classes, submissions, analytics, reports, web, ocr, model
from app.generators.registry import registry
from app.generators.phonetics import build_expected_cells

INITIAL_BANK_TASKS = [
    {
        "task_id": "tsk_8201",
        "prompt_tt": "Куегыз сүзне юнәлеш килешендә: китап ->",
        "expected_answer": "КИТАПКА",
        "cell_count": 8,
        "grade_level": 7,
        "topic_tag": "case_dative",
        "topic_name_tt": "Юнәлеш килеше",
        "task_type": "case_inflection",
        "is_public_in_bank": True,
    },
    {
        "task_id": "tsk_8202",
        "prompt_tt": "Куегыз сүзне чыгыш килешендә: өстәл ->",
        "expected_answer": "ӨСТӘЛДӘН",
        "cell_count": 8,
        "grade_level": 7,
        "topic_tag": "case_ablative",
        "topic_name_tt": "Чыгыш килеше",
        "task_type": "case_inflection",
        "is_public_in_bank": True,
    },
    {
        "task_id": "tsk_8203",
        "prompt_tt": "Сүзгә күплек сан кушымчасын ялгагыз: бала ->",
        "expected_answer": "БАЛАЛАР",
        "cell_count": 8,
        "grade_level": 5,
        "topic_tag": "plural_affixes",
        "topic_name_tt": "Күплек сан",
        "task_type": "plural_affixes",
        "is_public_in_bank": True,
    },
    {
        "task_id": "tsk_8204",
        "prompt_tt": "Сүзгә күплек сан кушымчасын ялгагыз: урман ->",
        "expected_answer": "УРМАННАР",
        "cell_count": 8,
        "grade_level": 5,
        "topic_tag": "plural_affixes",
        "topic_name_tt": "Күплек сан",
        "task_type": "plural_affixes",
        "is_public_in_bank": True,
    },
    {
        "task_id": "tsk_8205",
        "prompt_tt": "Антонимны (кире мәгънәне) табыгыз: ялган ->",
        "expected_answer": "ХАКЛЫК",
        "cell_count": 6,
        "grade_level": 6,
        "topic_tag": "antonyms_adjectives",
        "topic_name_tt": "Сыйфат антонимнары",
        "task_type": "antonyms",
        "is_public_in_bank": True,
    },
    {
        "task_id": "tsk_8206",
        "prompt_tt": "Тәрҗемә итегез (спасибо): ->",
        "expected_answer": "РӘХМӘТ",
        "cell_count": 7,
        "grade_level": 5,
        "topic_tag": "vocabulary_translation",
        "topic_name_tt": "Сүзлек байлыгы (тәрҗемә)",
        "task_type": "translation",
        "is_public_in_bank": True,
    },
]

INITIAL_STUDENTS = [
    ("stu_01", "Галиев", "Амир", "Рустемович", "Галиев Амир Рустемович", "Галиев А."),
    ("stu_02", "Закирова", "Ләйсән", "Ильдаровна", "Закирова Ләйсән Ильдаровна", "Закирова Л."),
    ("stu_03", "Хабибуллин", "Тимур", "Маратович", "Хабибуллин Тимур Маратович", "Хабибуллин Т."),
    ("stu_04", "Сафина", "Дилә", "Нияз кызы", "Сафина Дилә Нияз кызы", "Сафина Д."),
    ("stu_05", "Исмәгыйлев", "Булат", "Айдар улы", "Исмәгыйлев Булат Айдар улы", "Исмәгыйлев Б."),
]

async def seed_initial_data():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TaskBankModel).limit(1))
        if not result.scalar_one_or_none():
            for t in INITIAL_BANK_TASKS:
                task_obj = TaskBankModel(**t)
                session.add(task_obj)
            await session.commit()

        # Seed initial teacher
        t_res = await session.execute(select(TeacherModel).limit(1))
        if not t_res.scalar_one_or_none():
            sample_teacher = TeacherModel(
                teacher_uuid="c56a4180-65aa-42ec-a945-5fd21dec0538",
                device_os="android",
                app_version="1.0.4",
                teacher_name="Каримова Гөлнара Илдар кызы",
                school_name="Гимназия №2 им. Ш. Марджани",
                grade_5_min_pct=85,
                grade_4_min_pct=70,
                grade_3_min_pct=50,
                confidence_flag_threshold=0.65,
            )
            session.add(sample_teacher)
            await session.commit()

        # Seed initial class
        c_res = await session.execute(select(ClassModel).where(ClassModel.class_id == "cls_7a_2026"))
        if not c_res.scalar_one_or_none():
            sample_class = ClassModel(
                class_id="cls_7a_2026",
                teacher_uuid="c56a4180-65aa-42ec-a945-5fd21dec0538",
                name="7-А",
                subject="tatar_language",
                academic_year="2026-2027",
            )
            session.add(sample_class)
            await session.commit()

            for sid, ln, fn, mn, fulln, shn in INITIAL_STUDENTS:
                stu = StudentModel(
                    student_id=sid,
                    class_id="cls_7a_2026",
                    last_name=ln,
                    first_name=fn,
                    middle_name=mn,
                    full_name=fulln,
                    short_name=shn,
                )
                session.add(stu)
            await session.commit()

        # Seed initial assembled test
        asm_res = await session.execute(select(AssembledTestModel).where(AssembledTestModel.test_id == "TAT-2026-Q1"))
        existing_asm = asm_res.scalar_one_or_none()
        
        template_geometry = {
            "format": "A4",
            "corner_aruco_dict": settings.ARUCO_DICT,
            "corner_aruco_ids": settings.CORNER_ARUCO_IDS,
            "cell_dimensions_mm": {
                "width": settings.CELL_WIDTH_MM,
                "height": settings.CELL_HEIGHT_MM,
            },
        }

        v1_questions = [
            {"question_number": 1, "marker_id": 11, "prompt": "Куегыз сүзне юнәлеш килешендә: китап ->", "topic_tag": "case_dative", "topic_name_tt": "Юнәлеш килеше", "cell_count": 8, "expected_answer": "КИТАПКА", "expected_cells": build_expected_cells("КИТАПКА", 8)},
            {"question_number": 2, "marker_id": 12, "prompt": "Куегыз сүзне чыгыш килешендә: өстәл ->", "topic_tag": "case_ablative", "topic_name_tt": "Чыгыш килеше", "cell_count": 8, "expected_answer": "ӨСТӘЛДӘН", "expected_cells": build_expected_cells("ӨСТӘЛДӘН", 8)},
            {"question_number": 3, "marker_id": 13, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: бала ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "БАЛАЛАР", "expected_cells": build_expected_cells("БАЛАЛАР", 8)},
            {"question_number": 4, "marker_id": 14, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: урман ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "УРМАННАР", "expected_cells": build_expected_cells("УРМАННАР", 8)}
        ]

        v2_questions = [
            {"question_number": 1, "marker_id": 11, "prompt": "Куегыз сүзне чыгыш килешендә: өстәл ->", "topic_tag": "case_ablative", "topic_name_tt": "Чыгыш килеше", "cell_count": 8, "expected_answer": "ӨСТӘЛДӘН", "expected_cells": build_expected_cells("ӨСТӘЛДӘН", 8)},
            {"question_number": 2, "marker_id": 12, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: бала ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "БАЛАЛАР", "expected_cells": build_expected_cells("БАЛАЛАР", 8)},
            {"question_number": 3, "marker_id": 13, "prompt": "Сүзгә күплек сан кушымчасын ялгагыз: урман ->", "topic_tag": "plural_affixes", "topic_name_tt": "Күплек сан", "cell_count": 8, "expected_answer": "УРМАННАР", "expected_cells": build_expected_cells("УРМАННАР", 8)},
            {"question_number": 4, "marker_id": 14, "prompt": "Куегыз сүзне юнәлеш килешендә: китап ->", "topic_tag": "case_dative", "topic_name_tt": "Юнәлеш килеше", "cell_count": 8, "expected_answer": "КИТАПКА", "expected_cells": build_expected_cells("КИТАПКА", 8)}
        ]

        bundle_data = {
            "assignment_id": "TAT-2026-Q1",
            "title": "Татар теле. 7 сыйныф. Исем килешләре һәм кушымчалар",
            "total_variants": 2,
            "variants": [
                {
                    "variant_id": 1,
                    "qr_signature": '{"tid":"TAT-2026-Q1","var":1,"page":1,"tot":1,"n_q":4}',
                    "template_geometry": template_geometry,
                    "questions": v1_questions
                },
                {
                    "variant_id": 2,
                    "qr_signature": '{"tid":"TAT-2026-Q1","var":2,"page":1,"tot":1,"n_q":4}',
                    "template_geometry": template_geometry,
                    "questions": v2_questions
                }
            ]
        }

        if not existing_asm:
            sample_asm = AssembledTestModel(
                test_id="TAT-2026-Q1",
                title="Татар теле. 7 сыйныф. Исем килешләре һәм кушымчалар",
                grade_level=7,
                total_variants=2,
                bundle_json=json.dumps(bundle_data, ensure_ascii=False),
            )
            session.add(sample_asm)
        else:
            existing_asm.bundle_json = json.dumps(bundle_data, ensure_ascii=False)
            session.add(existing_asm)
        await session.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_initial_data()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="FastAPI Backend for «Дәресханә» (Tatar OCR) with Deterministic Task Generator",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "status_code": 500},
    )

app.include_router(web.router)
app.include_router(constructor.router, prefix=settings.API_V1_STR)
app.include_router(assignments.router, prefix=settings.API_V1_STR)
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(classes.router, prefix=settings.API_V1_STR)
app.include_router(submissions.router, prefix=settings.API_V1_STR)
app.include_router(analytics.router, prefix=settings.API_V1_STR)
app.include_router(reports.router, prefix=settings.API_V1_STR)
app.include_router(model.router, prefix=settings.API_V1_STR)
app.include_router(ocr.router, prefix=settings.API_V1_STR)
app.include_router(ocr.legacy_router)

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}


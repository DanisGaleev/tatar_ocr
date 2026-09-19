# «Кара куян» (Tatar OCR) — API Contract Specification

This document provides the definitive API contracts and architectural design for **«Кара куян»**, built around two foundational principles:
1. **Zero-Photo Transmission**: Camera images never leave the teacher's phone. All ArUco perspective rectifications, cell cropping, and TFLite neural network character recognitions happen 100% locally on the device in memory.
2. **Backend Analytical Processing**: The phone uploads only lightweight structured verification results (1–2 KB JSON per sheet). The backend performs all heavy historical data aggregation, student learning curve analysis, class heatmaps, error categorizations, and gradebook exports.

---

## 1. System Architecture & Zero-Photo Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       1. PREPARATION (Online / Web or App)                 │
│                                                                             │
│  Teacher Web / Mobile:                                                      │
│    ├──> GET /api/v1/classes                      (Download student roster)  │
│    ├──> GET /api/v1/assignments/{id}/batch-blanks (Print personalized PDFs) │
│    └──> GET /api/v1/assignments/{id}/offline-bundle (Cache answer key & geo)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      2. GRADING IN CLASSROOM (100% Offline)                 │
│                                                                             │
│  Teacher's Phone (Photos NEVER leave the device):                            │
│    1. Camera frame captured in RAM                                          │
│    2. Local OpenCV rectifies ArUco corners                                  │
│    3. Local QR scanner decodes {"tid":"TAT-Q1", "var":1, "stu_id":"stu_01"} │
│    4. Compares 10x10 mm letter crops against cached Offline Bundle          │
│    5. Local TFLite runs inference -> Displays Green / Red / Flag on screen   │
│    6. Teacher taps questionable cells to confirm / override (HITL)          │
│    7. Calculates score & grade; Photo is discarded from RAM                 │
│    8. Graded structured JSON stored in phone's local SQLite                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    3. ANALYTICS & SYNC (Zero-Photo Upload)                  │
│                                                                             │
│  Teacher Phone:                                                             │
│    └──> POST /api/v1/submissions/batch-sync                                 │
│         Payload: Only numbers, letters, statuses, confidences (~1.5 KB)     │
│         NO IMAGES SENT                                                      │
│                                                                             │
│  Backend Analytics Engine:                                                  │
│    ├── Aggregates student error curves & letter confusion matrices          │
│    ├── Builds class-wide grammatical topic heatmaps                         │
│    ├── Updates teacher gradebook & leaderboard                              │
│    └── Exports official Excel sheets (edu.tatar.ru / МЭШ)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. API Overview & Router Directory

| Prefix | Router Name | Description |
| :--- | :--- | :--- |
| `/api/v1/auth` | **Teacher Identity** | Teacher UUID handshake, default grading scales, and settings |
| `/api/v1/classes` | **Classes & Students** | Manage classes and student rosters with real names (bulk import, sync) |
| `/api/v1/assignments` | **Assignment Delivery** | Get tests, download ready A4 printable PDF blanks, **Offline Bundle** |
| `/api/v1/constructor` | **Assignment Constructor** | Search task bank, create custom tasks, assemble modular tests & variants |
| `/api/v1/submissions` | **Submissions Ingestion** | Ingesting locally graded structured JSON (Zero-Photo), batch sync |
| `/api/v1/analytics` | **Backend Analytics Engine**| Student curves, class topic heatmaps, difficult Tatar letters |
| `/api/v1/reports` | **Gradebook Exports** | Export class performance sheets for electronic school journals (Excel) |
| `/api/v1/model` | **Model & Manifest** | Model weights manifest, preprocessing specs, and 39-class alphabet array |

---

## 3. Detailed Router Specifications

---

### Router 0: Teacher Identity & Handshake (`/api/v1/auth`)

#### `POST /api/v1/auth/device-handshake`
Registers or verifies the teacher device UUID generated upon first install.

- **Headers**:
  - `X-Teacher-UUID: <uuid-v4>` (Required)
- **Request Body**:
```json
{
  "device_os": "android",
  "app_version": "1.0.4",
  "teacher_name": "Каримова Гөлнара Илдар кызы",
  "school_name": "Гимназия №2 им. Ш. Марджани",
  "grading_scale": {
    "grade_5_min_pct": 85,
    "grade_4_min_pct": 70,
    "grade_3_min_pct": 50
  }
}
```
- **Response `200 OK`**:
```json
{
  "status": "active",
  "teacher_uuid": "c56a4180-65aa-42ec-a945-5fd21dec0538",
  "teacher_name": "Каримова Гөлнара Илдар кызы",
  "preferences": {
    "grading_scale": {
      "grade_5_min_pct": 85,
      "grade_4_min_pct": 70,
      "grade_3_min_pct": 50
    },
    "confidence_flag_threshold": 0.65
  }
}
```

---

### Router 1: Classes & Student Rosters (`/api/v1/classes`)

#### 1.1 `GET /api/v1/classes`
List all classes belonging to the teacher.

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Response `200 OK`**:
```json
{
  "classes": [
    {
      "class_id": "cls_7a_2026",
      "name": "7-А",
      "subject": "tatar_language",
      "academic_year": "2026-2027",
      "student_count": 25
    }
  ]
}
```

#### 1.2 `GET /api/v1/classes/{class_id}/students`
Returns the student roster for caching on the mobile device.

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Response `200 OK`**:
```json
{
  "class_id": "cls_7a_2026",
  "class_name": "7-А",
  "students": [
    {
      "student_id": "stu_01",
      "last_name": "Галиев",
      "first_name": "Амир",
      "middle_name": "Рустемович",
      "full_name": "Галиев Амир Рустемович",
      "short_name": "Галиев А."
    },
    {
      "student_id": "stu_02",
      "last_name": "Закирова",
      "first_name": "Ләйсән",
      "middle_name": "Ильдаровна",
      "full_name": "Закирова Ләйсән Ильдаровна",
      "short_name": "Закирова Л."
    }
  ]
}
```

#### 1.3 `POST /api/v1/classes/{class_id}/students/bulk-import`
Batch import students from a pasted plain-text list (from school journal or Excel).

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Request Body**:
```json
{
  "raw_text": "Галиев Амир Рустемович\nЗакирова Ләйсән Ильдаровна\nХабибуллин Тимур Маратович"
}
```
- **Response `201 Created`**:
```json
{
  "added_count": 3,
  "students": [
    { "student_id": "stu_01", "full_name": "Галиев Амир Рустемович" },
    { "student_id": "stu_02", "full_name": "Закирова Ләйсән Ильдаровна" },
    { "student_id": "stu_03", "full_name": "Хабибуллин Тимур Маратович" }
  ]
}
```

---

### Router 2: Assignment Delivery & Offline Bundles (`/api/v1/assignments`)

#### 2.1 Get Assignment Metadata
`GET /api/v1/assignments/{assignment_id}`
- **Response `200 OK`**: Metadata, title, grade level, topic tags, points.

#### 2.2 Download Printable Blank PDF (Generic or Single Student)
`GET /api/v1/assignments/{assignment_id}/blank.pdf`
- **Query Parameters**:
  - `variant` (integer, default: 1)
  - `student_id` (string, optional): Pre-prints student name in header boxes.
- **Response**: `application/pdf` binary stream.

#### 2.3 Download Batch Blank PDFs for Class
`GET /api/v1/assignments/{assignment_id}/batch-blanks.pdf`
- **Query Parameters**:
  - `class_id` (string, required): e.g., `cls_7a_2026`
  - `alternate_variants` (boolean, default: true)
- **Response**: `application/pdf` where each sheet is pre-printed with the student's name and personalized QR code.

#### 2.4 Download Offline Verification Bundle (Ground-Truth Answer Key)
`GET /api/v1/assignments/{assignment_id}/offline-bundle`
- Downloads everything the mobile app needs to perform **100% offline grading and answer validation on the phone**.
- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Query Parameters**:
  - `variant` (integer, optional)
- **Response `200 OK`**:
```json
{
  "assignment_id": "TAT-2026-Q1",
  "title": "Татар теле. 7 сыйныф. Исем килешләре",
  "total_variants": 2,
  "variants": [
    {
      "variant_id": 1,
      "qr_signature": "{\"tid\":\"TAT-2026-Q1\",\"var\":1,\"page\":1,\"tot\":1,\"n_q\":8}",
      "template_geometry": {
        "format": "A4",
        "corner_aruco_dict": "DICT_4X4_50",
        "corner_aruco_ids": [0, 1, 2, 3],
        "cell_dimensions_mm": { "width": 10.0, "height": 10.0 }
      },
      "questions": [
        {
          "question_number": 1,
          "marker_id": 11,
          "prompt": "Куегыз сүзне юнәлеш килешендә: китап ->",
          "topic_tag": "case_dative",
          "topic_name_tt": "Юнәлеш килеше",
          "cell_count": 8,
          "expected_answer": "КИТАПКА",
          "expected_cells": [
            { "index": 0, "char": "К", "unicode": "U+041A" },
            { "index": 1, "char": "И", "unicode": "U+0418" },
            { "index": 2, "char": "Т", "unicode": "U+0422" },
            { "index": 3, "char": "А", "unicode": "U+0410" },
            { "index": 4, "char": "П", "unicode": "U+041F" },
            { "index": 5, "char": "К", "unicode": "U+041A" },
            { "index": 6, "char": "А", "unicode": "U+0410" },
            { "index": 7, "char": " ", "unicode": "U+0020", "is_empty_allowed": true }
          ]
        },
        {
          "question_number": 2,
          "marker_id": 12,
          "prompt": "Куегыз сүзне чыгыш килешендә: өстәл ->",
          "topic_tag": "case_ablative",
          "topic_name_tt": "Чыгыш килеше",
          "cell_count": 8,
          "expected_answer": "ӨСТӘЛДӘН",
          "expected_cells": [
            { "index": 0, "char": "Ө", "unicode": "U+04E8" },
            { "index": 1, "char": "С", "unicode": "U+0421" },
            { "index": 2, "char": "Т", "unicode": "U+0422" },
            { "index": 3, "char": "Ә", "unicode": "U+04D8" },
            { "index": 4, "char": "Л", "unicode": "U+041B" },
            { "index": 5, "char": "Д", "unicode": "U+0414" },
            { "index": 6, "char": "Ә", "unicode": "U+04D8" },
            { "index": 7, "char": "Н", "unicode": "U+041D" }
          ]
        }
      ]
    }
  ]
}
```

---

### Router 3: Assignment Constructor & Bank (`/api/v1/constructor`)

#### 3.1 Search Task Bank
`GET /api/v1/constructor/tasks`
- **Query Params**: `query`, `topic_tag`, `grade_level` (5–9), `max_cells` (<= 12), `scope` (`all` | `my_tasks`).

#### 3.2 Create Custom Single Task
`POST /api/v1/constructor/tasks`
- **Request Body**:
```json
{
  "prompt_tt": "Сүзгә күплек сан кушымчасын ялгагыз: дус ->",
  "expected_answer": "ДУСЛАР",
  "cell_count": 6,
  "grade_level": 5,
  "topic_tag": "plural_affixes",
  "topic_name_tt": "Күплек сан",
  "is_public_in_bank": false
}
```

#### 3.3 Browse Ready-Made Complete Tests
`GET /api/v1/constructor/ready-tests`
- Curated 8-question sheets with pre-configured A4 templates.

#### 3.4 Assemble Modular Test
`POST /api/v1/constructor/tests`
- Composes a new test from selected task IDs with A4 physical limit validation (maximum 8 modular question blocks).
- **Request Body**:
```json
{
  "title": "7 сыйныф. 1 нче чирек контроль эше",
  "grade_level": 7,
  "generate_variants_count": 2,
  "task_items": [
    { "task_id": "tsk_8201", "order": 1 },
    { "task_id": "tsk_8202", "order": 2 }
  ],
  "shuffle_questions_in_variants": true
}
```

#### 3.5 Fork / Clone Test
`POST /api/v1/constructor/tests/{test_id}/fork`

---

### Router 4: Submissions Ingestion & Batch Sync (`/api/v1/submissions`)

**Zero-Photo Policy**: Photos remain on the phone. Only structured results (characters, booleans, scores, overrides) are ingested here.

#### 4.1 Batch Sync Graded Submissions (Class Upload)
`POST /api/v1/submissions/batch-sync`
Transmits the entire set of graded papers once the teacher is back online. Average payload for a class of 25 students is **only ~35 KB**.

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Request Body**:
```json
{
  "assignment_id": "TAT-2026-Q1",
  "class_id": "cls_7a_2026",
  "synced_at": "2026-09-18T13:00:00Z",
  "submissions": [
    {
      "client_submission_uuid": "d290f1ee-6c54-4b01-90e6-d701748f0851",
      "student_id": "stu_01",
      "student_name": "Галиев Амир Рустемович",
      "variant": 1,
      "checked_at": "2026-09-18T12:20:10Z",
      "overall_score": 7.0,
      "max_score": 8.0,
      "final_grade": 4,
      "teacher_reviewed_flags": 1,
      "questions_results": [
        {
          "question_number": 1,
          "marker_id": 11,
          "topic_tag": "case_dative",
          "is_correct": true,
          "points_earned": 1.0,
          "cells": [
            {
              "cell_index": 0,
              "expected_char": "К",
              "predicted_char": "К",
              "confidence": 0.96,
              "status": "MATCH"
            },
            {
              "cell_index": 1,
              "expected_char": "И",
              "predicted_char": "И",
              "confidence": 0.89,
              "status": "MATCH"
            }
          ]
        },
        {
          "question_number": 2,
          "marker_id": 12,
          "topic_tag": "case_ablative",
          "is_correct": false,
          "points_earned": 0.0,
          "cells": [
            {
              "cell_index": 5,
              "expected_char": "Д",
              "predicted_char": "Т",
              "confidence": 0.54,
              "status": "FLAG_OVERRIDDEN_BY_TEACHER",
              "teacher_override": "WRONG"
            }
          ]
        }
      ]
    }
  ]
}
```
- **Response `200 OK`**:
```json
{
  "status": "synced",
  "received_count": 1,
  "inserted_count": 1,
  "updated_count": 0,
  "analytics_recalculated": true
}
```

#### 4.2 Save Single Graded Submission
`POST /api/v1/submissions`
Used for live upload when checking while online.

#### 4.3 Query Submissions
`GET /api/v1/submissions`
- **Query Params**: `class_id`, `assignment_id`, `student_id`.

---

### Router 5: Backend Analytics Engine (`/api/v1/analytics`)

The backend continuously crunches the structured verification data to provide actionable classroom insights.

#### 5.1 Student Analytics (Individual)
`GET /api/v1/analytics/students/{student_id}`
Computes learning curves, error frequencies by topic, and specific letter confusions for an individual student.

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Response `200 OK`**:
```json
{
  "student_id": "stu_01",
  "full_name": "Галиев Амир Рустемович",
  "class_name": "7-А",
  "total_tests_completed": 12,
  "average_score_pct": 82.5,
  "average_grade": 4.3,
  "grade_distribution": { "5": 5, "4": 5, "3": 2, "2": 0 },
  "frequent_weak_topics": [
    {
      "topic_code": "case_locative",
      "topic_name_tt": "Урын-вакыт килеше (-да/-дә, -та/-тә)",
      "total_questions": 10,
      "wrong_count": 4,
      "accuracy_pct": 60.0
    },
    {
      "topic_code": "plural_affixes",
      "topic_name_tt": "Күплек сан кушымчалары (-лар/-ләр)",
      "total_questions": 8,
      "wrong_count": 2,
      "accuracy_pct": 75.0
    }
  ],
  "problematic_letters": [
    {
      "letter": "Ң",
      "total_occurrences": 14,
      "misrecognized_or_wrong": 4,
      "accuracy_pct": 71.4,
      "common_confusions": ["Н"]
    },
    {
      "letter": "Ә",
      "total_occurrences": 18,
      "misrecognized_or_wrong": 3,
      "accuracy_pct": 83.3,
      "common_confusions": ["А"]
    }
  ],
  "history": [
    {
      "submission_id": "sub_8f23a",
      "assignment_id": "TAT-2026-Q1",
      "assignment_title": "Исем килешләре",
      "date": "2026-09-17T09:30:00Z",
      "score": 7,
      "max_score": 8,
      "grade": 4
    }
  ]
}
```

#### 5.2 Class Aggregated Analytics & Student Leaderboard
`GET /api/v1/analytics/classes/{class_id}`
Class heatmap: identifies which grammar rules the teacher must re-explain on the next lesson, plus a full pupil performance table.

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Response `200 OK`**:
```json
{
  "class_id": "cls_7a_2026",
  "class_name": "7-А",
  "students_count": 25,
  "average_class_score_pct": 81.2,
  "grade_distribution": { "5": 9, "4": 11, "3": 4, "2": 1 },
  "top_class_mistakes": [
    {
      "topic_code": "case_ablative",
      "topic_name_tt": "Чыгыш килешендә сингармонизм ялгышлары (-дан/-дән, -тан/-тән)",
      "failure_rate_pct": 42.3,
      "affected_students_count": 11
    },
    {
      "topic_code": "antonyms_adjectives",
      "topic_name_tt": "Сыйфат антонимнары (ялган - хаклык)",
      "failure_rate_pct": 34.6,
      "affected_students_count": 9
    }
  ],
  "difficult_characters_across_class": [
    { "letter": "Ң", "error_rate_pct": 36.5 },
    { "letter": "Ө", "error_rate_pct": 28.0 }
  ],
  "students_performance_table": [
    {
      "student_id": "stu_02",
      "full_name": "Закирова Ләйсән Ильдаровна",
      "average_score_pct": 94.0,
      "average_grade": 4.8,
      "tests_completed": 8
    },
    {
      "student_id": "stu_01",
      "full_name": "Галиев Амир Рустемович",
      "average_score_pct": 82.5,
      "average_grade": 4.3,
      "tests_completed": 8
    }
  ]
}
```

#### 5.3 Assignment Analytics
`GET /api/v1/analytics/assignments/{assignment_id}`
Item analysis per question across all completed sheets.

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Response `200 OK`**:
```json
{
  "assignment_id": "TAT-2026-Q1",
  "title": "Исем килешләре һәм кушымчалар",
  "total_submissions": 25,
  "average_score_pct": 83.2,
  "questions_analytics": [
    {
      "question_number": 1,
      "marker_id": 11,
      "prompt": "Куегыз сүзне юнәлеш килешендә: китап ->",
      "expected_answer": "КИТАПКА",
      "accuracy_pct": 92.0,
      "wrong_submissions": [
        { "student_id": "stu_03", "student_name": "Хабибуллин Тимур", "written_answer": "КИТАПГА" }
      ]
    }
  ]
}
```

---

### Router 6: Electronic Gradebook Export (`/api/v1/reports`)

#### 6.1 `GET /api/v1/reports/assignments/{assignment_id}/gradebook.xlsx`
Downloads an Excel spreadsheet ready for direct upload into school electronic gradebooks (e.g., edu.tatar.ru or МЭШ).

- **Headers**: `X-Teacher-UUID: <uuid-v4>`
- **Query Parameters**:
  - `class_id`: `cls_7a_2026`
  - `format`: `xlsx` (or `csv`)
- **Response `200 OK`**:
  - `Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
  - Columns: Student Full Name, Variant, Points Earned, Max Points, Grade (2–5), Date.

---

### Router 7: Model Manifest & Preprocessing Spec (`/api/v1/model`)

#### 7.1 `GET /api/v1/model/manifest`
Returns the authoritative 39-class alphabet array, model version info, SHA-256 hash, and exact image preprocessing parameters to ensure 100% parity between server training and mobile on-device TFLite/ONNX inference.

- **Response `200 OK`**:
```json
{
  "model_name": "finetuned_uppercase39",
  "model_version": "1.0.0",
  "num_classes": 39,
  "alphabet_classes": [
    "А", "Ә", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "Җ",
    "З", "И", "Й", "К", "Л", "М", "Н", "Ң", "О", "Ө",
    "П", "Р", "С", "Т", "У", "Ү", "Ф", "Х", "Һ", "Ц",
    "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я"
  ],
  "sha256": "b55481b407751ece8237c77bbfa6270cc3a271d73d04b3ef58a4c25ee01361d3",
  "preprocessing": {
    "input_resolution": [64, 64],
    "channels": 1,
    "color_mode": "grayscale",
    "margin_trim_pct": 11.0,
    "connected_component_segmentation": true,
    "ink_threshold_offset": 22.0,
    "ink_min_area_pixels": 35,
    "bounding_box_padding_px": 2,
    "target_occupancy_px": 46.0,
    "canvas_background_val": 250,
    "normalization_formula": "((pixel / 255.0) - 0.5) / 0.5"
  }
}
```

> **Authoritative Class Indexing Note**:
> In standard Tatar Cyrillic orthography, the 6 special Tatar letters (**Ә, Җ, Ң, Ө, Ү, Һ**) appear immediately after their base Cyrillic counterparts:
> - Index 0: `А`, Index 1: `Ә`
> - Index 8: `Ж`, Index 9: `Җ`
> - Index 16: `Н`, Index 17: `Ң`
> - Index 18: `О`, Index 19: `Ө`
> - Index 24: `У`, Index 25: `Ү`
> - Index 27: `Х`, Index 28: `Һ`
>
> Never append `Ә` to the end of the alphabet array, as that causes catastrophic off-by-one indexing against the trained neural network output logits.


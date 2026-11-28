# Independent Fresh-Context Review — Stage 2 (`backend-ocr-pdf-generators-contract`)

**Verdict:** `accept-with-notes`

Stage 2 satisfies all acceptance criteria (`AC-1` through `AC-7`) and verification gates (`GATE-NATIVE`, `GATE-BENCHMARK`, `GATE-BACKEND`, `GATE-SPECS`) with zero Critical or Major blockers. Independent verification in this read-only review confirmed:
- `pytest -p no:cacheprovider backend\tests -q`: **157 passed, 7 warnings in 17.18s** (exit code `0`).
- `v3\evaluate_blanks.py --model models\tatar_ocr_v4_handwritten39_int8.onnx`: **`Acc_char = 96.88% (435/449)`**, **`CER = 3.12% (14 edits)`**, **`R_Tatar = 97.33% (73/75)`**, **`FEM = 78.95% (45/57)`**, **`FFM (Lev<=1) = 96.49% (55/57)`**, **`HCER_80% = 1.78%`** (0-diff parity vs `HEAD`-equivalent baseline).

---

## Per-Lens Assessment

### 1. API & Zero-Photo / 152-ФЗ — **PASS**
- **Zero-Photo & 152-ФЗ compliance:** Student worksheet camera frames remain strictly on-device (`backend/app/routers/submissions.py` ingests structured JSON only). `backend/app/routers/constructor.py` (`/scan-task` and `/scan-task-base64`) processes textbook exercise images strictly in memory (`image_bytes`) via `TaskExtractorService` without writing raw images to disk or DB.
- **`backend/app/routers/assignments.py`:**
  - `GET /assignments/{assignment_id}/blank.pdf` enforces `variant: int = Query(1, ge=1)` (returns `422` on `variant=0`), returns `404` on unknown `assignment_id`, unknown `variant`, or unknown `student_id`, guards `json.loads(test_obj.bundle_json)` (`500`), sanitizes `Content-Disposition` filename, and passes `student_id` to `render_blank_pdf`.
  - `GET /assignments` and `GET /assignments/{assignment_id}` guard `created_at` / `assigned_at` / `due_date` nullability before `.isoformat()`.
- **`backend/app/routers/constructor.py` & `backend/app/schemas/task.py`:**
  - `TaskBankItem` and `CreateTaskRequest` include `task_type: Optional[str] = "custom"`; `POST /constructor/tasks` validates `expected_answer` with `validate_ocr_answer` (no digit-only bypass) and rejects cell overflow (`422`).
  - `POST /constructor/generate` inspects `inspect.signature(gen.generate).parameters` and returns `422` when `case_code` or `custom_stems` are unsupported for the selected `task_type`.
  - `POST /constructor/tests` sorts `payload.task_items` by `order` and produces distinct Variants 2–4 when `shuffle_questions_in_variants=True` (seeded `random.Random(seed + v_idx * 1337)` with cyclic-shift fallback if unchanged).
  - `GET /constructor/ready-tests` logs `logger.warning` on corrupt `bundle_json` instead of silent swallow.
  - `/scan-task` caps reads at `MAX_IMAGE_SIZE_BYTES + 1`; `/scan-task-base64` enforces `mime_type.startswith("image/")`, strips `data:...;base64,` prefixes, and validates base64 + size limit.
  - Living OpenAPI (`openapi_specification.yaml`) / `api_specification.md` sync is explicitly deferred to Stage 3 per user instruction and tracked in `spec.md`, `checks.json`, and `docs/STATUS.md`.

### 2. C++20 / NDK Parity & SSOT — **PASS**
- **NDK / `TATAR_OCR_BUILD_PYTHON=OFF` isolation:** `native/include/tatar_ocr.hpp` and `native/src/tatar_ocr.cpp` contain zero `pybind11` includes or references; all Python bindings are isolated in `native/src/py_module.cpp` behind `if(TATAR_OCR_BUILD_PYTHON)` in `native/CMakeLists.txt`.
- **`pipeline_spec.json` & 1–12 cell detection:**
  - `pipeline_spec.json` adds `grid.short_h = [75, 1]` while preserving `grid.long_h = [180, 1]`, and widens `grid.questions.cell_count_range` from `[8, 12]` to `[1, 12]`.
  - `native/src/tatar_ocr.cpp` loads `short_h` in `PipelineSpec::load`, builds `h_lines_short`, and only falls back to `find_row_vertical_bounds_ex(h_lines_short, ..., short_x_max, ...)` when primary `long_h` finds no row bounds (`!h_found`). `snap_row_grid` / `match_grid_chain` returns 2 vertical line peaks for 1-cell rows (`best_chain.size() - 1 == 1`), so canonical 8–12 cell paths remain untouched.
- **PDF SSOT (`backend/app/generators/pdf_blank.py`):**
  - Reads `canvas`, `aruco`, `grid.student_name`, and `grid.questions` directly from `pipeline_spec.json` (`10 px/mm`): 15 name cells (`default_start_x=465` / `46.5 mm`, `default_pitch=88.0` / `8.8 mm`, `9.2 mm` height), `canonical_anchors` per `marker_id`, `n_q = len(rendered_questions)` where `rendered_questions = questions[:max_questions]`.
  - Encodes `"stu_id"` in QR payload when `student_id` is provided (both single `render_blank_pdf` and batch `render_batch_blanks_pdf` via `classes.py`).
  - Consistently uses `_FONT_NAME`, renders `title`, wraps multi-line prompts cleanly without 4th-line truncation, and places the 1-cell label (`<- 1 хәреф`) to the right of the box away from the left ArUco marker.

### 3. ML Benchmark & OCR Alphabet — **PASS**
- **39-letter OCR alphabet enforcement:**
  - `phonetics.py` loads `OCR_ALPHABET` from `pipeline_spec.json` (`АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯ`) and `TaskDraft.model_post_init` invokes `validate_ocr_answer` on every generated draft across all 24 generators.
  - `create_custom_task` (`constructor.py`) and `TaskExtractorService.extract_from_image` (`service.py`) both enforce `validate_ocr_answer`.
  - `FindErrorSentenceGenerator` (`syntax_tasks.py`) uses Cyrillic `А`–`Г` and shuffles sentences with `rng`. `SentenceMCClozeGenerator`, `TextComprehensionGenerator`, and `TextTitleMainIdeaGenerator` also shuffle MC options with `rng`.
- **Linguistic & generator fixes:**
  - `phonetics.py`: `is_front_vowel` handles `ё`, `ю`, and final-syllable loanwords (`счёт`→`True`, `актёр`→`True`, `ёлка`→`False`, `каен`→`False`); `inflect_comparative` handles voicing and consonant clusters (`ак`→`АГЫРАК`, `карт`→`КАРТЫРАК`); `get_possessive_suffix` handles elision stems (`борын`→`БОРНЫМ`, `сеңел`→`СЕҢЛЕМ`), `ипи`→`ИПИЕ`, and raises `PhoneticsError` when `number != "sg"`.
  - `lexicon.py`: `"төзәт"` replaces `"төзә"`, `"ҖИТЕЗ"` pairs with `"тиз"`, `"куян"` is in nasal `PLURAL_STEMS` (`КУЯННАР`), `ANTONYM_PAIRS` has no duplicate pairs, and `PossessiveCaseGenerator` filters `_VALID_POSSESSIVE_CASE_COMBOS` to `<= 12` cells.
  - `registry.py`: `assemble_variants` reaches all 24 generator types and deduplicates `(prompt_tt, expected_answer)` per variant.
- **12-sheet benchmark:** Independently re-run and verified at 0-diff parity (`Acc_char=96.88%`, `R_Tatar=97.33%`, `FEM=78.95%`, `FFM=96.49%`).

### 4. Backend QA & Workspace Hygiene — **PASS**
- **Test coverage:** `backend/tests/test_stage2_ocr_pdf_generators.py` plus `test_api.py` thoroughly cover `AC-1`–`AC-7`, including a real E2E `render_blank_image` → `cv2.QRCodeDetector` + `BlankOCRScanner._native` test across `[1, 4, 6, 8, 12]` question cells + 15 name cells, an exhaustive 24-generator × 20-seed sweep, and mocked `YandexGPTClient` / `TaskExtractorService` hardening tests.
- **Suite execution:** Independently re-run after all edits (`157 passed, 7 warnings in 17.18s`).
- **Workspace hygiene:** `native/build_core` is absent (`False`), and no `tmp_*`, `debug_*`, `check_*`, or `_peek_*` files exist in the repository.

---

## Findings

### Critical / Major
- None.

### Minor
1. **Unhandled `ValueError` outside `try` in `/scan-task` and `/scan-task-base64`** (`backend/app/routers/constructor.py` lines 399 and 449): `service = get_task_extractor_service()` is called outside the `try ... except ValueError` block. If `TASK_EXTRACTOR_MOCK_MODE=False` and `YANDEX_API_KEY` is unset, `get_task_extractor_service()` raises an unhandled `ValueError` (raw 500) rather than an explicit `HTTPException` (`503`/`500` with structured detail).
2. **Stage 2 spec/plan symbol & baseline wording drift** (`harness/changes/active/backend-ocr-pdf-generators-contract/{spec.md,plan.md,tasks.md}`):
   - `spec.md` (`AC-1`), `plan.md` (`M1`), and `tasks.md` (row 1) initially referenced `estimate_row_span_cells`, whereas `native/src/tatar_ocr.cpp` implements 1–12 cell detection via `find_row_vertical_bounds_ex` + `short_h` fallback + `snap_row_grid` (`match_grid_chain`) with no `estimate_row_span_cells` symbol.
   - `spec.md` (`AC-1`) states `FFM >= 98.25%`, whereas the verified `v4 INT8` `HEAD`-equivalent baseline in `v3/evaluate_blanks.py` (and `checks.json` / `docs/STATUS.md`) is `FFM (Lev<=1) = 96.49% (55/57)`.
3. **Inconsistent `ge=1` validation on `offline-bundle?variant=`** (`backend/app/routers/assignments.py` line 95): `get_offline_bundle` leaves `variant: Optional[int] = Query(None)` without `ge=1`, so `variant=0` returns `404` while `get_blank_pdf` returns `422` (`ge=1`). (Similarly, `search_task_bank` in `constructor.py` line 54 has `max_cells: Optional[int] = Query(None, le=12)` without `ge=1`.)

### Notes
1. **OpenAPI / `api_specification.md` sync deferred to Stage 3:** Stage 2 API contract deltas (`TaskBankItem.task_type`, `blank.pdf` `404`s, `/constructor/generate` `422`s, `find_error_sentence` `А`–`Г`, QR `stu_id`) must be included in the Stage 3 OpenAPI + `api_specification.md` sync alongside Stage 1 deltas.
2. **Android NDK submodule rebuild:** `kara_kuyan_frontend` should rebuild its Android `.so` after picking up the `native/` + `pipeline_spec.json` `short_h` changes so 1-cell blank scanning is enabled on Android.

---

## Recommendation

1. **Close/archive Stage 2 (`backend-ocr-pdf-generators-contract`)**.
2. Track Minor #1 (`get_task_extractor_service()` error handling in `constructor.py`), Minor #3 (`ge=1` on `offline-bundle?variant=` and `search_task_bank?max_cells=`), and full OpenAPI / `api_specification.md` alignment in **Stage 3**.

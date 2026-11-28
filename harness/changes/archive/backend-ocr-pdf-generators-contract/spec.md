# Spec — Stage 2: OCR Contract, PDF SSOT, Generators & Task Extractor

## Goal

Eliminate contract breaks between task generation, PDF blank rendering, C++20 native OCR (`pipeline_spec.json` + `native/src/tatar_ocr.cpp`), and textbook task extraction (`task_extractor`), while fixing Tatar morphology/lexicon bugs and silent parameter/error-handling defects.

## Non-goals

- Stage 3 scope: `X-Teacher-UUID` multi-tenant isolation & DB `teacher_uuid` columns, `/api/v1/model/manifest` update, `offline-bundle?variant=N` `total_variants` contract change, and full `openapi_specification.yaml` / `api_specification.md` sync.
- Stage 4 scope: N+1 query optimization in `assignments.py`/`classes.py` and Yandex HTTP client refactoring.
- Re-training `.onnx` model weights (alphabet remains the 39 uppercase Tatar Cyrillic letters in `pipeline_spec.json`).

## Assumptions

- `kara_kuyan_frontend` will rebuild its `native/` Android NDK submodule after `pipeline_spec.json` and `native/` changes are merged.
- `C:\Users\galee\PycharmProjects\tatar_ocr\.native_deps` contains OpenCV, ONNX Runtime, and `nlohmann_json` for building `tatar_ocr_native`.

## Spec Deltas (OpenSpec — against `docs/specs/README.md`)

### ADDED
- `pipeline_spec.json`: `grid.short_h = [75, 1]` for secondary horizontal line detection on 1-cell (`100 px`) question rows without altering `grid.long_h = [180, 1]`.

### MODIFIED
- `pipeline_spec.json`: `grid.questions.cell_count_range` changed from `[8, 12]` to `[1, 12]` to support 1–12 cell blanks in C++20 `native/`.
- `api_specification.md` / `openapi_specification.yaml` (to be synced in Stage 3):
  - `FindErrorSentenceGenerator` and `create_custom_task` enforce the 39-letter Cyrillic OCR alphabet (`А`–`Я` + `Ә, Ө, Ү, Җ, Ң, Һ`, no digits `0-9`, hyphens, or spaces); `find_error_sentence` uses Cyrillic letters `А`–`Г`.
  - `GET /assignments/{assignment_id}/blank.pdf` returns `404` on unknown `variant` (`ge=1`) or unknown `student_id`.
  - Batch/personalized QR codes include `"stu_id"` when `student_id` is supplied; PDF blanks strictly follow `pipeline_spec.json` (15 name cells at `88px` pitch, `canonical_anchors`, `title` rendered, full wrapped prompt).
  - `TaskBankItem` includes `task_type`.
  - `POST /constructor/generate` rejects unsupported `case_code` / `custom_stems` with `422`.

### REMOVED
- `POST /constructor/tasks`: digit-only answer bypass (`raw_ans.isdigit()`).
- `get_task_extractor_service`: silent fallback to `MockOCRClient`/`MockLLMTaskClassifier` when `YANDEX_API_KEY` is missing and mock mode is disabled.

## Acceptance Criteria

- `AC-1`: `pipeline_spec.json`, `native/include/tatar_ocr.hpp`, and `native/src/tatar_ocr.cpp` support `1..12` question cells via `short_h` + `estimate_row_span_cells`, compile cleanly for both `TATAR_OCR_BUILD_PYTHON=ON` and `OFF`, and show zero regression on the 12-sheet canonical benchmark (`Acc_char >= 96.88%`, `R_Tatar >= 97.33%`, `FFM >= 98.25%`).
- `AC-2`: `pdf_blank.py` reads geometry SSOT from `pipeline_spec.json` (15 name cells `8.8×9.2 mm`, `canonical_anchors`, `n_q = len(questions[:8])`, `_FONT_NAME`, renders `title`, encodes `"stu_id"` in QR, renders full prompt without 4th-line truncation, moves 1-cell label away from ArUco); `assignments.py` returns `404` for invalid `variant` / `student_id` and guards `json.loads` / `created_at` / `Content-Disposition`.
- `AC-3`: All 24 generators and `create_custom_task` enforce the 39-letter OCR alphabet (`validate_ocr_answer`), `FindErrorSentenceGenerator` uses `А`–`Г`, MC/reading/error generators shuffle options with `rng`, `PossessiveCaseGenerator` never exceeds 12 cells on built-in `NOUN_STEMS`, `CaseInflectionGenerator` & `PossessiveCaseGenerator` reject invalid `case_code`, and `registry.assemble_variants` reaches all 24 generator types and deduplicates within each variant.
- `AC-4`: `phonetics.py` and `lexicon.py` fix `is_front_vowel` (`ё`, `ю`, final-syllable loanwords), `inflect_comparative` (`ак->АГЫРАК`, `карт->КАРТЫРАК`), `get_possessive_suffix` (`борын->БОРНЫМ`, `сеңел->СЕҢЛЕМ`, `ипи->ИПИЕ`, `number!="sg"` error), `"төзәт"`, `"ҖИТЕЗ"`, `"куян"` nasal placement, duplicate antonym pair, and `topic_name_tt` consistency across all 24 generators.
- `AC-5`: `constructor.py` & `schemas/task.py` populate `TaskBankItem.task_type`, respect `ConstructorTaskRef.order`, `seed`, and distinct Variants 2–4 in custom `task_items`, validate `case_code`/`custom_stems` in `/generate`, log corrupt `bundle_json` in `/ready-tests`, cap upload read at `10 MB + 1`, and validate `mime_type` + strip data-URI prefix in `/scan-task-base64`.
- `AC-6`: `task_extractor` (`service.py`, `yandex_gpt.py`) raises `ValueError` instead of silently mocking when `YANDEX_API_KEY` is missing and mock mode is off, preserves `grade_hint` priority over LLM guesses, respects explicit `is_supported: false`, handles `confidence: null` and non-int fields safely, caps `prompt_tt` at 512 chars, logs partial `validation_errors`, and passes `task_type` in responses.
- `AC-7`: Full `backend/tests` suite (including exhaustive 24-generator dictionary test and real E2E `render_blank_image` → `tatar_ocr_native` verification) passes after the final edit.

## Open Questions

- None.

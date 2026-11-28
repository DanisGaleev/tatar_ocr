# Spec — Russian Primary School Task Generation, Addition Modes & OCR Alphabet Mask

## Goal

Extend **«Кара куян»** (`tatar_ocr`) with a parallel **Russian language (grades 1–4, primary school)** subsystem that coexists cleanly alongside the existing Tatar language (`tt`, grades 5–9) subsystem:
1. **Deterministic Russian primary-school generators**: 14 unambiguous, error-free task types grouped into 4 pedagogical categories (`orthography`, `morphology`, `lexicon`, `phonetics`), producing 1-to-12 cell A4-compatible answers in Russian Cyrillic (`А-Я`, `Ё`).
2. **All 4 task addition/selection modes** for Russian (`language="ru"`):
   - Deterministic generation (`POST /api/v1/constructor/generate`)
   - Manual task creation (`POST /api/v1/constructor/tasks`)
   - Photo/image exercise extraction (`POST /api/v1/constructor/scan-task`, `POST /api/v1/constructor/scan-task-base64`)
   - Selecting ready tasks & complete tests from the system (`GET /api/v1/constructor/tasks`, `GET /api/v1/constructor/ready-tests`, `POST /api/v1/constructor/tests`)
3. **Parallel language isolation**: Tatar (`language="tt"`) and Russian (`language="ru"`) never interfere in task bank search, generation, test assembly, PDF rendering, answer checking, or seed data. Existing Tatar contracts remain 100% backward compatible.
4. **OCR Russian Alphabet Mask**: Add a 33-letter Russian alphabet mask (`russian_33`, excluding Tatar-specific `Ә, Ө, Ү, Җ, Ң, Һ`) into `pipeline_spec.json`, `tatar_ocr_core` (C++20 `NativeBlankScanner`), `pybind11` bindings, `blank_pipeline.py`, and `/api/v1/model/manifest`.

## Analysis of Existing 4 Tatar Generation / Addition Modes

1. **Deterministic Linguistic Generation** (`POST /api/v1/constructor/generate`, `registry.py`, `phonetics.py`): 24 Tatar generators with isolated `random.Random(seed)`, `TaskDraft` (1..12 cells), `generate`, `generate_batch`, `create_custom`, and optional `save_to_bank`.
2. **Manual Custom Task Creation** (`POST /api/v1/constructor/tasks`): Validates `expected_answer` (digits or Tatar Cyrillic via `clean_word`), enforces `len <= cell_count <= 12`, persists with `task_type="custom"`, visible under `scope=my_tasks`.
3. **Photo / Image Extraction** (`POST /api/v1/constructor/scan-task`, `scan-task-base64`, `TaskExtractorService`): Yandex OCR (`["tt", "ru", "en"]`) + YandexGPT (with `MockOCRClient`/`MockLLMTaskClassifier` fallback), splits multi-item exercises, validates single-word `<= 12` char answers, optionally saves to `task_bank`.
4. **System Bank & Ready-Made Tests** (`GET /api/v1/constructor/tasks`, `GET /api/v1/constructor/ready-tests`, `POST /api/v1/constructor/tests`): Searches `task_bank`, lists `assembled_tests`, assembles A4 modular tests (up to 8 questions, 1..4 variants, offline bundle + printable PDF blanks).

## Non-goals

- Retraining or quantizing a new ONNX model specifically for Russian (the existing 39-class `tatar_ocr_v4_handwritten39_int8.onnx` already contains all 33 Russian uppercase letters and is masked at inference time).
- Altering Zero-Photo student submission storage rules (152-ФЗ remains strictly enforced).
- Breaking existing Tatar default behavior when `language` is omitted (`language` defaults to `"tt"`).

## Assumptions

- `pipeline_spec.json` class order is 39 characters: `"АӘБВГДЕЁЖҖЗИЙКЛМНҢОӨПРСТУҮФХҺЦЧШЩЪЫЬЭЮЯ"`, containing all 33 Russian uppercase letters plus 6 Tatar-specific letters (`Ә, Җ, Ң, Ө, Ү, Һ`).
- Russian primary-school curriculum covers grades `1..4`, while Tatar covers grades `5..9`, so backend grade validation widens to `1..9`.

## Spec Deltas (OpenSpec — against `docs/specs/README.md`)

### ADDED
- `pipeline_spec.json`: `model.alphabets` mapping `"tt"` (39 chars) and `"ru"` (33 chars).
- `native/include/tatar_ocr.hpp`, `native/src/tatar_ocr.cpp`, `native/src/py_module.cpp`, `blank_pipeline.py`: `language` parameter (`"tt"` | `"ru"`) on `classify_cell` and `process_blank`, plus `build_language_mask` / `alphabet_for_language` / `apply_language_logit_mask` zeroing the 6 Tatar-specific logits and re-normalizing softmax over the 33 Russian letters.
- `backend/app/generators/russian/`: 14 deterministic primary-school Russian generators across 4 categories (`orthography`, `morphology`, `lexicon`, `phonetics`) + `clean_russian_word`, `mask_orthogram_combination`, `analyze_russian_phonetics`.
- `TaskBankModel.language`, `AssembledTestModel.language` (`"tt"` / `"ru"`), idempotent SQLite column migration in `init_db()`.
- `language` query/body/response fields across `/constructor/task-types`, `/constructor/tasks`, `/constructor/generate`, `/constructor/tests`, `/constructor/ready-tests`, `/constructor/verify-answer`, `/constructor/scan-task`, `/constructor/scan-task-base64`, `/assignments`, `/assignments/{id}/offline-bundle`, and `alphabet_mask` (`"tatar_39"` / `"russian_33"`).
- `/api/v1/model/manifest`: `language_alphabets` and `language_masks` maps for `"tt"` and `"ru"`.
- Initial Russian seed tasks, class `cls_3a_2026`, and ready-made test `RUS-2026-Q1` in `seed_initial_data()`.

### MODIFIED
- `grade_level` validation widened from `5..9` to `1..9` in `TaskDraft`, `TaskBankItem`, `CreateTaskRequest`, `GenerateTasksRequest` (optional `grade_level`, defaults to `3` for `"ru"` and `7` for `"tt"`), `ScanTaskRequestBase64`, `AssembleTestRequest`, and `/constructor/tasks` / `/constructor/scan-task`.
- `CreateTaskRequest`: accepts optional `prompt` / `topic_name` aliases alongside `prompt_tt` / `topic_name_tt` for Russian/neutral client payloads.
- `checker.py`: `RUSSIAN_HOMOGLYPH_MAP` (`Y` -> `У` for `"ru"`, `Y` -> `Ү` for `"tt"`) and `language` parameter on `normalize_char`, `verify_question_cells`, `verify_test_submission`.
- `pdf_blank.py`: Russian labels/footer when `language="ru"` (`"Ответ:"`, `"Русский алфавит: А–Я, Ё..."`, `"lang": "ru"` in QR).
- `task_extractor`: `language="ru"` uses `languageCodes: ["ru"]` in Yandex OCR, Russian primary-school system prompt in YandexGPT, Russian mock patterns in `MockOCRClient`/`MockLLMTaskClassifier`, and Russian word validation in `TaskExtractorService`.

### REMOVED
- None.

## Acceptance Criteria

- `AC-1`: 14 Russian primary-school generators (across 4 categories) generate deterministic, 100% Russian Cyrillic (`<= 12` chars), zero-homophone tasks with valid `expected_cells` and `create_custom` support.
- `AC-2`: All 4 Russian task addition modes (deterministic `/generate`, manual `/tasks`, photo `/scan-task` & `/scan-task-base64`, ready tasks/tests `/tasks`, `/ready-tests`, `/tests`) work end-to-end and persist with `language="ru"`.
- `AC-3`: Parallel Tatar (`tt`) and Russian (`ru`) isolation: `tt` and `ru` tasks/generators/tests never leak into each other; mixed-language test assembly returns HTTP 422; cross-language generator calls return HTTP 400; all existing Tatar behavior/tests remain unbroken.
- `AC-4`: Russian answer checking (`checker.py`) and Russian PDF blank rendering (`pdf_blank.py`) work cleanly for `language="ru"` without Tatar-only characters or labels.
- `AC-5`: `pipeline_spec.json`, C++20 `tatar_ocr_core`, `pybind11` bindings, `blank_pipeline.py`, and `/api/v1/model/manifest` enforce the 33-letter Russian mask (`language="ru"` never predicts `Ә, Ө, Ү, Җ, Ң, Һ` in `ch`, `top3`, or `top_items`), while `language="tt"` remains unchanged.
- `AC-6`: C++20 `native/` builds cleanly with both `TATAR_OCR_BUILD_PYTHON=ON` and `TATAR_OCR_BUILD_PYTHON=OFF` (Android NDK parity).
- `AC-7`: Full backend pytest suite passes after final edit, and OCR Tatar benchmark (if available) shows zero regression.

## Open Questions

- None (all defaults preserve full backward compatibility with `kara_kuyan_frontend`).

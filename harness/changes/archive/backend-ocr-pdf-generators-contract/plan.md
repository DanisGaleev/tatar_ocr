# Plan (ExecPlan) — Stage 2: OCR Contract, PDF SSOT, Generators & Task Extractor

## Approach

Implement 1–12 cell detection cleanly in `pipeline_spec.json` + `native/src/tatar_ocr.cpp` using a secondary `short_h` horizontal kernel (only when primary `long_h` finds no row bounds) + span estimation, keeping primary `long_h` path identical on canonical 8–12 cell blanks. Align `pdf_blank.py` to `pipeline_spec.json` SSOT, enforce the 39-letter Cyrillic OCR alphabet across all generators/constructor/extractor, fix Tatar phonetics/lexicon defects, and verify end-to-end with native C++ builds, the 12-sheet benchmark, and `pytest`.

## Milestones

1. **M1 — Native C++20 & `pipeline_spec.json` (`AC-1`):** Add `short_h`, `cell_count_range: [1, 12]`, `find_row_vertical_bounds_ex`, `estimate_row_span_cells`; build `tatar_ocr_native` + `TATAR_OCR_BUILD_PYTHON=OFF`; run `v3/evaluate_blanks.py`.
2. **M2 — PDF SSOT & `assignments.py` (`AC-2`):** Align `pdf_blank.py` to `pipeline_spec.json` + QR `stu_id` + `title` + full prompt wrapping + `render_blank_image`; harden `assignments.py` 404s.
3. **M3 — Alphabet, Phonetics, Lexicon & Generators (`AC-3`, `AC-4`):** Fix `phonetics.py`, `base.py`, `lexicon.py`, `morphology.py`, `case_inflection.py`, `syntax_tasks.py`, `reading_tasks.py`, `word_formation.py`, `phonetics_tasks.py`, `registry.py`.
4. **M4 — Constructor & Task Extractor (`AC-5`, `AC-6`):** Fix `schemas/task.py`, `routers/constructor.py`, `task_extractor/{service.py,yandex_gpt.py}`.
5. **M5 — Tests, Benchmark & Fresh-Context Review (`AC-7`):** Add exhaustive + E2E native tests, verify benchmark + `pytest backend\tests -q`, run Fresh-Context Evaluator across all 4 domain lenses.

## File Ownership

| File / Area | Writer | Notes |
|---|---|---|
| `pipeline_spec.json`, `native/include/tatar_ocr.hpp`, `native/src/tatar_ocr.cpp` | coordinator | 1–12 cell detection |
| `backend/app/generators/pdf_blank.py`, `backend/app/routers/assignments.py` | coordinator | PDF SSOT & 404s |
| `backend/app/generators/{base,phonetics,lexicon,registry,case_inflection,morphology,word_formation,lexicon_tasks,phonetics_tasks,syntax_tasks,reading_tasks}.py` | coordinator | Alphabet, phonetics, 24 generators |
| `backend/app/schemas/task.py`, `backend/app/routers/constructor.py`, `backend/app/services/task_extractor/{service,yandex_gpt}.py` | coordinator | Constructor & extractor |
| `backend/tests/{test_api,test_stage2_ocr_pdf_generators}.py` | coordinator | Regression & E2E tests |
| `harness/changes/active/backend-ocr-pdf-generators-contract/reviews/review.md` | evaluator | Independent 4-lens review |

## Risks & Mitigations

- Changing `long_h` directly would break canonical 8–12 cell detection → keep `long_h = [180, 1]` primary and only fall back to `short_h = [75, 1]` when `long_h` finds no row bounds.
- Fixing `is_front_vowel` / `inflect_comparative` could break existing unit tests → verify against `test_phonetics.py`, `test_ground_truth_lexicon.py`, and `test_new_generators.py`.

## Surprises & Discoveries

*(Update during execution whenever codebase/tests/build reveal unexpected facts)*

## Decision Log

- `2026-11-28`: Stage 1 (`backend-critical-correctness`) archived with OpenAPI sync deferred to Stage 3 per user instruction.
- `2026-11-28`: Support 1–12 cell blanks in `pipeline_spec.json` + `native/` while enforcing the 39-letter Cyrillic OCR alphabet (`А`–`Г` for `find_error_sentence`).

# Fresh-Context Evaluation Report — `russian-primary-tasks-and-ocr-mask`

## Verdict: **accept**

All acceptance criteria and living specification deltas have been verified. An initial evaluation pass identified two defects (`analyze_russian_phonetics` missing soft `[й']` for strong-position iotated vowels, and incomplete `openapi_specification.yaml` path/schema sync); both were fixed, `#include <cstdint>` was added to `native/include/tatar_ocr.hpp`, all tests/builds were re-executed after the final edit, and the re-evaluation confirmed `accept`.

## Verification Summary by Lens

1. **API Contract & Zero-Photo (152-ФЗ):**
   - `language` defaults to `"tt"` across models, schemas, routers, and SQLite migration (`backend/app/core/database.py`), preserving 100% backward compatibility with existing Tatar workflows.
   - `openapi_specification.yaml` and `api_specification.md` are synced with all added/modified schemas and paths (`/api/v1/assignments`, `/api/v1/constructor/task-types`, `/generate`, `/verify-answer`, `/scan-task`, `/scan-task-base64`, `/ready-tests`, `/tests`, `/api/v1/model/manifest`).
   - Zero-Photo policy (152-ФЗ) preserved: no student photos persisted or logged.

2. **C++20 / Python / Android NDK Parity:**
   - Zero `pybind11` or Python headers in `native/include/tatar_ocr.hpp` and `native/src/tatar_ocr.cpp`; `#include <cstdint>` added.
   - `pipeline_spec.json` (`model.alphabets.ru`), `NativeBlankScanner::build_language_mask("ru")`, `run_batch`, `build_prediction`, and `blank_pipeline.py` (`apply_language_logit_mask`) zero out the 6 Tatar-only classes (`Ә, Ө, Ү, Җ, Ң, Һ`) before softmax normalization over the 33 Russian classes.
   - C++20 builds passed with both `TATAR_OCR_BUILD_PYTHON=ON` and `TATAR_OCR_BUILD_PYTHON=OFF`.

3. **Russian Primary Generators & Lexicon Quality:**
   - All 14 generators across 4 categories (`orthography`, `morphology`, `lexicon`, `phonetics`) verified deterministic across seeds, strictly Russian Cyrillic (`^[А-ЯЁ]+$`), `1 <= len(expected_answer) <= cell_count <= 12`, zero homophone ambiguity, and `bad all_hard across 200 seeds: []`.

4. **Backend QA & Hygiene:**
   - Full `.venv` pytest suite after final edit: **169 passed** (16.16s).
   - Pure-Python subset verified directly by evaluator: **76 passed** (0.44s).
   - 12-sheet Tatar OCR benchmark (`v3/evaluate_blanks.py --model models/tatar_ocr_v4_handwritten39_int8.onnx`): `Acc_char = 96.88% (435/449)`, `R_Tatar = 97.33% (73/75)`, `FEM = 78.95% (45/57)`.
   - Zero temporary `tmp_*`/`debug_*` files; `native/build_nopy` removed.

## Non-Blocking Notes (for future polish)

- `AssembleTestRequest.grade_level` defaults to `7` when omitted even if `language="ru"` (callers should pass `1..4` explicitly for Russian tests).
- `backend/app/routers/analytics.py` and `reports.py` retain Tatar fallback strings (`"Укучы ..."`, `"Билгесез"`, `"Сорау №..."`, `"Сыйныф: ..."`).
- `NativeBlankScanner::process_blank(img_bgr, "ru")` applies `allowed_mask` to both question cells and student name header cells.

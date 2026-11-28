# Tasks — Stage 2: OCR Contract, PDF SSOT, Generators & Task Extractor

| # | Task | Files owned | Check IDs | Status | Verification evidence |
|---|---|---|---|---|---|
| 1 | Add `short_h` & `1..12` cell detection in `pipeline_spec.json` + C++20 `native/`, build both ON/OFF targets, run 12-sheet benchmark | `pipeline_spec.json`, `native/include/tatar_ocr.hpp`, `native/src/tatar_ocr.cpp` | `AC-1`, `GATE-NATIVE`, `GATE-BENCHMARK` | done | `TATAR_OCR_BUILD_PYTHON=ON/OFF` exit 0; `evaluate_blanks.py`: `Acc_char=96.88% (435/449)`, `R_Tatar=97.33% (73/75)`, `FEM=78.95%`, `FFM=96.49%` (0-diff vs HEAD-equiv spec) |
| 2 | Align `pdf_blank.py` to `pipeline_spec.json` SSOT, add `stu_id` QR, `render_blank_image`, and fix `assignments.py` 404s | `backend/app/generators/pdf_blank.py`, `backend/app/routers/assignments.py` | `AC-2` | done | `test_e2e_render_blank_image_to_native_scanner_1_to_12_cells_and_qr` & `test_assignments_blank_pdf_validation_and_404_guards` pass |
| 3 | Fix 39-letter alphabet validation, `phonetics.py`, `lexicon.py`, and all 24 generators + `registry.py` | `backend/app/generators/*.py` | `AC-3`, `AC-4` | done | 24-generator x 20-seed sweep + phonetics/lexicon assertions + `test_phonetics.py`/`test_new_generators.py` pass |
| 4 | Fix `schemas/task.py`, `routers/constructor.py`, and `services/task_extractor/{service,yandex_gpt}.py` | `backend/app/schemas/task.py`, `backend/app/routers/constructor.py`, `backend/app/services/task_extractor/*` | `AC-5`, `AC-6` | done | `test_constructor_generate_validation_and_assemble_order_seed`, `test_scan_task_base64_mime_and_data_uri_prefix`, `test_task_extractor_config_and_parsing_hardening` pass |
| 5 | Update `test_api.py`, add `test_stage2_ocr_pdf_generators.py` (exhaustive + real E2E native OCR test), run `pytest` | `backend/tests/{test_api,test_stage2_ocr_pdf_generators}.py` | `AC-7`, `GATE-BACKEND` | done | `pytest backend\tests -q`: `157 passed, 7 warnings in 17.16s` |
| 6 | Run Fresh-Context Evaluator (all 4 lenses) and update `checks.json`, `review.md`, `STATUS.md` | `harness/changes/active/backend-ocr-pdf-generators-contract/*`, `docs/STATUS.md` | `GATE-SPECS` | done | `reviews/review.md` (`accept-with-notes`, 0 Critical/Major blockers) |

## Session Handoff

- **Last completed slice:** Stage 2 implementation, C++20 ON/OFF builds, 12-sheet benchmark, and 157-test pytest suite passing.
- **Next immediate step:** Complete Fresh-Context Evaluator pass (`reviews/review.md`) and confirm Stage 2 archival before Stage 3.
- **Uncommitted / tricky state:** Rebuild Android NDK `.so` in sister repo `kara_kuyan_frontend` when syncing `native/` submodule so 1-cell `short_h` support is active on Android.

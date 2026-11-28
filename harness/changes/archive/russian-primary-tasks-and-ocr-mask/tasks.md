# Tasks — Russian Primary School Tasks & OCR Alphabet Mask

| # | Task | Files owned | Check IDs | Status | Verification evidence |
|---|---|---|---|---|---|
| 1 | Verify Python venv & native deps in worktree | `plan.md` | `GATE-BACKEND` | done | `139 passed` via main `.venv`; `.native_deps` located |
| 2 | Implement Russian phonetics, lexicon & 14 primary-school generators | `backend/app/generators/russian/*` | `AC-1` | done | `test_russian_generators.py` passed |
| 3 | Integrate `language` (`tt`/`ru`) in registry, checker, DB, schemas, routers, PDF, seeds | `backend/app/**` | `AC-2`, `AC-3`, `AC-4` | done | `test_russian_api_and_isolation.py` + all Tatar tests passed (`169 passed`) |
| 4 | Extend `task_extractor` for Russian photo scanning | `backend/app/services/task_extractor/*` | `AC-2`, `AC-3` | done | `test_task_extractor.py` + `test_russian_api_and_isolation.py` passed |
| 5 | Implement Russian 33-letter OCR mask in spec, C++20 native core & `blank_pipeline.py` | `pipeline_spec.json`, `native/*`, `blank_pipeline.py` | `AC-5`, `AC-6` | done | `tatar_ocr_core` (`ON` & `OFF`) built cleanly; `test_ocr_russian_mask.py` (`4 passed`) |
| 6 | Write & run Russian generator, API/isolation, and OCR mask tests + native builds | `backend/tests/*`, `native/build*` | `AC-1`..`AC-7`, `GATE-*` | done | `169 passed`; benchmark `Acc_char=96.88%`, `R_Tatar=97.33%` |
| 7 | Sync living specs & run Fresh-Context Evaluator review | `api_specification.md`, `openapi_specification.yaml`, `docs/*`, `reviews/review.md` | `GATE-SPECS` | done | Verdict `accept` recorded in `reviews/review.md` |

## Session Handoff

- **Last completed slice:** Fresh-Context Evaluator returned `accept`; all 12 checks in `checks.json` are `PASS`.
- **Next immediate step:** Await user confirmation to move `harness/changes/active/russian-primary-tasks-and-ocr-mask/` to `harness/changes/archive/russian-primary-tasks-and-ocr-mask/`.
- **Uncommitted / tricky state:** `native/build/` holds the compiled `.pyd` for `blank_pipeline.py` tests; `native/build_nopy/` removed after verification.

## Completion Definition

- [x] Every check in `checks.json` has `"status": "PASS"` with recorded `command` + `evidence`.
- [x] No open question in `spec.md` was silently guessed.
- [x] Spec Deltas from `spec.md` synced into living docs (`docs/specs/README.md`).
- [x] Independent Fresh-Context Review in `reviews/review.md` recorded with verdict `accept`.

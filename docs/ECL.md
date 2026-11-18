# ECL — Engineering Change Lifecycle

`tatar_ocr` uses a unified long-running agent harness combining **ECL** (one active change, file ownership), **OpenSpec** (delta specs synced to living specs), **OpenAI ExecPlans** (milestone & discovery tracking in `plan.md`), and **Anthropic Long-Running Harness** (Default-FAIL `checks.json`, Session Handoff, Fresh-Context Evaluator).

---

## 1. Small Change vs Structured Change

- **Small Change:** a localized single-file fix (typo, doc wording, isolated bug fix inside one function/file) with **no** behavioral change to API contracts, DB schema, `pipeline_spec.json`, C++20 ABI, or ML weights. May proceed directly without creating a `harness/changes/active/` folder, provided relevant isolated tests pass and are reported.
- **Structured Change:** anything touching multiple files, API contracts (`api_specification.md` / `openapi_specification.yaml`), DB models, `native/` C++20 code, `pipeline_spec.json`, ML models/training/eval, or new dependencies. **Must** have an active change folder under `harness/changes/active/<slug>/`.

## 2. One Active Change

- At most **one** directory lives in `harness/changes/active/` at a time.
- If urgent unrelated work arrives while a change is active, park the current change into `harness/changes/parking/<slug>/` (record `Blocker` in `summary.md` and update `docs/STATUS.md`) before starting the new one.

## 3. File Ownership

- One writer per file. `plan.md` and `tasks.md` declare file ownership for every task.
- Parallel subagents may perform read-only analysis or edit strictly disjoint files.

## 4. Lifecycle (6 Artifacts)

Copy all templates from `harness/templates/change/` into `harness/changes/active/<slug>/`:

1. **`summary.md`** — ID, status, owner, one-line goal, blocker (if parked).
2. **`spec.md` (OpenSpec-style)** — WHAT & WHY: Goal, Non-goals, Assumptions, Open questions, Acceptance criteria, and **`## Spec Deltas (ADDED / MODIFIED / REMOVED)`** referencing the living specs in `docs/specs/README.md`.
3. **`plan.md` (ExecPlan-style)** — HOW: chosen approach, rejected alternatives, Milestones, File ownership, Risks, plus living sections **`Surprises & Discoveries`** and **`Decision Log`** updated whenever execution reveals new facts.
4. **`tasks.md`** — atomic ordered slices with file ownership, status, and **Session Handoff** notes so a fresh context can resume immediately.
5. **`checks.json` (Default-FAIL contract)** — machine-readable checklist of every acceptance criterion and relevant quality gate. Every check starts `"status": "FAIL"` and transitions to `"PASS"` **only** when `command`, `evidence` (actual observed output/metrics), and `verified_at` are recorded after the last relevant edit.
6. **`reviews/review.md` (Fresh-Context Evaluator)** — independent review pass (by a separate subagent or a fresh context/session that did not author the implementation). The evaluator checks code diff against `spec.md` deltas, inspects `checks.json`, and re-runs verification commands independently.

## 5. Session Handoff & Context Reset Protocol

For multi-session or complex tasks, avoid cramming everything into one degraded context window:
- Work in atomic slices (1–2 tasks from `tasks.md`).
- After verifying a slice, update `checks.json`, `tasks.md`, `plan.md` (`Surprises & Discoveries`), and the `Session Handoff` section in `docs/STATUS.md`.
- A new session always boots by reading `AGENTS.md` → `docs/STATUS.md` → active change artifacts → `git log -n 5 --oneline`.

## 6. Closure & Spec Sync

A change is closed only after `reviews/review.md` has verdict `accept` (or `accept-with-notes` approved by the user):
1. Apply all `ADDED / MODIFIED / REMOVED` Spec Deltas from `spec.md` to the living specs (`docs/specs/README.md` targets: `api_specification.md`, `openapi_specification.yaml`, `docs/PIPELINE.md`, `pipeline_spec.json`, etc.).
2. Update `docs/STATUS.md` (Completed changes table, current metrics/test counts, next priorities).
3. Move `harness/changes/active/<slug>/` to `harness/changes/archive/<slug>/`.
4. Lead (user) confirms closure / commit. Agents do not silently skip living-spec sync.

## 7. Policy Evolution

- Changes to `docs/ECL.md` require explicit user approval.

# Tasks — <title>

Atomic, ordered slices. Each task names its writer and maps to `checks.json`.

| # | Task | Files owned | Check IDs | Status | Verification evidence |
|---|---|---|---|---|---|
| 1 | <task> | `<paths>` | `AC-1` | pending | `<command + observed output>` |

## Session Handoff

- **Last completed slice:** <none>
- **Next immediate step:** <exact task # and file to open next>
- **Uncommitted / tricky state:** <none>

## Completion Definition

- [ ] Every check in `checks.json` has `"status": "PASS"` with recorded `command` + `evidence`.
- [ ] No open question in `spec.md` was silently guessed.
- [ ] Spec Deltas from `spec.md` synced into living docs (`docs/specs/README.md`).
- [ ] Independent Fresh-Context Review in `reviews/review.md` recorded with verdict `accept`.

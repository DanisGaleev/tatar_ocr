# GEMINI.md

Read and follow [`AGENTS.md`](AGENTS.md) as the single source of truth for instructions, commands, boundaries, and context loading order.

For any feature request, bugfix, API/contract edit, C++20 `native/` update, or ML pipeline/model change, **proactively** load and follow [`.agents/skills/tatar-ocr-delivery/SKILL.md`](.agents/skills/tatar-ocr-delivery/SKILL.md) without waiting for the user to ask for ECL, specs, `checks.json`, or independent review:
- Automatically classify Small vs Structured Change.
- Automatically create/update `harness/changes/active/<slug>/` (`spec.md`, `plan.md`, `tasks.md`, `checks.json`) and ask the user for brief confirmation before heavy implementation.
- Automatically run an independent Fresh-Context Evaluator subagent pass for `reviews/review.md` before declaring a Structured Change done, then offer one-step closure/archival.

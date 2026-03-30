---
title: Failure, retry, and resume
diataxis: explanation
audience: users operating persisted experiments
goal: Explain the runtime behavior of failures, retries, and resume.
---

# Failure, retry, and resume

What it is: the user-facing behavior for partial failures, retryable persistence, and continuing interrupted work.

When it matters: whenever a provider call, parsing step, scoring step, or persistence action fails.

What you provide: runtime retry settings and a store that persists enough state to resume.

What Themis provides: failure events, partial-failure status handling, and per-stage resume behavior.

What to inspect when it goes wrong: stage-specific failures inside execution state, evaluation failures, and runtime retry settings.

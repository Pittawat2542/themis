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

Use this flow to reason about whether the next action is retrying a stage or continuing from stored state.

```mermaid
flowchart TD
    A["Stage executes"] --> B{"Stage failed?"}
    B -->|No| C["Advance to next stage"]
    B -->|Yes| D["Record failure event"]
    D --> E{"Retry allowed?"}
    E -->|Yes| F["Retry stage"]
    E -->|No| G["Persist partial state"]
    G --> H["Resume from stored progress later"]
```

Retry is a same-stage recovery decision, while resume is a later continuation decision over persisted state.

What to inspect when it goes wrong: stage-specific failures inside execution state, evaluation failures, and runtime retry settings.

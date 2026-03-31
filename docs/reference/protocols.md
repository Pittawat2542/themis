---
title: Protocols reference
diataxis: reference
audience: users and contributors implementing extensions
goal: Document the extension contracts exposed by Themis.
---

# Protocols reference

Use this page when you are implementing custom components rather than using builtin ids.

Important runtime instrumentation contracts:

- `LifecycleSubscriber`: stage callbacks plus `on_event(...)` hooks for observing generation, reduction, parsing, scoring, and judging
- `TracingProvider`: `start_span(...)` / `end_span(...)` hooks for run- and stage-level tracing

Important config/runtime contracts:

- `Generator`, `CandidateReducer`, `Parser`, and metric protocols define the extension surface used in both Python authoring and config-loaded experiments
- `WorkflowRunner` executes workflow-backed metrics; `LifecycleSubscriber` and `TracingProvider` observe execution without changing `run_id`

Generated contracts:

::: themis.core.protocols

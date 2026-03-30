---
title: Experiment lifecycle reference
diataxis: reference
audience: Python users authoring and executing experiments
goal: Document the primary experiment authoring and execution APIs.
---

# Experiment lifecycle reference

Primary entry points:

- `evaluate(...)`: Layer 1 convenience API
- `Experiment.from_config(...)`: load a config-backed experiment
- `Experiment.compile()`: build a `RunSnapshot`
- `Experiment.run()` / `run_async()`: execute a snapshot
- `Experiment.rejudge()` / `rejudge_async()`: rerun workflow-backed metrics from stored upstream artifacts

Lookup notes:

- `compile()` freezes identity and provenance into a `RunSnapshot`
- `run()` executes work; `compile()` does not
- `rejudge()` requires stored upstream artifacts and, for memory-backed runs, the original store instance

Use the generated API pages for full signatures and docstrings, and pair this page with [Compile vs run](../explanation/compile-vs-run.md) when behavior is conceptually unclear.

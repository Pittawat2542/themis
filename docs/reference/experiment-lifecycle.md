---
title: Experiment lifecycle reference
diataxis: reference
audience: Python users authoring and executing experiments
goal: Document the primary experiment authoring and execution APIs.
---

# Experiment lifecycle reference

Primary entry points:

- `evaluate(model=..., data=..., metric=..., ...)`: Layer 1 convenience API
- `Experiment.from_config(...)`: load a config-backed experiment
- `Experiment.compile()`: build a `RunSnapshot`
- `Experiment.run()` / `run_async()`: execute a snapshot
- `Experiment.replay()` / `replay_async()`: rerun downstream stages from stored upstream artifacts
- `Experiment.rejudge()` / `rejudge_async()`: shorthand for `replay(stage="judge")`

Lookup notes:

- `compile()` freezes identity and provenance into a `RunSnapshot`
- `run()` executes work; `compile()` does not
- `replay()` requires stored upstream artifacts and, for memory-backed runs, the original store instance
- `rejudge()` is a workflow-metric specialization of `replay(stage="judge")`

Use the generated API pages for full signatures and docstrings, and pair this page with [Compile vs run](../explanation/compile-vs-run.md) when behavior is conceptually unclear.

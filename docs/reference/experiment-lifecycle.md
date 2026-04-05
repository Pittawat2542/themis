---
title: Experiment lifecycle reference
diataxis: reference
audience: Python users authoring and executing experiments
goal: Document the primary experiment authoring and execution APIs.
---

# Experiment lifecycle reference

## Primary entry points

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `evaluate(model=..., data=..., metric=..., ...)` | Layer 1 convenience API | You want the smallest Python entry point for a straightforward run | Best for short scripts and notebooks |
| `Experiment.from_config(...)` | Config loader | You want config-backed experiments instead of inline Python-only authoring | Resolves config and components before execution |
| `Experiment.compile()` | Snapshot builder | You want a `RunSnapshot` before deciding whether to execute | Freezes identity and provenance but does not run work |
| `Experiment.run()` / `run_async()` | Executor | You want to execute a compiled experiment | Writes stage artifacts and results to the configured store |
| `Experiment.replay()` / `replay_async()` | Downstream rerun API | You want to rerun downstream stages from stored upstream artifacts | Requires stored upstream artifacts |
| `Experiment.rejudge()` / `rejudge_async()` | Judge-stage shortcut | You want to rerun only workflow-backed judging | Equivalent to `replay(stage="judge")` |

## Lookup notes

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `compile()` | Identity step | You want to inspect what the logical run will be before execution | Produces a `RunSnapshot` and `run_id` |
| `run()` | Execution step | You are ready to execute planned work | `compile()` alone does not perform generation or scoring |
| `replay()` | Artifact reuse step | You want new downstream outputs from fixed upstream artifacts | Memory-backed runs require access to the original in-process store |
| `rejudge()` | Workflow-specialized replay | Only judge-stage artifacts need to change | A convenience wrapper around `replay(stage="judge")` |

Use the generated API pages for full signatures and docstrings, and pair this page with [Compile vs run](../explanation/compile-vs-run.md) when behavior is conceptually unclear.

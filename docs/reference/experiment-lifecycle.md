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
| `evaluate(model=..., data=..., metric=..., ...)` | Layer 1 convenience API | You want the smallest synchronous Python entry point for a straightforward run | Best for short scripts; use only outside a running event loop |
| `evaluate_async(model=..., data=..., metric=..., ...)` | Layer 1 convenience API | You want the smallest async Python entry point for a straightforward run | Preferred in notebooks and async applications |
| `Experiment.from_config(...)` | Config loader | You want config-backed experiments instead of inline Python-only authoring | Resolves config and components before execution |
| `Experiment.compile()` | Snapshot builder | You want a `RunSnapshot` before deciding whether to execute | Freezes identity and provenance but does not run work |
| `Experiment.run()` / `run_async()` | Executor | You want to execute a compiled experiment | Use the async form when an event loop is already running |
| `Experiment.replay()` / `replay_async()` | Downstream rerun API | You want to rerun downstream stages from stored upstream artifacts | Requires stored upstream artifacts; use the async form inside async environments |
| `Experiment.rejudge()` / `rejudge_async()` | Judge-stage shortcut | You want to rerun only workflow-backed judging | Equivalent to `replay(stage="judge")`; use the async form inside async environments |

## Lookup notes

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `compile()` | Identity step | You want to inspect what the logical run will be before execution | Produces a `RunSnapshot` and `run_id` |
| `run()` | Execution step | You are ready to execute planned work | `compile()` alone does not perform generation or scoring; sync wrappers reject active event loops |
| `replay()` | Artifact reuse step | You want new downstream outputs from fixed upstream artifacts | Memory-backed runs require access to the original in-process store |
| `rejudge()` | Workflow-specialized replay | Only judge-stage artifacts need to change | A convenience wrapper around `replay(stage="judge")` |

Use the generated API pages for full signatures and docstrings, and pair this page with [Compile vs run](../explanation/compile-vs-run.md) when behavior is conceptually unclear.

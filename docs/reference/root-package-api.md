---
title: Root package API
diataxis: reference
audience: Python users of the public package surface
goal: Enumerate the main root-package exports and what each category is for.
---

# Root package API

## Primary public exports

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| Package metadata | Category | You need package version information | Includes `__version__` |
| Experiment authoring | Category | You want the main Python surfaces for authoring and executing runs | Includes `Experiment`, `evaluate` |
| Prompt authoring | Category | Prompt material should be part of experiment identity | Includes `PromptSpec` |
| Persistence | Category | You need concrete stores or store abstractions | Includes `InMemoryRunStore`, `SqliteRunStore`, `RunStore`, `sqlite_store` |
| Execution and results | Category | You need compiled snapshots, status, results, or runtime tuning models | Includes `RunSnapshot`, `RunResult`, `RunEstimate`, `RunStatus`, `RuntimeConfig` |
| Inspection and reporting | Category | You want reporting, inspection, or summary helpers | Includes `Reporter`, `StatsEngine`, `get_run_snapshot`, `get_execution_state`, `get_evaluation_execution`, `quickcheck`, `snapshot_report` |
| Bundle workflows | Category | You want to export or import stage artifacts | Includes generation, reduction, parse, score, and evaluation bundle helpers |

Use [Experiment lifecycle](experiment-lifecycle.md) for the core execution flow and [Stores and inspection](stores-and-inspection.md) for reporting and persistence helpers.

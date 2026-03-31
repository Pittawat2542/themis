---
title: Root package API
diataxis: reference
audience: Python users of the public package surface
goal: Enumerate the main root-package exports and what each category is for.
---

# Root package API

Primary public exports from `themis`:

- experiment authoring: `Experiment`, `evaluate`
- persistence: `InMemoryRunStore`, `SqliteRunStore`, `RunStore`, `sqlite_store`
- execution and results: `RunSnapshot`, `RunResult`, `RunEstimate`, `RunStatus`, `RuntimeConfig`
- inspection and reporting: `Reporter`, `StatsEngine`, `get_run_snapshot`, `get_execution_state`, `get_evaluation_execution`, `quickcheck`, `snapshot_report`
- bundle workflows: `export_generation_bundle`, `import_generation_bundle`, `export_evaluation_bundle`, `import_evaluation_bundle`

Use [Experiment lifecycle](experiment-lifecycle.md) for the core execution flow and [Stores and inspection](stores-and-inspection.md) for reporting and persistence helpers.

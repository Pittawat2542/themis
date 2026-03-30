---
title: Stores and inspection reference
diataxis: reference
audience: users inspecting persisted runs
goal: Document persistence helpers, reporting/export APIs, and inspection helpers.
---

# Stores and inspection reference

Store-related symbols:

- `InMemoryRunStore`
- `SqliteRunStore`
- `RunStore`
- `sqlite_store`

Inspection and export helpers:

- `get_execution_state(store, run_id)`
- `get_evaluation_execution(store, run_id, case_id, metric_id)`
- `Reporter`
- `snapshot_report`
- `quickcheck`
- generation/evaluation bundle export and import helpers

Use persistent stores whenever the workflow needs resume, reporting, comparison, export, or later inspection from another process.

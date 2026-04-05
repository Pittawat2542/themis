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

- `get_run_snapshot(store, run_id)`
- `get_execution_state(store, run_id)`
- `get_evaluation_execution(store, run_id, case_id, metric_id)`
- `Reporter`
- `snapshot_report`
- `quickcheck`
- generation/evaluation bundle export and import helpers
- reduction/parse/score bundle export and import helpers in Python

Persistence boundaries:

- persistent stores are required for cross-run cache reuse
- `InMemoryRunStore` does not provide cross-run stage cache behavior
- use persistent stores whenever you need resume, reporting, comparison, export, imported-artifact replay, or cache-aware incremental reuse

Use persistent stores whenever the workflow needs resume, reporting, comparison, export, or later inspection from another process.

---
title: Stores and inspection reference
diataxis: reference
audience: users inspecting persisted runs
goal: Document persistence helpers, reporting/export APIs, and inspection helpers.
---

# Stores and inspection reference

## Store-related symbols

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `InMemoryRunStore` | Store implementation | The run is local, short-lived, and does not need reopen support | No cross-process persistence |
| `SqliteRunStore` | Store implementation | You want the default persistent local backend | Good default for resume, report, compare, and export |
| `RunStore` | Store protocol | You are implementing or typing against the storage abstraction | Use for custom backends or shared interfaces |
| `sqlite_store` | Store factory helper | You want a concise way to create a SQLite-backed store | Wraps the persistent SQLite backend |

## Inspection and export helpers

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `get_run_snapshot(store, run_id)` | Inspection helper | You want identity and provenance details for a stored run | Reads the compiled snapshot only |
| `get_execution_state(store, run_id)` | Inspection helper | You want stage progress, counts, and failure state | Best for resume decisions |
| `get_evaluation_execution(store, run_id, case_id, metric_id)` | Inspection helper | You want workflow-backed evaluation details for one case and metric | Applies to judge-backed metrics |
| `Reporter` | Reporting API | You want JSON, Markdown, CSV, or LaTeX output from stored runs | Works from persisted projections |
| `snapshot_report` | Summary helper | You want a concise snapshot-oriented report in Python | Smaller surface than full `Reporter` |
| `quickcheck` | Status helper | You want a compact run summary | Good for operational checks |
| Generation and evaluation bundle export/import helpers | Artifact portability helpers | You want portable generation or evaluation artifacts | Also exposed in the CLI for common handoff paths |
| Reduction, parse, and score bundle export/import helpers | Python-only artifact helpers | You want intermediate-stage handoff beyond the CLI boundary | Currently Python-only |

## Persistence boundaries

| Option | Best for | Persistence / runtime behavior | Caveats |
| --- | --- | --- | --- |
| Persistent stores | Resume, reporting, comparison, export, imported-artifact replay, and cache-aware incremental reuse | Preserve artifacts across processes and later sessions | Usually the right choice for real runs |
| `InMemoryRunStore` | Short local runs and deterministic smoke tests | Keeps artifacts only in the current process | No cross-run stage cache behavior |
| Cross-run cache reuse | Reusing stored upstream work over time | Depends on a persistent backend | Not available with memory-only storage |

persistent stores are required for cross-run cache reuse.
`InMemoryRunStore` does not provide cross-run stage cache behavior.

Use persistent stores whenever the workflow needs resume, reporting, comparison, export, or later inspection from another process.

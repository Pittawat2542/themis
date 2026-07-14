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
| `JsonlRunStore` | Store implementation | You want filesystem-readable snapshots, events, documents, and blobs | Simple local persistence; event reads scan JSONL unless a backend override is added |
| `PostgresRunStore` | Store implementation | You want database-backed metadata and event persistence | Requires the `postgres` extra and a filesystem blob root |
| `MongoDbRunStore` | Store implementation | You want MongoDB-backed metadata and event persistence | Requires the `mongodb` extra and a filesystem blob root |
| `RunStore` | Store protocol | You are implementing or typing against the storage abstraction | Use for custom backends or shared interfaces |
| `RunStoreBase` | Custom-backend base class | You want projections, resume, registry, checkpoints, cache, and freshness derived for you | Implement snapshot, event, run-id, document, blob, and deletion primitives only |
| `StorageConfig` | Store provenance model | You provide a custom backend | Expose a stable instance as the store's `storage_config` attribute |
| `memory_store`, `sqlite_store`, `jsonl_store`, `postgres_store`, `mongodb_store` | Store factory helpers | You want concise construction of a built-in backend | External backends still require their optional extras and connection settings |
| `register_store_backend`, `available_store_backends`, `create_run_store` | Store registry and factory | You need config-driven construction of a custom backend | Registration is process-local |

## Inspection and export helpers

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `get_run_snapshot(store, run_id)` | Inspection helper | You want identity and provenance details for a stored run | Reads the compiled snapshot only |
| `get_execution_state(store, run_id)` | Inspection helper | You want stage progress, counts, and failure state | Best for resume decisions |
| `get_evaluation_execution(store, run_id, case_id, metric_id)` | Inspection helper | You want workflow-backed evaluation details for one case and metric | Applies to judge-backed metrics |
| `get_case_audit`, `get_attempt_history`, `get_score_claim_history` | Audit and history helpers | You want case evidence, execution attempts, or prior score claims | History is derived from persisted events |
| `get_telemetry_summary` | Telemetry helper | You want provider-call, latency, token, retry, or failure aggregates | Missing telemetry yields empty summary fields |
| `get_projection` | Projection helper | You want a named stored read model plus freshness information | Accepts a projection consistency policy |
| `get_run_record`, `query_run_records`, `resolve_run_id`, `resolve_run_record` | Run registry helpers | You want exact, filtered, or label-based run resolution | Ambiguous or missing resolution raises an error |
| `Reporter` | Default reporting API | You want typed summaries or JSON, Markdown, CSV, and LaTeX output from stored runs | Summary exports work from persisted projections; raw rows are available through `score_rows(...)` |
| `ReporterProtocol`, `register_reporter(...)`, `create_reporter(...)`, `available_reporters(...)` | Reporter boundary | You want a custom read-side export or reporting implementation over stored evidence | Reporter choice is not identity-bearing |
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

## Custom backend primitives

Subclass `RunStoreBase` and implement initialization; `write_snapshot` and
`read_snapshot`; idempotent `append_event` and sequenced `read_event_records`;
`list_run_ids`; generic `write_document`, `read_document`, and
`delete_document`; `store_blob` and `load_blob`; and `delete_run`. The base class
implements the consumer-facing `RunStore` operations. Built-in backends use the
same base and may override derived operations for efficiency.

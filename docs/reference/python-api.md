---
title: Python API reference
diataxis: reference
audience: Python users of Themis
goal: Provide a generated entry point for the public Python surface.
---

# Python API reference

This page is the generated entry point into the public Python API. Use the smaller reference pages in this section when you already know the category of symbol you need.

## Root exports

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `__version__` | Constant | You want the installed package version | Useful for docs, debugging, and release checks |
| `Experiment` | Core class | You want the main reusable experiment authoring surface | Use for config-backed or Python-authored experiments |
| `DatasetSourceSpec` | Config model | You want reusable, identity-bearing dataset inputs for experiments | Use with `Experiment(dataset_sources=[...])` or config-authored dataset sources |
| `ExecutionCheckpoint` | Store model | You want to inspect or implement checkpointed resume state | Acceleration artifact; events remain authoritative |
| `InMemoryRunStore` | Store implementation | You want ephemeral local storage | No cross-process persistence |
| `ProjectionCursor` | Store model | You want to inspect projection progress over a run event stream | Used by store backends to track projection freshness |
| `PromptSpec` | Prompt model | You want prompt instructions, prefixes, suffixes, or prompt blocks as part of experiment identity | Shared across generation and builtin judge workflows |
| `Reporter` | Reporting API | You want exports such as JSON, Markdown, CSV, or LaTeX | Works from stored projections |
| `RerunPlan` / `RerunSelector` | Rerun models | You want targeted rerun request payloads | Used by `Experiment.rerun(...)` and the CLI rerun command |
| `RunEstimate` | Data model | You want planned task counts and token estimates | Informational only; not pricing |
| `RunLineage` | Registry model | You want parent-child run relationships such as replay or rerun links | Stored as part of run records |
| `RunQuery` | Registry model | You want to resolve or list stored runs by registry metadata | Supports tags, baselines, dataset identity, lineage, and timestamps |
| `RunRecord` | Registry model | You want the persisted registry view of one run | Includes metadata used by lookup and reporting |
| `RunResult` | Data model | You want the top-level execution result returned by a run | Includes status and benchmark output |
| `RunSnapshot` | Data model | You want the compiled identity and provenance artifact | Produced by `compile()` |
| `RunStatus` | Enum-like status model | You want run lifecycle state values | Useful in automation and inspection |
| `RunStore` | Storage protocol | You are typing against or implementing custom stores | Abstract interface rather than a concrete backend |
| `RuntimeConfig` | Config model | You want runtime tuning without changing logical identity | Covers concurrency, retries, and deferred execution paths |
| `SqliteRunStore` | Store implementation | You want the default persistent local store | Good default for real runs |
| `StatsEngine` | Analysis helper | You want statistical comparison utilities | Used in comparison and reporting flows |
| `evaluate` | Convenience function | You want the shortest synchronous Python path to a run | Best for simple scripts; call only when no event loop is already running |
| `evaluate_async` | Convenience function | You want the shortest async Python path to a run | Use in notebooks, async apps, and any environment with a running event loop |
| `export_evaluation_bundle` | Artifact helper | You want portable evaluation workflow artifacts | Best for judge-backed replay or handoff |
| `export_generation_bundle` | Artifact helper | You want portable generation artifacts | Good for external evaluation pipelines |
| `export_parse_bundle` | Artifact helper | You want portable parsed-output artifacts | Python-only today |
| `export_reduction_bundle` | Artifact helper | You want portable reduction-stage artifacts | Python-only today |
| `export_score_bundle` | Artifact helper | You want portable pure-score artifacts | Python-only today |
| `get_case_audit` | Inspection helper | You want full per-case generation, parse, reduction, metric, and failure artifacts | Best for debugging or audit trails |
| `get_evaluation_execution` | Inspection helper | You want one stored workflow execution | Judge-backed metrics only; pass `dataset_id` or `case_key` when duplicate `case_id`s exist across datasets |
| `get_execution_state` | Inspection helper | You want stored progress and failure details | Best before resume or replay decisions |
| `get_run_record` | Inspection helper | You want the registry record for a stored run | Raises if the run is not present in the store |
| `get_run_snapshot` | Inspection helper | You want compiled identity and provenance details | Read-only lookup |
| `get_telemetry_summary` | Inspection helper | You want run-level latency, token, retry, and cost summaries | Derived from persisted case audit projections |
| `import_evaluation_bundle` | Artifact helper | You want to ingest external evaluation artifacts into a store | Match bundle shape to the target run |
| `import_generation_bundle` | Artifact helper | You want to ingest generation artifacts into a store | Enables later replay without regeneration |
| `import_parse_bundle` | Artifact helper | You want to ingest parsed-output artifacts | Python-only today |
| `import_reduction_bundle` | Artifact helper | You want to ingest reduction-stage artifacts | Python-only today |
| `import_score_bundle` | Artifact helper | You want to ingest score artifacts | Python-only today |
| `quickcheck` | Inspection helper | You want a compact run summary | Smaller surface than full reporting |
| `snapshot_report` | Reporting helper | You want a concise Python report from stored snapshot data | Lighter than `Reporter` |
| `sqlite_store` | Store factory helper | You want a quick SQLite store constructor | Shortcut for the persistent local backend |

## Generated modules

Root package:

::: themis

Catalog namespace:

::: themis.catalog

Core namespace:

::: themis.core

Adapters:

::: themis.adapters

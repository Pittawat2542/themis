# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Changed
- Added identity-bearing metric interpretation contracts and neutral `scored`
  outcomes for metrics without binary correctness semantics.
- Made omitted provider request limits unlimited and added strict seed-capability
  validation with requested/applied seed telemetry.
- Introduced schema-v2 idempotent event IDs, execution attempts, sequenced event
  queries, explicit projection freshness, and non-blocking evidence writes.
- Added bounded worker leases, definition-root and digest validation, optional
  HMAC-signed manifests, and terminal deferred-execution failure records.
- Added streaming JSONL reads and measured SQLite load baselines at 1k and 10k
  events.

### Security
- Added `minimal`, `standard`, and `full` evidence retention policies with
  standard sanitization by default and a strict provider-header allowlist.
- Sanitized runtime events, JSON/text blobs, stage caches, tracebacks, and
  imported/exported bundles before persistence.

## [6.0.0] - 2026-07-13

### Changed
- Consolidated all public authoring paths on `themis.Experiment` and made
  catalog kits return that canonical type.
- Made frozen models deeply immutable and allowed `Experiment.compile()` to
  record explicit store and runtime provenance.
- Completed the focused component, runtime, storage, analysis, preset, and
  artifact APIs and versioned CLI JSON responses.

### Breaking Changes
- Removed public core-returning launcher and catalog APIs, moved presets to
  `themis.presets`, and changed machine-readable CLI output to a v1 envelope.

## [5.0.0] - 2026-07-12

Themis 5 removes the v4 compatibility surface and establishes one Python-first
contract for researchers and evaluation engineers.

### Changed
- Replaced the broad root namespace with `Experiment`, `Generation`,
  `Evaluation`, `RunOptions`, core data models, and the small `evaluate` helper.
- Unified generators around `generate(...) -> Candidate` and workflow metrics
  around `WorkflowMetric` with an explicit subject kind.
- Made Python modules canonical experiment definitions; YAML and TOML are now
  operational launchers using `definition: module:symbol`.
- Added explicit metric direction and pairing coverage to statistical comparisons.

### Security
- Removed implicit host execution from code metrics. Unsafe local subprocess
  execution now requires `allow_unsafe=True`; code benchmarks require an
  explicitly configured sandbox executor.

### Removed
- Removed session/generation aliases, lifecycle mega-subscriber hooks, generic
  graph/execution-backend APIs, v4 config loading, and root re-export sprawl.

## [4.0.2] - 2026-04-06

Themis 4.0.2 is a patch release that tightens live benchmark materialization
and adds a catalog-wide materialization checker for release validation.

### Added
- Added an exhaustive catalog materialization checker script for validating
  shipped benchmark recipes against live dataset sources.

### Fixed
- Fixed catalog benchmark loading to handle streamed Hugging Face datasets and
  benchmark-specific raw-file materialization paths.
- Fixed live benchmark materialization for HealthBench, LiveCodeBench,
  ProcBench, RoleBench, SuperChem, and related benchmark variants.
- Fixed multiple-choice prompt rendering for benchmark rows with more than ten
  answer options.

## [4.0.1] - 2026-04-06

Themis 4.0.1 is a patch release that stabilizes benchmark catalog workflows,
docs rendering, and release validation after the first stable public launch.

### Fixed
- Fixed catalog benchmark loading and validation so public benchmark discovery,
  materialization, and monkeypatched loader scenarios behave consistently.
- Fixed built-in code benchmark wiring to support HumanEval-style execution
  scoring in the benchmark catalog.
- Fixed docs-site rendering issues affecting dark mode presentation and Mermaid
  diagrams.
- Fixed release packaging validation so built-wheel smoke tests verify the
  current release version instead of a stale hardcoded value.

### Changed
- Updated package metadata, runtime provenance defaults, and citation metadata
  for the 4.0.1 release.

## [4.0.0] - 2026-04-05

Themis 4.0.0 is the first stable public release and marks a complete
architectural overhaul of the framework. The v4 line replaces the legacy
system with a typed, composable evaluation platform built for production
evaluation workflows.

### Added
- Added a typed evaluation runtime centered on `Experiment(...)` and
  `evaluate(...)`, with datasets, generators, parsers, reducers, metrics,
  judge workflows, snapshots, and reproducible run artifacts.
- Added a stable public API surface including `Experiment`, `evaluate`,
  `Reporter`, `RunStore`, `RunSnapshot`, `StatsEngine`, and inspection/export
  utilities.
- Added the `themis` CLI with end-to-end workflow coverage across `run`,
  `resume`, `replay`, `estimate`, `quickcheck`, `report`, `inspect`,
  `compare`, `export`, `submit`, `worker`, `batch`, `init`, and
  `quick-eval`.
- Added pluggable persistent run storage backends for `sqlite`, `jsonl`,
  `mongodb`, `postgres`, and in-memory execution.
- Added a benchmark catalog system with reusable components, manifests,
  dataset materialization workflows, and named benchmark execution paths.
- Added reporting, comparison, replay, inspection, export/import, and
  stage-aware artifact introspection workflows.
- Added integration surfaces and optional extras for `openai`, `vllm`,
  `langgraph`, `datasets`, `mongodb`, and `postgres`.
- Added a Diataxis-style documentation set, runnable examples, contributor
  guides, and explicit release/versioning practices.
- Added stronger CI, docs validation, packaging metadata, and release
  automation for the 4.0.0 lifecycle.

### Changed
- Replaced the internal and experimental pre-v4 system with a production-ready,
  user-facing evaluation platform.
- Established reproducible, scriptable workflows across CLI execution,
  persistent storage, benchmark cataloging, reporting, and inspection.

### Breaking Changes
- v4 is a full rewrite and completely replaces the pre-v4 codebase.
- Previous runtime, storage, and catalog APIs are not compatible with v4.
- Legacy import paths, workflows, and module structure have been removed or
  redesigned.

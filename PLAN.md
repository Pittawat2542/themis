# PLAN.md: Themis Native Operational Depth Roadmap

## Summary
Build the strongest practical capabilities from `olmo-eval` into Themis without importing, vendoring, or depending on `olmo-eval`.

The implementation must preserve Themis philosophy:
- Python **Executable Definitions** remain canonical.
- Suites, presets, CLI flags, queue manifests, and batch requests are **Automation Surfaces**.
- Every executable unit still compiles to a `RunSnapshot`.
- Scores remain **Score Claims** over persisted **Evidence**.
- Heavy infrastructure remains optional.

## Global Implementation Rules
- Use TDD for every phase: failing test, minimal implementation, passing test, refactor.
- Keep changes behind Themis-native modules and existing seams.
- Do not add `olmo-eval` as a dependency or copy its implementation.
- Keep default test loop deterministic and free of Docker, network, GPU, or real provider requirements.
- Make conventional commits along the way:
  - `docs: record operational surface guardrails`
  - `feat(catalog): add benchmark experiment kits`
  - `feat(catalog): add suite definitions`
  - `feat(core): add runtime presets`
  - `feat(core): add resource planning`
  - `feat(core): add code execution backend`
  - `feat(core): record provider telemetry`
  - `feat(reporting): add suite coverage and pairwise claims`
  - `feat(submission): enrich deferred execution manifests`
  - `docs: document operational evaluation workflows`
  - `test: cover benchmark kits and suites`

## Phase 0: Guardrails And Vocabulary

### Expected Features
- Record that Themis can support benchmark kits, suites, runtime presets, resource plans, telemetry, and infrastructure adapters only as automation surfaces over executable Python definitions.
- Clarify that runtime resource allocation is provenance unless it changes logical experiment behavior.
- Establish names before adding modules so future engineers do not invent competing terms.

### Expected API Changes
- No runtime API changes.
- Documentation vocabulary additions:
  - `Benchmark Kit`
  - `Suite Definition`
  - `Runtime Preset`
  - `Execution Resource Plan`
  - `Provider Telemetry`

### Implementation Notes
- Add ADR: `docs/adr/0002-operational-surfaces-remain-automation-surfaces.md`.
- Extend `CONTEXT.md`.
- Keep ADR consistent with `docs/adr/0001-python-executable-definitions-are-canonical.md`.

### Tests Affected
- `tests/test_docs_site.py`
- `tests/test_public_docs.py`
- docs inventory tests if they enforce glossary/topic coverage.

### Definition Of Done
- ADR exists and clearly states no new operational surface becomes canonical.
- `CONTEXT.md` contains all new terms.
- Docs tests pass.
- Conventional commit: `docs: record operational surface guardrails`.

## Phase 1: Executable Benchmark Kits

### Expected Features
- The catalog can produce complete `Experiment` objects from benchmark entries.
- Built-in kit families cover:
  - math final-answer
  - multiple choice
  - rubric-judged QA
  - panel-judged QA
  - code generation
- Users can inspect available kits and override defaults in Python.

### Expected API Changes
Add:
- `themis.catalog.build_benchmark_experiment(...)`
- `themis.catalog.get_benchmark_kit(...)`
- `themis.catalog.list_benchmark_kits(...)`

Add types:
- `BenchmarkExperimentDefaults`
- `BenchmarkKit`

### Implementation Notes
- Create `themis/catalog/benchmarks/kits.py`.
- Kits should wrap existing catalog materialization and return `Experiment`.
- Use existing `GenerationConfig`, `EvaluationConfig`, `StorageConfig`, and `RuntimeConfig`.
- Update `themis.catalog.run(...)` to prefer kits but preserve current behavior.
- Kit overrides that change generation/evaluation behavior must affect run identity. Storage/runtime-only changes should affect provenance only.

### Tests Affected
- Add `tests/catalog/test_benchmark_kits.py`.
- Update `tests/catalog/test_catalog_run.py`.
- Update docs/example tests if new examples are added.

### Definition Of Done
- Every existing manifest benchmark either has a kit or a deliberate fallback path.
- Kit-built experiments compile successfully.
- At least one benchmark per kit family runs with deterministic demo components.
- Public discovery functions are documented and tested.
- Conventional commit: `feat(catalog): add benchmark experiment kits`.

## Phase 2: Suite Definitions As Automation Surface

### Expected Features
- Users can define and run named suites that expand into multiple benchmark or executable definitions.
- Suites can contain benchmark ids and nested suite ids.
- Suite runs produce multiple normal Themis runs, each with its own `RunSnapshot`.
- Suite membership is recorded as run metadata, tags, or lineage, not as a replacement run identity.

### Expected API Changes
Add:
- `themis.catalog.register_suite(...)`
- `themis.catalog.get_suite(...)`
- `themis.catalog.list_suites(...)`
- `themis.catalog.expand_suite(...)`

Add types:
- `SuiteDefinition`
- `SuiteItem`
- `SuiteExpansion`
- `SuiteAggregation`

CLI additions:
- `themis suite list`
- `themis suite inspect <suite-id>`
- `themis suite run <suite-id>`

### Implementation Notes
- Create `themis/catalog/suites.py`.
- Add starter built-in suites:
  - `math-core`
  - `qa-core`
  - `code-core`
  - `general-core`
- Detect cycles during expansion.
- Prevent suite ids from colliding with benchmark ids.
- Suite aggregation should read persisted projections and reporter summaries.

### Tests Affected
- Add `tests/catalog/test_suites.py`.
- Add `tests/cli/test_suite_cli.py`.
- Update reporter tests once suite summary exists.

### Definition Of Done
- Nested suite expansion is deterministic.
- Cycles fail with a clear error.
- Running a suite creates one run per expanded executable item.
- Suite aggregation works from stored evidence.
- Conventional commit: `feat(catalog): add suite definitions`.

## Phase 3: Runtime Presets

### Expected Features
- Named presets make common workflows easy without introducing a separate harness model.
- Presets resolve to normal Themis config models.
- Presets support baseline, judge-backed, multi-candidate, code-execution, fast local, and careful local workflows.

### Expected API Changes
Add:
- `themis.core.get_preset(...)`
- `themis.core.list_presets(...)`
- `themis.core.apply_preset(...)`

Add types:
- `RuntimePreset`
- `SessionPreset`
- `EvaluationPreset`
- `ExperimentPreset`

Possible root exports:
- `get_preset`
- `list_presets`
- `apply_preset`

### Implementation Notes
- Create `themis/core/presets.py`.
- Applying a preset returns a new `Experiment`; never mutate the original.
- Preset ids and resolved values should be recorded in provenance metadata.
- Runtime-only presets must not affect run identity.
- Generation/evaluation presets may affect run identity.

### Tests Affected
- Add `tests/core/test_presets.py`.
- Update `tests/test_public_api.py` if root exports are added.
- Update docs inventory tests.

### Definition Of Done
- Preset application is deterministic.
- Runtime-only preset changes provenance, not run id.
- Logical preset changes alter run id.
- Invalid preset ids include close-match suggestions.
- Conventional commit: `feat(core): add runtime presets`.

## Phase 4: Resource-Aware Planning

### Expected Features
- The planner reports expected work before running:
  - generation calls
  - judge calls
  - parse/score work
  - required execution backends
  - provider call counts
  - stage concurrency
  - warnings
- Resource planning is inspectable and can be included in submission manifests.

### Expected API Changes
Add:
- `Planner.resource_plan(snapshot, runtime)`

Extend or add models:
- `ExecutionResourcePlan`
- optional fields on `RunEstimate`

CLI output changes:
- `themis run estimate`
- `themis quickcheck`

### Implementation Notes
- Extend `themis/core/planner.py`.
- Add resource models to `themis/core/results.py` or a focused planning module if `results.py` becomes too broad.
- Keep resource plan out of `RunIdentity`.
- Store the plan as provenance or an early evidence projection when useful.

### Tests Affected
- Extend `tests/core/test_planner.py`.
- Extend `tests/cli/test_direct_command_units.py`.
- Add tests confirming resource plan does not affect run id.

### Definition Of Done
- Estimate output reports resource demand.
- Multi-candidate and workflow-backed metrics increase expected work.
- Code-execution metrics declare backend requirements.
- Resource-only differences do not change run identity.
- Conventional commit: `feat(core): add resource planning`.

## Phase 5: Sandboxed Code Execution

### Expected Features
- Code-generation benchmarks can execute submitted code safely through Themis-owned backends.
- Local subprocess execution works by default for deterministic tests.
- Docker execution is optional and externally marked.
- Execution logs, stdout, stderr, timing, exit code, and timeout status are persisted as Evidence.

### Expected API Changes
Add types:
- `CodeExecutionRequest`
- `CodeExecutionResult`
- `CodeExecutionLimits`
- `CodeExecutionStatus`

Add adapters:
- `LocalSubprocessExecutionBackend`
- `DockerExecutionBackend` behind optional dependency or external marker

Add builtins:
- code execution metric
- code result parser
- code execution failure categories

### Implementation Notes
- Create `themis/core/code_execution.py`.
- Extend `themis/core/execution_backends.py` only if the existing backend interface remains coherent; otherwise keep code execution in a focused module.
- Reuse `themis/catalog/builtins/code_execution.py` as the public builtin surface.
- Do not make Docker or container libraries required runtime dependencies.

### Tests Affected
- Add `tests/core/test_code_execution.py`.
- Extend `tests/catalog/test_code_benchmark_wiring.py`.
- Add external Docker tests, skipped by default.
- Ensure default `uv run pytest` does not require Docker.

### Definition Of Done
- Local backend can pass, fail, timeout, and capture output.
- Code metric returns structured `MetricResult` or `ScoreError`.
- Code execution artifacts are inspectable.
- Docker support is optional and externally tested.
- Conventional commit: `feat(core): add code execution backend`.

## Phase 6: Provider Telemetry As Evidence

### Expected Features
- Generation and judge calls record provider telemetry where available.
- Telemetry covers:
  - provider id
  - model id
  - latency
  - retry count
  - token usage
  - failure category
  - start/end timestamps
- Reporter and inspection helpers expose telemetry summaries.

### Expected API Changes
Add models:
- `ProviderTelemetry`
- `StageTelemetry`

Add events:
- `ProviderCallStartedEvent`
- `ProviderCallCompletedEvent`
- `ProviderCallFailedEvent`

Add projection:
- `telemetry_summary`

### Implementation Notes
- Extend `themis/core/models.py`, `events.py`, and `projections.py`.
- Update orchestrator generation and judge call paths.
- Update provider adapters to emit telemetry when possible.
- Missing telemetry should degrade to empty summaries, not failures.

### Tests Affected
- Extend `tests/core/test_events.py`.
- Add/extend `tests/adapters/test_provider_telemetry.py`.
- Extend `tests/core/test_reporter.py`.
- Add failure-path tests for provider telemetry.

### Definition Of Done
- Demo generator or fake provider emits minimal telemetry.
- Failed provider calls produce telemetry failure evidence.
- Reporter JSON includes telemetry summary.
- Telemetry does not affect run identity.
- Conventional commit: `feat(core): record provider telemetry`.

## Phase 7: Result Discovery, Suite Coverage, And Pairwise Claims

### Expected Features
- Themis can answer:
  - What runs exist for this benchmark or suite?
  - Which suite items have coverage?
  - Which runs are comparable?
  - What is the paired delta between baseline and candidate?
- Pairwise outputs are explicitly framed as **Score Claims** over Evidence.
- Missing coverage is reported, not silently ignored.

### Expected API Changes
Add reporter methods:
- `Reporter.suite_summary(...)`
- `Reporter.suite_coverage(...)`
- `Reporter.compare_runs(...)`
- `Reporter.compare_latest(...)`

Add models:
- `SuiteCoverageSummary`
- `PairwiseComparisonReport`
- possibly `ScoreClaim`

CLI additions:
- `themis report suite`
- `themis compare runs`
- `themis compare latest`
- `themis inspect coverage`

### Implementation Notes
- Build on `StatsEngine.compare(...)`.
- Extend run records only as needed for suite id, benchmark id, dataset ids, metric ids, tags, baseline label, and lineage.
- Comparison must use matched case keys.
- Output should include source run ids, pair counts, missing rows, dropped rows, and metric ids.

### Tests Affected
- Extend `tests/core/test_stats.py`.
- Extend `tests/core/test_reporter.py`.
- Extend `tests/cli/test_report_compare_export_cli.py`.
- Add coverage tests for partial suites.

### Definition Of Done
- Pairwise compare ignores unmatched rows and reports counts.
- Suite coverage is deterministic and visible in JSON/Markdown.
- CLI supports both exact run comparison and latest-baseline comparison.
- All comparison reports cite evidence sources.
- Conventional commit: `feat(reporting): add suite coverage and pairwise claims`.

## Phase 8: Operational Submission And Batch Improvements

### Expected Features
- Deferred execution manifests carry enough context for suites, presets, tags, and resource plans.
- Batch suite submission creates one request per expanded executable item.
- Workers can inspect pending, claimed, completed, and failed requests.
- Stored snapshot equality is validated before execution.

### Expected API Changes
Extend `SubmissionManifest` with optional:
- `suite_id`
- `preset_ids`
- `resource_plan`
- `tags`
- `created_at`

CLI additions or extensions:
- worker inspect commands
- batch suite submission commands

### Implementation Notes
- Update `themis/core/submission.py`.
- Keep filesystem queue as the default adapter.
- Avoid Beaker/Modal/provider-specific launch assumptions.
- Future launch adapters should satisfy existing execution interfaces.

### Tests Affected
- Extend `tests/core/test_submission.py`.
- Extend `tests/cli/test_execution_modes.py`.
- Add backwards-compatibility manifest loading tests.

### Definition Of Done
- Old manifests still load.
- New manifests round-trip.
- Mismatched stored snapshots fail clearly.
- Suite batch submission creates deterministic request files.
- Failed requests leave inspectable evidence.
- Conventional commit: `feat(submission): enrich deferred execution manifests`.

## Phase 9: Documentation, Packaging, And Release Gates

### Expected Features
- New workflows are discoverable through docs and runnable examples.
- Public API additions are intentional and tested.
- Optional dependencies remain isolated.
- Release checks prevent docs/API drift.

### Expected API Changes
No new runtime API unless earlier phases choose root exports.

Optional dependency groups may be added:
- `code-exec`
- `docker`

### Implementation Notes
- Add examples:
  - first suite run
  - benchmark kit customization
  - local code benchmark execution
  - compare two runs
- Update docs:
  - benchmark catalog reference
  - suite how-to
  - runtime presets
  - resource planning
  - telemetry
  - score claims and pairwise comparisons
- Update `scripts/docs/build_inventory.py`.

### Tests Affected
- `tests/test_docs_site.py`
- `tests/test_docs_inventory.py`
- `tests/test_docs_examples.py`
- `tests/test_readme_examples.py`
- `tests/test_release_metadata.py`
- `tests/test_public_api.py`

### Definition Of Done
- `uv run pytest` passes.
- `uv run mypy themis tests` passes.
- `uv run ruff check themis tests examples scripts` passes.
- `uv run mkdocs build --strict` passes.
- Docs inventory includes suites, presets, benchmark kits, and new public exports.
- Conventional commit: `docs: document operational evaluation workflows`.

## Recommended Rollout Order
1. Phase 0: guardrails and vocabulary.
2. Phase 1: benchmark kits.
3. Phase 2: suites.
4. Phase 7 partial: suite summaries and coverage on top of existing stats.
5. Phase 3: runtime presets.
6. Phase 4: resource planning.
7. Phase 6: provider telemetry.
8. Phase 5: sandboxed code execution.
9. Phase 8: submission and batch improvements.
10. Phase 9: docs and release hardening continuously.

## Cross-Phase Test Hygiene
- Prefer public-interface tests over private helper tests.
- Use fake providers, fake datasets, and in-memory stores by default.
- Mark Docker/network/GPU/provider tests as `external`.
- Avoid snapshot tests that lock incidental formatting.
- Keep each phase independently shippable.
- Do not broaden default dependencies unless a feature truly belongs in core.

## Cross-Phase Commit Hygiene
- Commit after each phase reaches its Definition of Done.
- Use conventional commits with scopes.
- Keep docs and tests in the same commit as the feature when they describe behavior introduced by that feature.
- Do not mix unrelated refactors into feature commits.
- Suggested commit order:
  - `docs: record operational surface guardrails`
  - `feat(catalog): add benchmark experiment kits`
  - `feat(catalog): add suite definitions`
  - `feat(core): add runtime presets`
  - `feat(core): add resource planning`
  - `feat(core): add code execution backend`
  - `feat(core): record provider telemetry`
  - `feat(reporting): add suite coverage and pairwise claims`
  - `feat(submission): enrich deferred execution manifests`
  - `docs: document operational evaluation workflows`

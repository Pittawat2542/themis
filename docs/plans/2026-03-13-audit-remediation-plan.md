# Codebase Audit Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce the architectural, typing, and maintenance risks identified in the audit without regressing the supported user workflows, replay correctness, resume behavior, or the ability to read and migrate persisted data.

**Architecture:** Use an incremental, characterization-test-first refactor. First lock down the event log, projection, and documented workflow behavior with stronger tests; then add typed boundaries around events and plugin resolution; finally decompose the oversized orchestration and projection modules behind stable entry points, allowing deprecation-backed API cleanup where the current shape is leaky.

**Tech Stack:** Python 3.12, Pydantic v2, sqlite3, pytest, uv, mkdocs

---

## Decision Weights

Use the following weighting when choosing between alternatives:

- **Correctness / behavioral preservation:** 35%
- **Migration safety / backwards compatibility:** 25%
- **Architectural leverage:** 20%
- **Implementation cost:** 10%
- **Public API and docs impact:** 10%

These weights favor incremental internal refactors over large rewrites, and typed compatibility layers over flag-day migrations.

## Hard Contracts

- Preserve a small, documented top-level public API, with deprecation for removals or relocations.
- Preserve deterministic identity semantics and the ability to resolve or migrate existing persisted hashes.
- Preserve append-only event semantics, replay correctness, and the ability to read or migrate existing stores.
- Keep optional features opt-in, with actionable install guidance when dependencies are missing.
- Preserve the core user workflows exposed through `Orchestrator` and `ExperimentResult`: run, inspect, compare, and report.
- Keep docs examples and quick-start imports aligned with the supported public API.

## Negotiable Implementation Details

- The exact export inventory in `themis/__init__.py` may evolve as long as the documented top-level API remains small and migration-safe.
- The internal representation of `spec_hash`, `transform_hash`, and `evaluation_hash` may evolve as long as deterministic identity and backwards resolution remain intact.
- SQLite schema details and persisted JSON envelopes may change behind compatibility decoding and explicit migrations.
- `Orchestrator` and `ExperimentResult` constructor signatures, stored attributes, and helper methods may be tightened if the core workflows remain stable and deprecations are documented.
- Optional dependency extra names and exact error strings may change if the feature boundaries and install guidance remain clear.

## Issue Coverage Matrix

| Issue from audit | Primary workstream | Recommended choice | Why this choice wins under the weighting |
| --- | --- | --- | --- |
| Stringly typed event metadata | Tasks 2, 4 | Introduce typed metadata models with compatibility decoding | Highest gain in correctness and replay safety without forcing a store rewrite |
| God objects in orchestration | Tasks 3, 10 | Incremental extraction behind stable entry points, with deprecations allowed for leaky constructor details | Safer than rewriting the runtime while still allowing API cleanup where the current shape exposes infrastructure concerns |
| Repository bloat in projections | Task 4 | Split replay, query, and view-building responsibilities | High leverage with low external impact |
| Service locator drift in plugin registry | Task 5 | Resolve plugin instances into explicit stage executables per session | Makes dependencies explicit without introducing a heavy DI framework |
| Hook weak typing / method-by-name dispatch | Task 5 | Add typed internal adapters before changing public hook API | Preserves compatibility while improving internal rigor |
| Compatibility shims and stale transitional code | Task 10 | Add deprecation path, then remove in one minor-version cycle | Balances cleanup with migration safety |
| Report builder mixed concerns | Task 7 | Keep facade, extract assemblers/services underneath | Preserves ergonomics while reducing class size |
| Telemetry string events and subscriber fragility | Task 6 | Introduce typed event names and guarded subscriber dispatch | Improves robustness with little API churn |
| AST-based governance brittleness | Task 8 | Narrow tests to architecture invariants, not stylistic incidental details | Retains governance value without making refactors painful |
| Lazy re-export IDE friction | Task 8 | Keep lazy loading, add explicit docs/stubs where helpful | Avoids import-cost regressions while improving discoverability |
| Short hash collision risk | Task 9 | Add collision detection and an extension path before changing lengths | Best migration-safety tradeoff |
| Overlay invalidation complexity | Tasks 2, 4 | Centralize overlay visibility rules around typed overlay metadata | Correctness-critical and tightly coupled to event typing |
| Duplication between staged runtime paths | Tasks 3, 10 | Consolidate shared candidate execution and retire legacy wrapper | Removes maintenance drag without changing behavior |
| Concrete storage wiring in facade | Task 3 | Preserve workflow ergonomics, not constructor signatures; extract factories/builders and narrow exposed state | Improves API quality without forcing a DI container or a flag-day rewrite |
| Extraction chain limitations | Task 5 | Keep fallback chain, add richer extraction result policy hooks internally | Avoids over-design while unblocking future policies |

## Execution Order

1. Strengthen the behavioral safety net.
2. Type the event metadata and overlay semantics.
3. Break up orchestration while behavior is locked down.
4. Break up projection replay/query responsibilities.
5. Resolve plugins and hooks into explicit stage executables.
6. Harden telemetry and observability.
7. Clean up reporting/statistics assembly.
8. Tighten governance and public-surface support code.
9. Add identity and storage hardening.
10. Remove transitional shims and dead compatibility paths.

This order matters: Tasks 2 through 5 are easier and safer once the baseline tests in Task 1 exist, and Task 10 should happen only after the new internal seams are in place.

### Task 1: Lock the Current Behavioral Baseline

**Issues covered:** regression risk during refactors, brittle hidden contracts, public workflow drift

**Files:**
- Modify: `tests/orchestration/test_trial_runner.py`
- Modify: `tests/orchestration/test_executor.py`
- Modify: `tests/storage/test_event_repo.py`
- Modify: `tests/storage/test_projection_replay.py`
- Modify: `tests/contracts/test_public_api.py`
- Modify: `tests/docs/test_docs_consistency.py`

**Implementation steps:**
1. Add characterization tests for the exact event sequences emitted by generation, transform, and evaluation paths.
2. Add characterization tests for overlay visibility and projection refresh edge cases.
3. Add tests that pin the documented `Orchestrator` and `ExperimentResult` workflows through the public API only, while avoiding over-specification of incidental constructor state.
4. Add tests that explicitly cover lazy import behavior and optional dependency messages.

**Validation:**
- Run: `uv run pytest tests/orchestration/test_trial_runner.py tests/orchestration/test_executor.py -q`
- Run: `uv run pytest tests/storage/test_event_repo.py tests/storage/test_projection_replay.py -q`
- Run: `uv run pytest tests/contracts/test_public_api.py tests/docs/test_docs_consistency.py -q`

**Recommended choice:** Prefer characterization tests over rewrite-first cleanup.

**Why:** This maximizes behavioral preservation and gives later tasks a safe landing zone.

### Task 2: Replace Stringly Typed Event Metadata with Typed Models

**Issues covered:** stringly typed event metadata, overlay invalidation complexity, weak replay contracts

**Files:**
- Modify: `themis/types/events.py`
- Modify: `themis/orchestration/runner_events.py`
- Modify: `themis/orchestration/trial_runner.py`
- Modify: `themis/storage/event_repo.py`
- Modify: `themis/storage/_projection_overlay.py`
- Modify: `themis/storage/projection_repo.py`
- Modify: `tests/storage/test_event_repo.py`
- Modify: `tests/storage/test_projection_replay.py`
- Modify: `tests/orchestration/test_trial_runner.py`
- Modify: `docs/concepts/storage-and-resume.md`

**Implementation steps:**
1. Introduce typed metadata models for stage-specific events such as inference, extraction, evaluation, projection, and item/prompt events.
2. Add compatibility parsing in the repository layer so persisted legacy JSON dicts can still hydrate into the typed metadata models.
3. Update event emitters to construct typed metadata objects instead of ad hoc dicts.
4. Update overlay filtering and projection refresh logic to query typed metadata instead of raw string keys.
5. Keep JSON persistence shape compatible by serializing typed metadata back to JSON at the repository boundary.

**Validation:**
- Run: `uv run pytest tests/storage/test_event_repo.py tests/storage/test_projection_replay.py tests/orchestration/test_trial_runner.py -q`
- Run: `uv run pytest tests/orchestration/test_projection_handler.py -q`

**Options considered:**
- Option A: Add helper accessors around dict metadata only.
- Option B: Introduce typed metadata models with compatibility decoding.

**Recommended choice:** Option B.

**Why:** Option B scores better on correctness and long-term leverage. Option A is cheaper but leaves the core replay contract weak.

### Task 3: Decompose Orchestration Without Breaking the Facade

**Issues covered:** god objects, transaction-script drift, concrete wiring in facade, duplicated orchestration behavior

**Files:**
- Modify: `themis/orchestration/orchestrator.py`
- Modify: `themis/orchestration/executor.py`
- Modify: `themis/orchestration/trial_runner.py`
- Create: `themis/orchestration/session_preparer.py`
- Create: `themis/orchestration/generation_stage.py`
- Create: `themis/orchestration/overlay_stage.py`
- Create: `themis/orchestration/trial_finalizer.py`
- Modify: `tests/orchestration/test_orchestrator.py`
- Modify: `tests/orchestration/test_executor.py`
- Modify: `tests/orchestration/test_trial_runner.py`

**Implementation steps:**
1. Extract session preparation logic from `TrialRunner.prepare_trial_session()` into a dedicated collaborator.
2. Extract generation candidate execution and overlay candidate execution into separate internal services.
3. Extract terminal trial finalization and circuit-breaker updates into explicit collaborators.
4. Keep `Orchestrator` as the primary user-facing entry point and keep `TrialExecutor`/`TrialRunner` as internal compatibility seams while smaller services are extracted.
5. Tighten constructor and attribute leaks that expose storage wiring directly if they are not part of the documented API, using deprecations where needed.
6. Remove any duplicated sequencing logic that can be shared through internal stage executors.

**Validation:**
- Run: `uv run pytest tests/orchestration/test_orchestrator.py tests/orchestration/test_executor.py tests/orchestration/test_trial_runner.py -q`
- Run: `uv run pytest tests/test_resume.py -q`

**Options considered:**
- Option A: Full rewrite around a new workflow engine.
- Option B: Incremental extraction behind the current facade.

**Recommended choice:** Option B.

**Why:** The runtime already works; a rewrite would score poorly on migration safety and correctness preservation, while incremental extraction leaves room to improve API cohesion.

### Task 4: Split Projection Replay, Query, and View Building

**Issues covered:** repository bloat, overlay invalidation complexity, mixed replay/query responsibilities

**Files:**
- Modify: `themis/storage/projection_repo.py`
- Modify: `themis/orchestration/projection_handler.py`
- Modify: `themis/storage/_projection_overlay.py`
- Modify: `themis/storage/_projection_codec.py`
- Modify: `themis/storage/_projection_persistence.py`
- Create: `themis/storage/projection_materializer.py`
- Create: `themis/storage/projection_queries.py`
- Create: `themis/storage/timeline_views.py`
- Modify: `tests/storage/test_projection_repo.py`
- Modify: `tests/storage/test_projection_replay.py`
- Modify: `tests/orchestration/test_projection_handler.py`

**Implementation steps:**
1. Move event replay and `CandidateReplayState` assembly into a dedicated projection materializer.
2. Move read-only SQL query helpers into a separate query module.
3. Move `RecordTimelineView` assembly into a dedicated builder/view service.
4. Reduce `SqliteProjectionRepository` to orchestration and transaction coordination.
5. Make `ProjectionHandler` depend on a smaller projection refresh API, not the full repository surface.

**Validation:**
- Run: `uv run pytest tests/storage/test_projection_repo.py tests/storage/test_projection_replay.py tests/orchestration/test_projection_handler.py -q`

**Recommended choice:** Split by responsibility, not by stage.

**Why:** Replay, SQL queries, and timeline view construction have different change rates and testing needs.

### Task 5: Replace Runtime Service-Location with Explicit Stage Resolution

**Issues covered:** service locator drift, weak hook typing, reflective extractor compatibility, extraction policy limits

**Files:**
- Modify: `themis/registry/plugin_registry.py`
- Modify: `themis/registry/compatibility.py`
- Modify: `themis/orchestration/task_resolution.py`
- Modify: `themis/orchestration/candidate_pipeline.py`
- Modify: `themis/evaluation/judge_service.py`
- Create: `themis/orchestration/resolved_plugins.py`
- Modify: `tests/registry/test_plugin_registry.py`
- Modify: `tests/contracts/test_compatibility.py`
- Modify: `tests/orchestration/test_candidate_pipeline.py`
- Modify: `tests/evaluation/test_judge_service.py`

**Implementation steps:**
1. Introduce resolved stage objects that carry the concrete engine/extractor/metric/judge instances or factories needed for execution.
2. Resolve plugins once per trial session or per executor stage, instead of repeatedly looking them up deep in the runtime.
3. Introduce a typed internal hook adapter so execution code no longer depends on `getattr()` for every hook call.
4. Replace `_invoke_extractor()` reflection with an explicit compatibility adapter, then deprecate the legacy two-argument extractor shape.
5. Keep the existing `PluginRegistry` public API intact while making the internals dependency-explicit.

**Validation:**
- Run: `uv run pytest tests/registry/test_plugin_registry.py tests/contracts/test_compatibility.py tests/orchestration/test_candidate_pipeline.py tests/evaluation/test_judge_service.py -q`

**Options considered:**
- Option A: Keep the registry as a service locator and add more validation.
- Option B: Resolve executable stage dependencies before the hot path.
- Option C: Introduce a full DI container.

**Recommended choice:** Option B.

**Why:** It fixes hidden dependencies without imposing container complexity on library users.

### Task 6: Harden Telemetry and Observability

**Issues covered:** string telemetry event names, subscriber failure propagation, observability coupling

**Files:**
- Modify: `themis/telemetry/bus.py`
- Modify: `themis/telemetry/langfuse_callback.py`
- Modify: `themis/orchestration/runner_events.py`
- Modify: `themis/storage/observability.py`
- Modify: `tests/telemetry/test_bus.py`
- Modify: `tests/orchestration/test_trial_runner.py`

**Implementation steps:**
1. Introduce a typed set of telemetry event names or a small event enum used internally.
2. Guard subscriber dispatch so one failing subscriber does not crash unrelated runtime behavior unless explicitly configured to do so.
3. Keep the current `TelemetryBus` surface simple; do not replace it with a heavier event framework.
4. Ensure Langfuse persistence and overlay selection remain compatible with the new typed telemetry path.

**Validation:**
- Run: `uv run pytest tests/telemetry/test_bus.py tests/orchestration/test_trial_runner.py -q`

**Recommended choice:** Keep the lightweight bus, add typing and failure isolation.

**Why:** The bus is already a good fit; robustness is the problem, not the pattern itself.

### Task 7: Reduce Reporting and Statistics Coupling

**Issues covered:** mixed concerns in report builder, reporting/runtime boundary blur, data-frame assembly duplication

**Files:**
- Modify: `themis/report/builder.py`
- Modify: `themis/runtime/comparison.py`
- Modify: `themis/stats/stats_engine.py`
- Modify: `themis/report/exporters.py`
- Create: `themis/report/metric_frame_builder.py`
- Create: `themis/report/report_metadata_builder.py`
- Modify: `tests/report/test_builder.py`
- Modify: `tests/report/test_exporters.py`
- Modify: `tests/stats/test_stats_engine.py`

**Implementation steps:**
1. Extract metric-frame assembly from `ReportBuilder`.
2. Extract report metadata/provenance assembly from `ReportBuilder`.
3. Keep `ReportBuilder` as the public fluent facade.
4. Reuse shared score-frame logic between comparison and reporting where practical.
5. Do not rewrite the statistics engine unless behavior or correctness gaps appear in the tests.

**Validation:**
- Run: `uv run pytest tests/report/test_builder.py tests/report/test_exporters.py tests/stats/test_stats_engine.py -q`

**Recommended choice:** Decompose the builder, preserve the user-facing entry point.

**Why:** This lowers maintenance cost without forcing user-facing churn.

### Task 8: Refine Governance, Public-Surface Support Code, and Documentation Contracts

**Issues covered:** AST test brittleness, lazy-import discoverability friction, governance false positives

**Files:**
- Modify: `tests/contracts/test_typing_conventions.py`
- Modify: `tests/contracts/test_public_api.py`
- Modify: `tests/docs/test_docs_consistency.py`
- Modify: `themis/records/__init__.py`
- Modify: `themis/types/__init__.py`
- Modify: `themis/stats/__init__.py`
- Modify: `docs/concepts/architecture.md`
- Modify: `README.md`

**Implementation steps:**
1. Keep governance tests that enforce true architecture constraints: documented public API, optional feature boundaries, doc alignment, and explicit typing in key modules.
2. Narrow AST-based tests so they do not block harmless refactors for incidental reasons.
3. Improve docs around lazy-loaded exports and optional extras.
4. If needed, add `.pyi` stubs or more explicit `__all__`/docstrings to improve IDE discoverability without removing lazy loading.

**Validation:**
- Run: `uv run pytest tests/contracts/test_typing_conventions.py tests/contracts/test_public_api.py tests/docs/test_docs_consistency.py -q`

**Recommended choice:** Keep governance, make it more intentional.

**Why:** Governance is a strength of the repo; it just needs to target the right invariants.

### Task 9: Add Identity, Hashing, and Storage Hardening

**Issues covered:** short-hash collision risk, storage evolution safety, identity ABI ambiguity

**Files:**
- Modify: `themis/types/hashable.py`
- Modify: `themis/specs/base.py`
- Modify: `themis/orchestration/task_resolution.py`
- Modify: `themis/storage/sqlite_schema.py`
- Modify: `themis/storage/event_repo.py`
- Modify: `tests/types/test_hashable.py`
- Modify: `tests/storage/test_sqlite_schema.py`
- Modify: `docs/concepts/storage-and-resume.md`

**Implementation steps:**
1. Add explicit collision detection when persisting specs or events whose short hashes unexpectedly collide with different canonical payloads.
2. Document that short hashes are user-facing identifiers layered over canonical payload identity, not the only possible internal identity representation.
3. Decide whether to keep 12-char hashes as stable public aliases while introducing fuller internal identities for persistence and joins.
4. If fuller internal identities are adopted, stage them behind a dedicated store-format migration with backwards resolution from existing short hashes.

**Validation:**
- Run: `uv run pytest tests/types/test_hashable.py tests/storage/test_sqlite_schema.py tests/storage/test_event_repo.py -q`

**Options considered:**
- Option A: Immediately switch all internal keys to full SHA-256 hashes.
- Option B: Keep short hashes as the only identity layer, adding collision detection only.
- Option C: Introduce fuller internal identities behind compatibility decoding while retaining short user-facing aliases.

**Recommended choice:** Option C if the migration surface is manageable; otherwise Option B as an interim hardening step.

**Why:** This keeps the user-facing ergonomics of short identifiers while giving the storage layer a path toward stronger identity guarantees. If that migration proves too invasive, Option B still reduces near-term risk.

### Task 10: Remove Transitional Paths and Dead Compatibility Layers

**Issues covered:** stale shims, duplicated staged runtime paths, reflective compatibility code

**Files:**
- Modify: `themis/orchestration/candidate_pipeline.py`
- Modify: `themis/orchestration/executor.py`
- Modify: `themis/specs/base.py`
- Modify: `themis/registry/plugin_registry.py`
- Modify: `CHANGELOG.md`
- Modify: `docs/changelog/index.md`
- Modify: `tests/orchestration/test_candidate_pipeline.py`
- Modify: `tests/specs/test_spec_base.py`

**Implementation steps:**
1. Mark legacy compatibility helpers as deprecated in code and docs.
2. Remove `execute_candidate_pipeline()` once the new staged execution path is the only supported internal path.
3. Remove or narrow `SpecBase.validate_semantic()` once all callers are gone.
4. Remove reflective fallback behavior that supports obsolete extractor signatures after the compatibility window closes.
5. Remove any alias APIs that only preserve old internal naming and have no external value.

**Validation:**
- Run: `uv run pytest tests/orchestration/test_candidate_pipeline.py tests/specs/test_spec_base.py tests/contracts/test_public_api.py -q`
- Run: `uv run pytest -q`

**Recommended choice:** Defer removal until after Tasks 2 through 5 land.

**Why:** Deleting compatibility layers early would make the larger refactors harder, not easier.

## Recommended Delivery Strategy

- **Phase A: Safety and typing**
  - Execute Tasks 1 and 2 first.
- **Phase B: Core runtime decomposition**
  - Execute Tasks 3, 4, and 5 next.
- **Phase C: Peripheral hardening**
  - Execute Tasks 6, 7, 8, and 9.
- **Phase D: Cleanup**
  - Execute Task 10 last.

## Recommended Commit Strategy

- `test: add orchestration and projection characterization coverage`
- `refactor: add typed trial event metadata models`
- `refactor: extract trial session preparation and stage executors`
- `refactor: split projection replay and timeline view assembly`
- `refactor: resolve stage plugins explicitly before execution`
- `refactor: harden telemetry dispatch and observability persistence`
- `refactor: extract report frame and metadata builders`
- `chore: tighten public-surface governance and docs contracts`
- `chore: add hash collision guards and storage notes`
- `chore: remove deprecated internal compatibility helpers`

## Exit Criteria

- `uv run pytest -q` passes.
- Public imports in `README.md` and docs stay aligned with the supported top-level API.
- Existing SQLite stores continue to hydrate and replay successfully, or have an explicit, tested migration path.
- Event replay and overlay selection no longer depend on string-key conventions scattered across the codebase.
- `TrialRunner`, `TrialExecutor`, and `SqliteProjectionRepository` each shrink materially and delegate to smaller collaborators.
- Plugin and hook dependencies are explicit in execution-time structures rather than discovered repeatedly through runtime lookup.

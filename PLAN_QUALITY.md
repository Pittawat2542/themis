# Code Quality Review — Principal Engineer Assessment

> **Reviewer**: Principal Research Engineer
> **Date**: 2026-02-14
> **Scope**: Full codebase (`themis/` — ~30,800 LOC, 90+ source files)
> **Verdict**: **Conditionally Approved** — Architecture is solid; tactical cleanup needed.

---

## 1. Core Data Model (`themis/core/`)

**Strengths**: Frozen dataclasses, `__all__` exports, `Generic[T]` on `Reference`, clear domain separation.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 1.1 | Medium | **Legacy type aliases**: Uses `Dict[str, Any]`, `List[...]` (9 occurrences) instead of modern `dict`, `list`. Rest of codebase already uses built-in generics. | `entities.py` |
| 1.2 | Low | **`SamplingConfig` duplication**: Identical class exists in both `core/entities.py` (frozen dataclass) and `config/schema.py` (mutable dataclass with defaults). Divergence risk. | `entities.py:15`, `schema.py:25` |
| 1.3 | Low | **Dead type variables**: `PredictionType`, `ReferenceType`, `ExtractionType` in `types.py` are declared but never used anywhere in the codebase. | `types.py:72-74` |

**Action Plan**:
- [ ] Replace `Dict[str, Any]` → `dict[str, Any]` and `List[...]` → `list[...]` in `entities.py`
- [ ] Evaluate whether `config.schema.SamplingConfig` should reference or extend `core.entities.SamplingConfig`
- [ ] Remove unused `PredictionType`, `ReferenceType`, `ExtractionType` from `types.py`

---

## 2. Configuration (`themis/config/`)

**Strengths**: Clean Hydra/OmegaConf integration, `from_file`/`from_dict`/`to_file` factory methods, sensible defaults.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 2.1 | Low | **Inline `# New field` comment** on `StorageConfig.default_path` suggests an unfinished migration. | `schema.py:54` |
| 2.2 | Low | Default `model_identifier = "fake-math-llm"` may confuse new users who expect no default or an explicit placeholder. | `schema.py:34` |

**Action Plan**:
- [ ] Remove stale `# New field` comments in `schema.py`
- [ ] Consider `model_identifier: str = ""` with validation, or remove the default entirely

---

## 3. Public API (`themis/api.py`)

**Strengths**: Ergonomic `evaluate()` one-liner with auto-detection, solid docstrings, `__all__` export.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 3.1 | High | **God function**: `evaluate()` is ~240 lines. Mixes provider detection, dataset resolution, metric resolution, prompt construction, and orchestration in a single function. | `api.py:122-363` |
| 3.2 | Medium | **`_resolve_metrics()` is ~110 lines** with a deeply nested inner function `_normalize_metric_name`. Should be extracted. | `api.py:452-559` |
| 3.3 | Low | `_PROVIDER_OPTION_KEYS` and `_ALLOWED_EXTRA_OPTIONS` are module-level tuples/sets that could be consolidated into a single constant. | `api.py:62-75` |

**Action Plan**:
- [ ] Extract `evaluate()` into smaller helpers: `_build_experiment_spec()`, `_resolve_provider()`, `_build_pipeline()`
- [ ] Move `_resolve_metrics()` and `_normalize_metric_name()` to a dedicated `themis/evaluation/metric_resolver.py`
- [ ] Consolidate provider option constants

---

## 4. Generation Layer (`themis/generation/`)

**Strengths**: Clean Strategy pattern (`GenerationStrategy`), Cartesian/Filtered expansion strategies, proper retry with exponential backoff, `_NON_RETRYABLE_ERROR_MARKERS` for smart retry decisions.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 4.1 | Medium | **`GenerationRunner.run()` mixes** sequential and parallel paths with complex conditional branching. The parallel path relies on an inner function `_submit_until_full`. | `runner.py:66-152` |
| 4.2 | Medium | **`plan.py` has duplicate logic**: `GenerationPlan` and `CartesianExpansionStrategy` implement nearly identical `_build_metadata` and `_build_reference` methods. | `plan.py:80-114`, `plan.py:210-247` |
| 4.3 | Low | **`types.py`** in generation is only 338 bytes — may not justify its own module. | `generation/types.py` |

**Action Plan**:
- [ ] In `runner.py`, extract parallel execution into a dedicated `_ParallelExecutor` or use a cleaner strategy dispatch
- [ ] In `plan.py`, have `GenerationPlan.expand()` delegate to `CartesianExpansionStrategy` internally to eliminate duplication
- [ ] Merge `generation/types.py` into `generation/__init__.py` if it remains trivial

---

## 5. Evaluation Layer (`themis/evaluation/`)

**Strengths**: Protocol-based `EvaluationPipelineContract`, composable pipeline with builder pattern, clear separation between extractors/metrics/strategies/statistics. Excellent `__all__` re-exports.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 5.1 | Medium | **`conditional.py` (410 lines)**: Conditional evaluation logic is complex and may benefit from decomposition. | `evaluation/conditional.py` |
| 5.2 | Low | **`metric_pipeline.py`** is only 329 bytes — trivial wrapper. Consider inlining or documenting why it exists separately. | `evaluation/metric_pipeline.py` |

**Action Plan**:
- [ ] Review `conditional.py` for extraction opportunities (e.g., condition evaluator vs. pipeline builder)
- [ ] Document or inline `metric_pipeline.py`

---

## 6. Experiment Orchestration (`themis/experiment/`)

**Strengths**: Clean manager separation (`CacheManager`, `IntegrationManager`), reproducibility manifests, streaming evaluation with batch flushing.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 6.1 | **Critical** | **`orchestrator.run()` is ~430 lines** with 3 nested inner functions (`_accumulate_evaluation_record`, `_flush_eval_batch`, `_iter_pending_tasks`). This is the single largest method in the codebase and violates SRP. | `orchestrator.py:115-545` |
| 6.2 | **Critical** | **`experiment/storage.py` is 1,742 lines** — the largest file in the codebase. Contains `ExperimentStorage` with 40+ methods spanning locking, SQLite, caching, checksums, retention, cleanup, and export. | `experiment/storage.py` |
| 6.3 | Medium | **`IntegrationManager.finalize()`** contains only `pass` statements — dead code that gives a false sense of cleanup. | `integration_manager.py:99-111` |
| 6.4 | Medium | **No `wandb.finish()` call** in `IntegrationManager` or `WandbTracker`. WandB runs may not be properly finalized. | `integration_manager.py`, `wandb.py` |

**Action Plan**:
- [ ] **Priority 1**: Decompose `orchestrator.run()` into phases: `_initialize_run()`, `_run_generation_loop()`, `_finalize_evaluation()`, `_build_report()`
- [ ] **Priority 1**: Split `experiment/storage.py` into sub-modules: `storage/locking.py`, `storage/cache.py`, `storage/sqlite_metadata.py`, `storage/integrity.py`
- [ ] Call `wandb.finish()` in `WandbTracker.finalize()` and wire it through `IntegrationManager.finalize()`
- [ ] Remove no-op `pass` blocks in `finalize()` or add proper cleanup

---

## 7. CLI (`themis/cli/`)

**Strengths**: Cyclopts-based, focused commands (`demo`, `eval`, `compare`, `share`, `serve`, `list`, `clean`), consistent `--verbose`/`--json-logs` flags.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 7.1 | High | **`main.py` is 854 lines** containing all 7 commands plus helper functions. A CLI  "God module." | `cli/main.py` |
| 7.2 | Medium | **`_generate_comparison_html()`** (57 lines) and **`_generate_comparison_markdown()`** (24 lines) are presentation logic embedded in the CLI module. | `cli/main.py:742-826` |
| 7.3 | Medium | **Duplicate field detection logic**: `_PROMPT_FIELD_CANDIDATES`, `_REFERENCE_FIELD_CANDIDATES`, `_ID_FIELD_CANDIDATES` also exist in `api.py`. | `cli/main.py:642-654`, `api.py` |

**Action Plan**:
- [ ] Extract each command into `cli/commands/{command}.py` (some already have this pattern in `cli/commands/comparison.py`)
- [ ] Move HTML/Markdown generation to `experiment/export.py` or a shared `utils/report_formatters.py`
- [ ] Consolidate field detection constants into a single `themis/datasets/field_detection.py`

---

## 8. Storage & Persistence (`themis/storage/`, `themis/backends/`)

**Strengths**: Atomic writes, file locking (cross-platform), SQLite metadata, checksum integrity, retention policies.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 8.1 | High | **Size**: As noted in §6.2, `experiment/storage.py` is 1,742 lines — needs decomposition. | `experiment/storage.py` |
| 8.2 | Medium | **Two `StorageConfig` classes**: `config/schema.py:StorageConfig` (for Hydra) and `experiment/storage.py:StorageConfig` (for runtime). Naming collision. | Both files |

**Action Plan**:
- [ ] Rename `experiment/storage.StorageConfig` → `StorageBehaviorConfig` or `StorageRuntimeConfig`
- [ ] Split `experiment/storage.py` as described in §6.2

---

## 9. Integrations (`themis/integrations/`)

**Strengths**: Optional dependency pattern (try/except import), config-gated initialization, clean delegation through `IntegrationManager`.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 9.1 | Medium | **`log_results()` assumes `record.responses`** attribute on `GenerationRecord`, which doesn't exist in `entities.py` (it has `output: ModelOutput | None`). This will crash at runtime. | `wandb.py:80` |
| 9.2 | Low | **No `wandb.finish()`** as noted in §6.4. | `wandb.py` |

**Action Plan**:
- [ ] **Priority 1**: Fix `wandb.py:log_results()` to use `record.output.text` instead of `record.responses`
- [ ] Add `finalize()` method with `wandb.finish()` call

---

## 10. Testing

**Strengths**: Structured test directories mirroring source, good config/dataset/spec coverage, doc-consistency tests (`test_cli_docs_consistency.py`).

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 10.1 | **Critical** | **No tests for `GenerationRunner`** — the core execution engine has zero unit tests. | `tests/generation/` missing |
| 10.2 | **Critical** | **No tests for `ExperimentOrchestrator`** — the most complex class in the project is untested. | `tests/experiment/` missing orchestrator tests |
| 10.3 | High | **No tests for `ExperimentStorage`** — 1,742 lines of storage logic with locking, SQLite, and checksums has no unit tests. | `tests/experiment/` missing storage tests |
| 10.4 | High | **No tests for `EvaluationPipeline`** — standard pipeline execution path is untested. | `tests/evaluation/` missing |
| 10.5 | Medium | **No integration tests** — `tests/integration/__init__.py` exists but is empty. | `tests/integration/` |
| 10.6 | Low | **Test file in source tree**: `tests/test_package_metadata.py` is fine, but `tests/factories.py` is a support file without the `test_` prefix, which could confuse test discovery. | `tests/factories.py` |

**Action Plan**:
- [ ] **Priority 1**: Add unit tests for `GenerationRunner` (mock provider, test retry logic, parallel execution)
- [ ] **Priority 1**: Add unit tests for `ExperimentOrchestrator` (mock runner/pipeline, test batch flow)
- [ ] **Priority 2**: Add unit tests for `ExperimentStorage` (temp directory, test locking, atomic writes)
- [ ] **Priority 2**: Add tests for `EvaluationPipeline` (test evaluate(), metric aggregation)
- [ ] **Priority 3**: Add at least one end-to-end integration test with the fake provider
- [ ] Rename `tests/factories.py` → `tests/conftest.py` or `tests/helpers.py`

---

## 11. Type Safety & Consistency

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 11.1 | Medium | **Mixed type annotation styles**: `core/entities.py` uses `Dict[str, Any]` / `List[...]` while the rest of the project uses `dict[str, Any]` / `list[...]` (enabled by `from __future__ import annotations`). | `core/entities.py` |
| 11.2 | Low | **`pyproject.toml` missing** `ruff` and `mypy` configuration sections. Linting/type-checking config is implicit. | `pyproject.toml` |
| 11.3 | Low | **No `mypy` strict mode** — `mypy` is listed as a dev dependency but no `mypy.ini` or `pyproject.toml [tool.mypy]` configuration exists. | Project root |

**Action Plan**:
- [ ] Modernize `entities.py` type annotations as noted in §1.1
- [ ] Add `[tool.ruff]` and `[tool.mypy]` sections to `pyproject.toml`
- [ ] Consider enabling `mypy --strict` incrementally with per-module overrides

---

## 12. Documentation & Developer Experience

**Strengths**: `ARCHITECTURE.md` with Mermaid diagrams, `CONTRIBUTING.md`, API/CLI guides, MkDocs setup, `py.typed` marker.

**Issues**:

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 12.1 | Low | `observability.md` has a typo: "guid" → "guide" in the opening paragraph. | `docs/guides/observability.md:3` |
| 12.2 | Low | **No `CHANGELOG.md` entries since initial release** (`CHANGELOG.md` is 605 bytes). | `docs/CHANGELOG.md` |

**Action Plan**:
- [ ] Fix typo in `observability.md`
- [ ] Establish a CHANGELOG update process (ideally automated from conventional commits)

---

## Priority Summary

| Priority | Items | Effort |
|----------|-------|--------|
| **P0 — Critical** | §6.1 Decompose `orchestrator.run()`, §6.2 Split `storage.py`, §9.1 Fix `wandb.log_results()`, §10.1-10.2 Add runner/orchestrator tests | Large |
| **P1 — High** | §3.1 Refactor `api.evaluate()`, §7.1 Split CLI, §10.3-10.4 Add storage/pipeline tests | Large |
| **P2 — Medium** | §1.1 Type annotations, §4.2 Plan duplication, §6.3-6.4 Finalize cleanup, §8.2 Config naming | Medium |
| **P3 — Low** | §1.3 Dead types, §2.1-2.2 Config cleanup, §11.2-11.3 Tool config, §12.1-12.2 Docs | Small |

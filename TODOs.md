# vNext Implementation TODOs

This checklist tracks the clean-architecture vNext work (no backward compatibility).

## Phase 0: Alignment
- [x] Confirm API naming (`run` vs `evaluate`) and model key format (`provider:model_id`)
- [x] Decide primary evaluation pipeline (`MetricPipeline` vs composable)

## Phase 1: Spec Models
- [x] Create `themis/specs/experiment.py` (`ExperimentSpec`)
- [x] Create `themis/specs/execution.py` (`ExecutionSpec`)
- [x] Create `themis/specs/storage.py` (`StorageSpec`)
- [x] Add `themis/specs/__init__.py` exports
- [x] Add spec tests in `tests/specs/`

## Phase 2: Canonical Evaluation Contract
- [x] Define `EvaluationPipeline` interface with `evaluation_fingerprint()`
- [x] Enforce `EvaluationRecord.scores` as `list[MetricScore]`
- [x] Add contract tests in `tests/evaluation/test_contract.py`

## Phase 3: ExperimentSession
- [x] Implement `themis/session.py` orchestration
- [x] Move logic from `themis/experiment/orchestrator.py`
- [x] Add `tests/session/test_session.py`

## Phase 4: Execution Integration
- [x] Wire `ExecutionBackend` into `GenerationRunner`
- [x] Add `tests/execution/test_runner_backend.py`
- [x] Update call sites to use `ExecutionSpec.backend`

## Phase 5: Storage Integration
- [x] Create `themis/storage/` package with `StorageBackend` + adapters
- [x] Move `ExperimentStorage` to `themis/storage/experiment_storage.py`
- [x] Update session to use `StorageSpec.backend`
- [x] Add `tests/storage/`

## Phase 6: Provider Routing
- [x] Standardize model keys to `provider:model_id`
- [x] Update `ModelSpec` to store canonical `model_key`
- [x] Update router dispatch to use `model_key`
- [x] Add collision tests in `tests/generation/test_router.py`

## Phase 7: Evaluation Pipeline Simplification
- [x] Pick primary pipeline (`MetricPipeline`)
- [x] Implement in `themis/evaluation/metric_pipeline.py`
- [x] If composable stays, make it implement `EvaluationPipeline`
- [x] Remove adapters and duplicate re-exports
- [x] Update evaluation tests to use new pipeline

## Phase 8: API Surface
- [x] Replace `themis/api.py` with thin wrapper over session
- [x] Decide whether `evaluate` remains as alias
- [x] Remove `themis/experiment/builder.py` and task helpers
- [x] Update `themis/__init__.py` exports

## Phase 9: CLI Rewrite
- [x] Update `themis/cli/main.py` to parse `ExperimentSpec`
- [x] Remove CLI commands that bypass spec model
- [x] Update CLI docs in `docs/guides/cli.md`

## Phase 10: Comparison/Server Alignment
- [x] Update `themis/comparison/` to consume canonical records
- [x] Update `themis/server/app.py` to use canonical entities
- [x] Add/adjust tests in `tests/comparison/` and `tests/server/`

## Phase 11: Docs Restructure
- [x] Replace nav in `mkdocs.yml` to a single path
- [x] Add `docs/ARCHITECTURE_VNEXT.md` to reference section
- [x] Update examples to use `ExperimentSpec` + `run`

## Phase 12: Cleanup & Release
- [ ] Delete legacy modules and tests
- [ ] Add breaking changes to `CHANGELOG.md`
- [ ] Bump major version in `pyproject.toml`
- [x] Run `uv run pytest`
- [ ] Tag release

# vNext Implementation TODOs

This checklist tracks the clean-architecture vNext work (no backward compatibility).

## Phase 0: Alignment
- [ ] Confirm API naming (`run` vs `evaluate`) and model key format (`provider:model_id`)
- [ ] Decide primary evaluation pipeline (`MetricPipeline` vs composable)

## Phase 1: Spec Models
- [ ] Create `themis/specs/experiment.py` (`ExperimentSpec`)
- [ ] Create `themis/specs/execution.py` (`ExecutionSpec`)
- [ ] Create `themis/specs/storage.py` (`StorageSpec`)
- [ ] Add `themis/specs/__init__.py` exports
- [ ] Add spec tests in `tests/specs/`

## Phase 2: Canonical Evaluation Contract
- [ ] Define `EvaluationPipeline` interface with `evaluation_fingerprint()`
- [ ] Enforce `EvaluationRecord.scores` as `list[MetricScore]`
- [ ] Add contract tests in `tests/evaluation/test_contract.py`

## Phase 3: ExperimentSession
- [ ] Implement `themis/session.py` orchestration
- [ ] Move logic from `themis/experiment/orchestrator.py`
- [ ] Add `tests/session/test_session.py`

## Phase 4: Execution Integration
- [ ] Wire `ExecutionBackend` into `GenerationRunner`
- [ ] Add `tests/execution/test_runner_backend.py`
- [ ] Update call sites to use `ExecutionSpec.backend`

## Phase 5: Storage Integration
- [ ] Create `themis/storage/` package with `StorageBackend` + adapters
- [ ] Move `ExperimentStorage` to `themis/storage/experiment_storage.py`
- [ ] Update session to use `StorageSpec.backend`
- [ ] Add `tests/storage/`

## Phase 6: Provider Routing
- [ ] Standardize model keys to `provider:model_id`
- [ ] Update `ModelSpec` to store canonical `model_key`
- [ ] Update router dispatch to use `model_key`
- [ ] Add collision tests in `tests/generation/test_router.py`

## Phase 7: Evaluation Pipeline Simplification
- [ ] Pick primary pipeline (`MetricPipeline`)
- [ ] Implement in `themis/evaluation/metric_pipeline.py`
- [ ] If composable stays, make it implement `EvaluationPipeline`
- [ ] Remove adapters and duplicate re-exports
- [ ] Update evaluation tests to use new pipeline

## Phase 8: API Surface
- [ ] Replace `themis/api.py` with thin wrapper over session
- [ ] Decide whether `evaluate` remains as alias
- [ ] Remove `themis/experiment/builder.py` and task helpers
- [ ] Update `themis/__init__.py` exports

## Phase 9: CLI Rewrite
- [ ] Update `themis/cli/main.py` to parse `ExperimentSpec`
- [ ] Remove CLI commands that bypass spec model
- [ ] Update CLI docs in `docs/guides/cli.md`

## Phase 10: Comparison/Server Alignment
- [ ] Update `themis/comparison/` to consume canonical records
- [ ] Update `themis/server/app.py` to use canonical entities
- [ ] Add/adjust tests in `tests/comparison/` and `tests/server/`

## Phase 11: Docs Restructure
- [ ] Replace nav in `mkdocs.yml` to a single path
- [ ] Add `docs/ARCHITECTURE_VNEXT.md` to reference section
- [ ] Update examples to use `ExperimentSpec` + `run`

## Phase 12: Cleanup & Release
- [ ] Delete legacy modules and tests
- [ ] Add breaking changes to `CHANGELOG.md`
- [ ] Bump major version in `pyproject.toml`
- [ ] Run `uv run pytest`
- [ ] Tag release

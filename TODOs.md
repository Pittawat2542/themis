# TODOs (TDD-First, No Backward Compatibility)

This plan intentionally favors clean architecture and research correctness over compatibility.  
Every item follows red -> green -> refactor.

## Engineering Rules

- Write failing tests first for all behavior changes.
- Remove deprecated/legacy paths instead of preserving them.
- Prefer explicit errors over silent fallbacks.
- Keep statistical methods scientifically defensible by default.
- Require reproducibility metadata in all run artifacts.

## P0 - Correctness and Research Validity

### 1) Rebuild `evaluate()` wiring as a strict façade

- [ ] Remove placeholder/legacy args that are not implemented (`distributed`, etc.) or implement them fully.
- [ ] Ensure `evaluate()` passes all execution/runtime options into session/orchestrator (provider kwargs, `on_result`, sampling controls).
- [ ] Remove post-hoc fake `num_samples` behavior; execute true multi-sampling in generation.
- [ ] Enforce explicit argument validation with actionable error messages.

TDD:
- [ ] Add API integration tests proving provider kwargs reach provider construction.
- [ ] Add test proving `on_result` is called once per produced record.
- [ ] Add test proving `num_samples=n` generates `n` real attempts (not duplicated references).
- [ ] Add test asserting unsupported args fail fast.

### 2) Make statistical comparison rigorous by default

- [ ] Unify to a single statistics module (`themis/evaluation/statistics`) and remove duplicate comparison-specific implementation.
- [ ] Require sample alignment by `sample_id` for paired tests.
- [ ] Replace rough p-value approximations with exact/scipy-backed implementations; define deterministic fallback policy.
- [ ] Add multiple-comparison correction policy as part of report generation defaults.

TDD:
- [ ] Add paired-test alignment tests (mismatched IDs must fail with clear message).
- [ ] Add golden tests for t-test/permutation/bootstrap outputs against reference implementations.
- [ ] Add tests verifying correction method behavior (Holm-Bonferroni) across multiple metrics.

### 3) Secure code execution metrics

- [ ] Replace inline `exec` execution with isolated worker process runtime.
- [ ] Enforce hard timeout and memory limits per test case.
- [ ] Restrict imports/syscalls and explicitly define allowed builtins.
- [ ] Record execution status taxonomy (timeout, runtime error, forbidden operation, assertion fail).

TDD:
- [ ] Add tests for timeout enforcement.
- [ ] Add tests for memory-limit enforcement.
- [ ] Add tests that filesystem/network/process-spawn operations are blocked.
- [ ] Add tests verifying deterministic result schema for pass/fail/error states.

## P1 - API and Storage Quality

### 4) Fix storage backend contracts and simplify adapter model

- [ ] Fix `LocalFileStorageBackend.save_evaluation_record` signature usage.
- [ ] Define one canonical storage protocol and make all adapters conform strictly.
- [ ] Remove partial/unimplemented adapter methods or implement fully.
- [ ] Add clear lifecycle semantics (`start_run`, `append`, `complete_run`, `fail_run`).

TDD:
- [ ] Add contract tests shared across all storage backends.
- [ ] Add adapter integration tests for generation + evaluation + report persistence.
- [ ] Add tests for concurrent access and reentrant locking invariants.

### 5) Reproducibility manifest as first-class artifact

- [x] Persist run manifest with: model/provider options, seeds, metric configs, extractor config, package versions, git commit hash.
- [x] Hash manifest and include in report metadata and cache keys.
- [x] Require manifest for run start; fail if incomplete.

TDD:
- [x] Add tests asserting manifest presence and required fields.
- [x] Add tests asserting cache invalidation when manifest-critical fields change.
- [x] Add tests asserting deterministic manifest hash generation.

## P2 - Performance and Scalability

### 6) Improve execution throughput and memory behavior

- [ ] Use completion-order result handling (`as_completed`) in threaded generation path.
- [ ] Stream records/evaluations in chunks instead of full in-memory materialization.
- [ ] Replace directory scanning for run lookup with indexed lookup in storage metadata DB.
- [ ] Add bounded-memory mode for large runs.

TDD:
- [ ] Add performance regression tests for head-of-line blocking scenarios.
- [ ] Add large-run tests asserting memory does not scale linearly with total samples.
- [ ] Add tests validating indexed run lookup behavior under many runs.

## P3 - UX, Docs, and Test Suite Hardening

### 7) Remove ambiguous behavior and outdated docs

- [ ] Align API docs with actual metric names and accepted inputs.
- [ ] Remove claims for unimplemented features.
- [ ] Make error messages reference exact accepted alternatives.

TDD:
- [ ] Add docs lint/check tests for stale feature flags and API signatures.
- [ ] Add tests validating metric-name normalization and error output.

### 8) Raise test quality and coverage for critical paths

- [ ] Replace placeholder `pass` tests with real behavioral assertions.
- [ ] Set minimum coverage thresholds per critical module (API, orchestration, statistics, execution safety, storage).
- [ ] Add end-to-end reproducibility test using fixed seed and manifest.

TDD:
- [ ] Add failing tests for each current placeholder in API tests.
- [ ] Add CI gate for module-level coverage thresholds.

## Proposed Execution Order

- [ ] Sprint 1: P0.1, P0.2
- [ ] Sprint 2: P0.3, P1.4
- [ ] Sprint 3: P1.5, P2.6
- [ ] Sprint 4: P3.7, P3.8

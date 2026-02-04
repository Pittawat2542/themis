# Architecture Overview

This document describes the current architecture and API direction for Themis. The goal is a single, explicit mental model for research workflows, with fewer overlapping entry points and stronger contracts across generation, evaluation, execution, and storage.

## Goals
- Single entry point with explicit configuration.
- One canonical spec model for both Python and CLI.
- Clean separation of concerns and formal interfaces.
- Storage, comparison, and server layers aligned to a single data model.
- Documentation that tells one story.

## Non-Goals
- Preserving existing function signatures or module paths.
- Supporting multiple overlapping “ways” to do the same task.
- Shipping all future integrations immediately.

## Current Pain Points
- Multiple entry points and partial overlaps (`evaluate`, CLI, builder, task helpers).
- Backends documented but not fully wired into the primary API.
- Evaluation records represented inconsistently across modules.
- Orchestrator relies on private fields and implicit pipeline shape.
- Ambiguous model routing when identifiers collide.

## Proposed User API
### Python
```python
from themis import run
from themis.specs import ExperimentSpec, ExecutionSpec, StorageSpec
from themis.evaluation import MetricPipeline

spec = ExperimentSpec(
    dataset=MyDataset(),
    prompt="Solve: {question}",
    model="litellm:gpt-4",
    sampling={"temperature": 0.0, "max_tokens": 512},
    pipeline=MetricPipeline(metrics=["exact_match"]),
)

report = run(
    spec,
    execution=ExecutionSpec(workers=8),
    storage=StorageSpec(path=".cache/experiments"),
)
```

### CLI
```bash
themis run --config experiment.yaml
```

The CLI becomes a thin wrapper around the same spec model.

## Canonical Spec Model
### ExperimentSpec
- `dataset`: `DatasetAdapter` or dataset reference.
- `prompt`: string or prompt template.
- `model`: explicit model key (`provider:model`) or `ModelSpec`.
- `sampling`: sampling configuration.
- `pipeline`: evaluation pipeline.
- `run_id`: explicit or auto-generated.

### ExecutionSpec
- `backend`: `ExecutionBackend`
- `workers`: int
- `retries`: retry policy

### StorageSpec
- `backend`: `StorageBackend`
- `path`: optional if backend provided
- `cache`: enable/disable caching

## Interfaces and Contracts
All core contracts are explicit and enforced.

### Required Interfaces
- `DatasetAdapter`
- `ModelProvider`
- `Metric`
- `EvaluationPipeline`
- `ExecutionBackend`
- `StorageBackend`

### EvaluationPipeline Contract
- `evaluate(records) -> EvaluationReport`
- `evaluation_fingerprint() -> dict`

### Model Key Contract
Models are addressed by a stable key.
- `model = "provider:model_id"`
- `ModelSpec(provider="litellm", identifier="gpt-4", alias="gpt4")`

## One Evaluation Model
All evaluation records use the same shape:
- `EvaluationRecord.scores: list[MetricScore]`
- Storage, comparison, server, and export use this directly.

## Module Layout (Proposed)
```
themis/
  api.py                -> thin wrapper around run()
  session.py            -> ExperimentSession (orchestrator)
  specs/
    experiment.py
    execution.py
    storage.py
  generation/
  evaluation/
    pipeline.py         -> EvaluationPipeline interface + MetricPipeline
    composable.py       -> optional advanced pipeline
  execution/            -> backends
  storage/              -> backends + ExperimentStorage adapter
  providers/
  datasets/
  comparison/
  server/
```

## Resolving Previously Identified Issues
1. Comparison and server shape mismatches.
   - Enforced by the canonical `EvaluationRecord` format.
2. `num_samples` unused.
   - Becomes `sampling.samples_per_prompt` in `ExperimentSpec`.
3. Backends not wired.
   - `ExecutionSpec.backend` and `StorageSpec.backend` are mandatory integration points.
4. Orchestrator private field access.
   - Replaced with `EvaluationPipeline.evaluation_fingerprint()`.
5. Composable pipeline incompatibility.
   - If retained, it must implement the `EvaluationPipeline` interface.
6. Provider routing collisions.
   - Model keys are `provider:model_id` and routing uses that directly.
7. Metric naming inconsistencies.
   - Metrics are resolved by a single registry and exposed in docs as canonical keys.

## Documentation Restructure
Single narrative, no forks:
1. Quickstart
2. Core Concepts
3. Configuration
4. Extensions
5. CLI Reference
6. API Reference

## Migration Strategy (No Backward Compatibility)
1. New API and spec model shipped in parallel branch.
2. CLI updated to use spec model.
3. Remove old builder/task helpers.
4. Remove implicit config logic in `evaluate`.
5. Update docs to reflect a single approach.

## Suggested Phasing
### Phase 1: Core Model
- Implement spec model and ExperimentSession.
- Define EvaluationPipeline contract.

### Phase 2: Integration
- Wire execution and storage backends into run().
- Align server and comparison with EvaluationRecord shape.

### Phase 3: Cleanup
- Remove legacy modules.
- Collapse docs into the new structure.

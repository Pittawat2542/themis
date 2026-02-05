# Architecture Overview

Themis has two public evaluation surfaces built on the same orchestration core:

- `themis.evaluate(...)`: high-level API for benchmark and dataset workflows.
- `ExperimentSession().run(spec, ...)`: explicit spec/session API for advanced control.

## Layered Design

```text
themis.evaluate(...) / CLI commands
            |
       ExperimentSession
            |
   ExperimentSpec / ExecutionSpec / StorageSpec
            |
GenerationPlan -> GenerationRunner -> EvaluationPipelineContract
            |
  ExperimentStorage / comparison / server / export
```

## Primary Components

### Public API Layer
- `themis.api.evaluate`: convenience wrapper that resolves presets, metrics, and defaults.
- `themis.session.ExperimentSession`: explicit orchestrator entrypoint.
- `themis.cli.main`: `demo`, `eval`, `compare`, `share`, `serve`, `list`, `clean`.

### Spec Layer
- `themis.specs.ExperimentSpec`: dataset, prompt, model, sampling, pipeline, run id.
- `themis.specs.ExecutionSpec`: worker/retry policy and optional execution backend.
- `themis.specs.StorageSpec`: storage path/backend and caching toggle.

### Generation + Evaluation Layer
- `themis.generation.GenerationPlan`: expands dataset into tasks.
- `themis.generation.GenerationRunner`: executes tasks against providers.
- `themis.evaluation.EvaluationPipelineContract`: enforced evaluation interface.
- `themis.evaluation.EvaluationPipeline` / `MetricPipeline`: standard metric execution.

### Persistence + Analysis Layer
- `themis.storage.ExperimentStorage`: filesystem-backed run storage.
- `themis.comparison.compare_runs`: statistical run-to-run comparison.
- `themis.experiment.export`: CSV/JSON/HTML export utilities.
- `themis.server.create_app`: REST/WebSocket API over run artifacts.

## Data Contracts

- Generation outputs are represented by `GenerationRecord`.
- Evaluation outputs are represented by `EvaluationRecord` where `scores` is `list[MetricScore]`.
- Run-level output is `ExperimentReport`.

This canonical shape is used consistently by storage, comparison, exports, and server endpoints.

## Model Routing Contract

- Recommended model key format: `provider:model_id` (for example `litellm:gpt-4`).
- High-level `evaluate(...)` also accepts provider-auto-detected model strings (for example `gpt-4`).

## Extension Points

- Metrics: `themis.register_metric(name, metric_cls)`
- Datasets: `themis.register_dataset(name, factory)`
- Providers: `themis.register_provider(name, factory)`
- Benchmarks: `themis.register_benchmark(preset)`
- Backends: custom `ExecutionBackend` and `StorageBackend`

## Current Trade-offs

- Distributed/custom execution requires an explicit `execution_backend` implementation.
- Custom dataset files are not yet supported in `themis eval ...` CLI; use Python API for dataset objects.
- Custom storage backends must be ExperimentStorage-compatible for full session integration.

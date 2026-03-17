# Advanced Workflows

## Use Project Files For Stable Shared Policy

Keep storage and execution policy in TOML or JSON, then keep benchmark
semantics in Python.

```python
orchestrator = Orchestrator.from_project_file(
    "project.toml",
    registry=registry,
    dataset_provider=dataset_provider,
)
```

## Evolve A Benchmark Incrementally

Common cases:

- add a model
- add a prompt variant
- add a parse pipeline
- add a score overlay
- change the dataset query

Run the new benchmark against the same project storage root so completed work is
reused where hashes still match.

## Hand Off Generation Or Evaluation

```python
bundle = orchestrator.export_evaluation_bundle(benchmark)
result = orchestrator.import_evaluation_results(bundle, external_trial_records)
```

## Scale Execution Deliberately

Themis owns orchestration state. Your engine still owns provider-specific rate
limits, retries, and batching details.

## Distinguish Progress Logging From Telemetry

Use `themis.progress` first for operator-facing logging. Add external telemetry
only when the user needs event-level observability or external sinks.

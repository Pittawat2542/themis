# Advanced Workflows

## Use Project Files For Stable Shared Policy

Keep shared infrastructure policy in TOML or JSON and keep the experiment matrix
in Python.

Example `project.toml`:

```toml
project_name = "offline-evals"
researcher_id = "team-research"
global_seed = 13

[storage]
root_dir = ".cache/themis/offline-evals"
backend = "sqlite_blob"
store_item_payloads = true
compression = "zstd"

[execution_policy]
max_retries = 2
retry_backoff_factor = 1.5
circuit_breaker_threshold = 4

[execution_backend]
kind = "local"
```

Load it with:

```python
orchestrator = Orchestrator.from_project_file(
    "project.toml",
    registry=registry,
    dataset_loader=dataset_loader,
)
```

Use project files for storage, retry, and backend policy. Keep models, prompts,
tasks, transforms, and evaluations in `ExperimentSpec`.

## Evolve An Existing Experiment Incrementally

Themis hashes the matrix into deterministic stage identities, so only new work
stays pending.

Common cases:

- add a model or prompt: new trial hashes
- add inference params: new trial hashes for new combinations
- add a metric: new evaluation overlay, existing generation can stay
- add a transform: new transform and evaluation overlays
- change the dataset slice: deterministic change to the planned trial set

Use `model_copy(update=...)` and confirm the delta first:

```python
diff = orchestrator.diff_specs(old_experiment, new_experiment)
print(diff.changed_experiment_fields)
print(len(diff.added_trial_hashes))
print(len(diff.added_evaluation_hashes))
```

Use the snippets in this file as the runnable pattern. Do not assume the user
has a local `examples/` directory.

## Hand Off Generation Or Evaluation To External Systems

Export only missing work, run it elsewhere, then import compatible records back.

Generation handoff:

```python
bundle = orchestrator.export_generation_bundle(experiment)
print(bundle.manifest.run_id)
print(len(bundle.items))
```

Evaluation handoff:

```python
bundle = orchestrator.export_evaluation_bundle(experiment)
print(bundle.manifest.run_id)
print(len(bundle.items))
```

After import, normal reporting and timeline APIs continue to work. Use the
snippets in this file as the import shape when local examples are unavailable.

## Scale Execution Deliberately

Themis manages orchestration state, not provider-specific throughput tuning.

Use:

- `ExecutionPolicySpec.max_in_flight_work_items` to bound orchestrated work
- `execution_backend.kind="local"` for in-process runs
- `execution_backend.kind="worker_pool"` or batch workflows when work should be
  picked up elsewhere

The engine implementation still owns provider rate limiting, retries, and
batch-adapter details.

## Add Telemetry Only When The User Needs It

Use a `TelemetryBus` for in-process events. Add Langfuse or other callbacks only
when the user wants observability.

```python
from themis.telemetry import TelemetryBus

bus = TelemetryBus()
bus.subscribe(lambda event: print(event.name, event.payload))
```

Use the `telemetry` extra when callbacks depend on external SDKs.

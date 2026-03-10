# Author Project Files

`Orchestrator.from_project_file()` accepts TOML or JSON.

## Recommended Fields

Keep these values stable in your project file:

- `project_name`
- `researcher_id`
- `global_seed`
- `storage.root_dir`
- `execution_policy.*`

## TOML Example

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
```

## Usage

```python
orchestrator = Orchestrator.from_project_file(
    "project.toml",
    registry=registry,
    dataset_loader=dataset_loader,
)
```

!!! warning
    Project files define shared policy, not the experiment matrix. Keep models,
    tasks, prompts, and parameter sweeps in `ExperimentSpec`.

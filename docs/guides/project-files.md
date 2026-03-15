# Author Project Files

`Orchestrator.from_project_file()` accepts TOML or JSON.

## Recommended Fields

Keep these values stable in your project file:

- `project_name`
- `researcher_id`
- `global_seed`
- `storage.root_dir` for `sqlite_blob`, or `storage.database_url` plus `storage.blob_root_dir` for `postgres_blob`
- `execution_policy.*`
- `execution_backend.*` when you want worker-pool or batch orchestration

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

[execution_backend]
kind = "local"
```

For Postgres-backed runs:

```toml
[storage]
backend = "postgres_blob"
database_url = "postgresql://localhost:5432/themis"
blob_root_dir = ".cache/themis/offline-evals/blobs"
store_item_payloads = true
compression = "zstd"

[execution_backend]
kind = "worker_pool"
lease_ttl_seconds = 180
poll_interval_seconds = 5
worker_tags = ["gpu:a100"]
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

When you call `Orchestrator.plan()` or export an external work bundle, Themis
persists the exact `ProjectSpec` snapshot from the file into the run manifest so
the run can be reproduced or diffed later.

That same backend selection also drives `submit()` and `resume()`: local runs
execute immediately in-process, while `worker_pool` and `batch` runs persist a
handle that external workers or imports can complete later.

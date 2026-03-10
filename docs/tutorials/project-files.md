# Load a Project File

In this tutorial you will take an inline `ProjectSpec` and move it into a
reusable `project.toml` file.

`Orchestrator.from_project_file()` supports `.toml` and `.json`.

## Before You Start

Start from the hello-world script in the previous tutorial or from
`examples/02_project_file.py`.

## Step 1: Write `project.toml`

```toml
project_name = "docs-tutorial"
researcher_id = "team-docs"
global_seed = 11

[storage]
backend = "sqlite_blob"
root_dir = ".cache/themis-docs/project-file"
store_item_payloads = true
compression = "zstd"

[execution_policy]
max_retries = 3
retry_backoff_factor = 1.5
circuit_breaker_threshold = 5
```

This file now owns the shared execution policy and storage configuration.

## Step 2: Replace the inline `ProjectSpec`

Remove the inline `ProjectSpec(...)` block from your script and load it from
disk instead:

```python
orchestrator = Orchestrator.from_project_file(
    "project.toml",
    registry=registry,
    dataset_loader=dataset_loader,
)
result = orchestrator.run(experiment)
```

## Step 3: Run the script again

When you rerun the script, the behavior should stay the same, but the project
configuration now lives in one reusable file.

Check that:

- the run still completes successfully
- the storage root comes from `project.toml`
- you can change retry or storage policy without editing Python

## Step 4: Understand the split

- `ProjectSpec` changes slowly and carries infrastructure defaults.
- `ExperimentSpec` changes often and carries the actual sweep.

That split makes it easy to rerun different matrices against the same storage
layout, retry policy, and seeding strategy.

You now have a portable project config that can be shared across multiple
scripts and experiments.

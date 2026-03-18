# Load a Project File

Use a project file when storage and execution policy should stay stable across
many benchmarks.

## Run It

```bash
uv run python examples/02_project_file.py
```

Output:

```text
[{'model_id': 'demo-model', 'slice_id': 'greeting', 'metric_id': 'exact_match', 'mean': 1.0, 'count': 1}]
```

## Pattern

- keep `project.toml` for storage, retry, and execution policy
- keep `BenchmarkSpec` in Python where slices and prompt variants evolve faster
- build the orchestrator with `Orchestrator.from_project_file(...)`

## Full Script

--8<-- "examples/02_project_file.py"

# Backends API

Themis supports pluggable execution and storage backends.

## Storage

- `themis.backends.storage.StorageBackend`
- `themis.backends.storage.ExperimentStorage`
- `themis.storage.ExperimentStorage`

Example with `evaluate()`:

```python
from themis import evaluate
from themis.storage import ExperimentStorage

report = evaluate(
    "demo",
    model="fake-math-llm",
    storage_backend=ExperimentStorage(".cache/experiments"),
)
```

## Execution

- `themis.backends.execution.ExecutionBackend`
- `themis.backends.execution.LocalExecutionBackend`
- `themis.backends.execution.SequentialExecutionBackend`

Example with `evaluate()`:

```python
from themis import evaluate
from themis.backends.execution import LocalExecutionBackend

report = evaluate(
    "demo",
    model="fake-math-llm",
    execution_backend=LocalExecutionBackend(max_workers=8),
    workers=8,
)
```

## Notes

- `evaluate()` accepts both `storage_backend` and `execution_backend` directly.
- Custom backend interfaces are stable, but full custom storage integration into
  the high-level API is still evolving.

# Backends API

Themis supports pluggable execution and storage backends.

## Storage

- `themis.backends.storage.StorageBackend`
- `themis.backends.storage.LocalFileStorageBackend`
- `themis.storage.ExperimentStorage`

Example with `evaluate()`:

```python
from themis import evaluate
from themis.backends.storage import LocalFileStorageBackend

report = evaluate(
    "demo",
    model="fake-math-llm",
    storage_backend=LocalFileStorageBackend(".cache/experiments"),
)
```

## Execution

- `themis.backends.execution.ExecutionBackend`
- `themis.backends.execution.LocalExecutionBackend`
- `themis.backends.execution.SequentialExecutionBackend`

Example with vNext specs:

```python
from themis.backends.execution import LocalExecutionBackend
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec

report = ExperimentSession().run(
    spec,
    execution=ExecutionSpec(backend=LocalExecutionBackend(max_workers=8), workers=8),
)
```

## Notes

- `evaluate()` and `ExperimentSession` currently require storage backends that are
  ExperimentStorage-compatible.
- Custom backend interfaces are stable, but full custom storage integration into
  the high-level API is still evolving.

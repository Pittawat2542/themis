# vNext Specs

The vNext API uses explicit spec objects to configure experiments.

## ExperimentSpec

```python
from themis.specs import ExperimentSpec

spec = ExperimentSpec(
    dataset=[{"id": "1", "question": "2+2", "answer": "4"}],
    prompt="Solve: {question}",
    model="fake:fake-math-llm",
    sampling={"temperature": 0.0, "max_tokens": 128},
    pipeline=my_pipeline,
    run_id="run-1",
)
```

## ExecutionSpec

```python
from themis.specs import ExecutionSpec

execution = ExecutionSpec(
    backend=my_backend,
    workers=4,
    max_retries=3,
)
```

## StorageSpec

```python
from themis.specs import StorageSpec

storage = StorageSpec(
    backend=my_storage_backend,
    path=".cache/experiments",
    cache=True,
)
```

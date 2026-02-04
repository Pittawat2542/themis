# ExperimentSession

`ExperimentSession` is the primary entry point for running experiments using explicit specs.

## Usage

```python
from themis.evaluation.pipeline import MetricPipeline
from themis.evaluation import extractors, metrics
from themis.session import ExperimentSession
from themis.specs import ExperimentSpec, ExecutionSpec, StorageSpec

pipeline = MetricPipeline(
    extractor=extractors.IdentityExtractor(),
    metrics=[metrics.ResponseLength()],
)

spec = ExperimentSpec(
    dataset=[{"id": "1", "question": "2+2", "answer": "4"}],
    prompt="Solve: {question}",
    model="fake:fake-math-llm",
    sampling={"temperature": 0.0, "max_tokens": 128},
    pipeline=pipeline,
)

report = ExperimentSession().run(
    spec,
    execution=ExecutionSpec(workers=2),
    storage=StorageSpec(path=".cache/experiments"),
)
```

## Notes
- `ExperimentSession` requires an evaluation pipeline that implements `EvaluationPipelineContract`.
- The model string supports `provider:model_id` format.

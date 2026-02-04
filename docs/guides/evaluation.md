# Evaluation Guide

## API Levels

Themis supports two evaluation entry points:

- `themis.evaluate(...)`: fastest path for common workflows.
- `ExperimentSession().run(spec, ...)`: explicit spec/session API with full control.

## Quick API

```python
from themis import evaluate

report = evaluate(
    "gsm8k",
    model="gpt-4",
    limit=100,
    metrics=["exact_match", "math_verify"],
)
```

## Spec API

```python
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.presets import get_benchmark_preset
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

preset = get_benchmark_preset("gsm8k")
pipeline = MetricPipeline(extractor=preset.extractor, metrics=preset.metrics)

spec = ExperimentSpec(
    dataset=preset.load_dataset(limit=100),
    prompt=preset.prompt_template.template,
    model="litellm:gpt-4",
    sampling={"temperature": 0.0, "max_tokens": 512},
    pipeline=pipeline,
    run_id="gsm8k-gpt4",
)

report = ExperimentSession().run(
    spec,
    execution=ExecutionSpec(workers=8),
    storage=StorageSpec(path=".cache/experiments", cache=True),
)
```

## Custom Dataset

```python
from themis.evaluation import extractors, metrics
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.session import ExperimentSession
from themis.specs import ExperimentSpec

dataset = [
    {"id": "1", "question": "2+2", "answer": "4"},
    {"id": "2", "question": "3+3", "answer": "6"},
]

pipeline = MetricPipeline(
    extractor=extractors.IdentityExtractor(),
    metrics=[metrics.ExactMatch(), metrics.ResponseLength()],
)

spec = ExperimentSpec(
    dataset=dataset,
    prompt="Q: {question}\nA:",
    model="fake:fake-math-llm",
    pipeline=pipeline,
)

report = ExperimentSession().run(spec)
```

## Reading Results

```python
for name, aggregate in report.evaluation_report.metrics.items():
    print(name, aggregate.mean, aggregate.count)

for failure in report.failures:
    print(failure.sample_id, failure.message)
```

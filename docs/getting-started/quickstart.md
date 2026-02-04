# Quick Start

## 1) Run Your First Evaluation

```python
from themis import evaluate

report = evaluate("gsm8k", model="gpt-4", limit=10)
accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

## 2) Use vNext Specs + Session

```python
from themis.evaluation.metric_pipeline import MetricPipeline
from themis.presets import get_benchmark_preset
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

preset = get_benchmark_preset("demo")
pipeline = MetricPipeline(extractor=preset.extractor, metrics=preset.metrics)

spec = ExperimentSpec(
    dataset=preset.load_dataset(limit=5),
    prompt=preset.prompt_template.template,
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

## 3) CLI Flow

```bash
themis eval gsm8k --model gpt-4 --limit 100 --run-id run-a
themis eval gsm8k --model gpt-4 --temperature 0.7 --limit 100 --run-id run-b
themis compare run-a run-b
```

## 4) Explore Examples

- `examples-simple/01_quickstart.py`
- `examples-simple/02_custom_dataset.py`
- `examples-simple/04_comparison.py`
- `examples-simple/07_provider_ready.py`
- `examples-simple/08_resume_cache.py`
- `examples-simple/09_research_loop.py`

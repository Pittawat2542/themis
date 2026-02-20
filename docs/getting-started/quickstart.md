# Quick Start

## 1) Run Your First Evaluation (No API key)

```python
from themis import evaluate

report = evaluate("demo", model="fake-math-llm", limit=10)
exact_match = report.evaluation_report.metrics["ExactMatch"].mean
print(f"ExactMatch: {exact_match:.2%}")
```

## 2) Run a Hosted Benchmark (Optional)

```python
from themis import evaluate

report = evaluate("gsm8k", model="gpt-4", limit=10)
accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

## 3) Use Specs + Session

```python
from themis.evaluation.pipeline import EvaluationPipeline
from themis.presets import get_benchmark_preset
from themis.session import ExperimentSession
from themis.specs import ExecutionSpec, ExperimentSpec, StorageSpec

preset = get_benchmark_preset("demo")
pipeline = EvaluationPipeline(extractor=preset.extractor, metrics=preset.metrics)

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

## 4) CLI Flow

```bash
uv run python -m themis.cli eval gsm8k --model gpt-4 --limit 100 --run-id run-a
uv run python -m themis.cli eval gsm8k --model gpt-4 --temperature 0.7 --limit 100 --run-id run-b
uv run python -m themis.cli compare run-a run-b
```

## 5) Explore Examples

- `examples-simple/01_quickstart.py`
- `examples-simple/02_custom_dataset.py`
- `examples-simple/04_comparison.py`
- `examples-simple/07_provider_ready.py`
- `examples-simple/08_resume_cache.py`
- `examples-simple/09_research_loop.py`

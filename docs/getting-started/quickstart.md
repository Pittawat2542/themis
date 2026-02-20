# Quick Start

## 1) Run Your First Evaluation (No API key)

```python
from themis import evaluate

report = evaluate("demo", model="fake-math-llm", limit=10)
exact_match = report.evaluation_report.metrics["ExactMatch"].mean
print(f"ExactMatch: {exact_match:.2%}")
```

## 2) Run a Hosted Benchmark

```python
from themis import evaluate

report = evaluate("gsm8k", model="gpt-4", limit=10)
accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

## 3) Use the Full `evaluate()` API

```python
from themis import evaluate

report = evaluate(
    "demo",
    model="fake-math-llm",
    limit=5,
    temperature=0.0,
    max_tokens=128,
    workers=2,
    run_id="my-run",
    storage_path=".cache/experiments",
)
```

## 4) CLI Flow

```bash
uv run python -m themis.cli eval gsm8k --model gpt-4 --limit 100 --run-id run-a
uv run python -m themis.cli eval gsm8k --model gpt-4 --temperature 0.7 --limit 100 --run-id run-b
uv run python -m themis.cli compare run-a run-b
```

## 5) Explore Examples

- `examples/01_quickstart.py`
- `examples/02_custom_dataset.py`
- `examples/04_comparison.py`
- `examples/07_provider_ready.py`
- `examples/08_resume_cache.py`
- `examples/09_research_loop.py`

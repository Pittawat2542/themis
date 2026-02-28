# Quick Start

## 1) Run Your First Evaluation (No API key required)

Use our built-in `fake` model provider with the `demo` benchmark to ensure everything is installed correctly:

```python
from themis import evaluate

report = evaluate("demo", model="fake:fake-math-llm", limit=10)
exact_match = report.evaluation_report.metrics["ExactMatch"]
print(f"ExactMatch: {exact_match.mean:.2%} (n={exact_match.count})")
```

## 2) Run a Hosted Benchmark

Evaluating real models requires the corresponding provider's API key (e.g., `OPENAI_API_KEY`). By default, Themis uses [LiteLLM](https://github.com/BerriAI/litellm) for robust multi-provider routing.

```python
import os
from themis import evaluate

os.environ["OPENAI_API_KEY"] = "sk-..."

# Run the GSM8K math benchmark with GPT-4
report = evaluate("gsm8k", model="openai/gpt-4o", limit=10)
accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

## 3) Use the Full `evaluate()` API

You can customize almost everything directly through `themis.evaluate()`:

```python
from themis import evaluate

report = evaluate(
    "demo",
    model="fake:fake-math-llm",
    limit=5,
    temperature=0.0,
    max_tokens=128,
    max_retries=5,
    workers=2,
    run_id="my-run",
    storage=".cache/experiments", # specify save location
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

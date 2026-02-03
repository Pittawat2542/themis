# Quick Start

Get up and running with Themis in 5 minutes.

## Your First Evaluation

The simplest way to use Themis is with the `evaluate()` function:

```python
from themis import evaluate

# Evaluate GPT-4 on GSM8K math problems
report = evaluate(
    "gsm8k",
    model="gpt-4",
    limit=10,  # Start with 10 samples
)

accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

That's it. Themis will:
1. Load the GSM8K benchmark
2. Generate responses using GPT-4
3. Evaluate with math metrics
4. Cache results for future runs

## Using the CLI

Prefer command-line tools?

```bash
# Run evaluation
themis eval gsm8k --model gpt-4 --limit 10

# List runs
themis list runs

# Start web dashboard
themis serve
```

## Built-in Benchmarks

Themis ships with multiple built-in benchmarks across math, science, and QA.

```python
# Math reasoning
evaluate("gsm8k", model="gpt-4")
evaluate("math500", model="gpt-4")
evaluate("aime24", model="gpt-4")

# Knowledge
evaluate("mmlu-pro", model="gpt-4")
evaluate("supergpqa", model="gpt-4")

# Testing
evaluate("demo", model="fake-math-llm")  # No API key needed
```

List available benchmarks:

```python
from themis.presets import list_benchmarks
print(list_benchmarks())
```

## Custom Dataset

Evaluate on your own data:

```python
# Your dataset
dataset = [
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "What is the capital of France?", "answer": "Paris"},
]

report = evaluate(
    dataset,
    model="gpt-4",
    prompt="Question: {prompt}\nAnswer:",
    metrics=["exact_match"],
)
```

## Configuration Options

Customize the evaluation:

```python
report = evaluate(
    "gsm8k",
    model="gpt-4",

    # Sampling
    temperature=0.7,
    max_tokens=512,

    # Execution
    num_samples=1,
    workers=8,

    # Storage
    storage=".cache",
    run_id="my-exp",
    resume=True,
)
```

## Comparing Runs

Compare different configurations:

```python
from themis.comparison import compare_runs

# Run experiments
evaluate("gsm8k", model="gpt-4", temperature=0.0, run_id="temp-0")
evaluate("gsm8k", model="gpt-4", temperature=0.7, run_id="temp-0.7")

# Compare statistically
report = compare_runs(
    run_ids=["temp-0", "temp-0.7"],
    storage_path=".cache/experiments",
)

print(report.summary())
```

Or via CLI:

```bash
themis compare temp-0 temp-0.7 --output comparison.html
```

## Web Dashboard

Start the API server and view results in your browser:

```bash
themis serve
```

Then open:
- Dashboard: `http://localhost:8080/dashboard`
- API docs: `http://localhost:8080/docs`

## Multiple Providers

Themis works with 100+ LLM providers via LiteLLM:

```python
# OpenAI
evaluate("gsm8k", model="gpt-4")

# Anthropic
evaluate("gsm8k", model="claude-3-opus-20240229")

# Azure OpenAI
evaluate("gsm8k", model="azure/gpt-4")

# Local models
evaluate("gsm8k", model="ollama/llama3")
```

## What's Next?

- **[Core Concepts](concepts.md)** - Understand how Themis works
- **[Evaluation Guide](../guides/evaluation.md)** - Deep dive into evaluation
- **[Comparison Guide](../guides/comparison.md)** - Statistical comparison
- **[Jupyter Tutorials](../tutorials/notebooks.md)** - Interactive learning
- **[API Reference](../api/overview.md)** - Complete API documentation

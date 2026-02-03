# Themis Documentation

Welcome to the Themis documentation. This guide will help you get the most out of Themis, from quick start to advanced usage.

## Documentation Structure

### Getting Started

- **[Quick Start](getting-started/quickstart.md)** - Get up and running in 5 minutes
- **[Installation](getting-started/installation.md)** - Install Themis with optional features
- **[Core Concepts](getting-started/concepts.md)** - Understand how Themis works

### User Guides

- **[Evaluation Guide](guides/evaluation.md)** - Run evaluations with built-in metrics
- **[API Server Guide](reference/api-server.md)** - Use the REST API and web dashboard
- **[Comparison Guide](guides/comparison.md)** - Statistical comparison of runs
- **[LiteLLM / Providers](guides/providers.md)** - Use 100+ LLM providers

### Advanced Topics

- **[Extending Themis](EXTENDING_THEMIS.md)** - Custom components
- **[Extension Quick Reference](EXTENSION_QUICK_REFERENCE.md)** - One-page extension cheat sheet
- **[Extending Backends](customization/backends.md)** - Custom storage and execution
- **[Storage Architecture](guides/storage.md)** - Storage internals
- **[Cache Invalidation](guides/manual_verification_cache.md)** - Smart caching behavior
- **[Cost Tracking](guides/cost-tracking.md)** - Monitor API costs

### Reference

- **[Configuration Reference](guides/configuration.md)** - Configuration options
- **[CLI Reference](guides/cli.md)** - Command-line usage
- **[Benchmarks](reference/benchmarks.md)** - Built-in benchmarks
- **[Python API](api/overview.md)** - API documentation

---

## Quick Start

### Installation

```bash
pip install themis-eval
pip install themis-eval[math,nlp,code,server]
```

### Your First Evaluation

```python
from themis import evaluate

report = evaluate(
    "gsm8k",
    model="gpt-4",
    limit=100,
)

accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

### Using the CLI

```bash
# Run evaluation
themis eval gsm8k --model gpt-4 --limit 100 --run-id my-experiment

# Compare two models
themis compare my-experiment-1 my-experiment-2

# Share results
themis share my-experiment-1 --output-dir share
```

---

## Core Concepts

### 1. Evaluation API

Themis provides a simple `evaluate()` function:

```python
from themis import evaluate

report = evaluate(
    "gsm8k",
    model="gpt-4",
    temperature=0.0,
    max_tokens=512,
    workers=8,
    storage=".cache",
    run_id="my-run",
    resume=True,
)
```

### 2. Benchmarks & Presets

Themis includes built-in benchmarks across math, science, medicine, and QA.

```python
from themis.presets import list_benchmarks
print(list_benchmarks())
```

### 3. Metrics

Supported metric families:
- **Math**: `ExactMatch`, `MathVerifyAccuracy`
- **NLP**: `BLEU`, `ROUGE`, `BERTScore`, `METEOR`
- **Code**: `PassAtK`, `ExecutionAccuracy`, `CodeBLEU`

```python
report = evaluate(
    "gsm8k",
    model="gpt-4",
    metrics=["exact_match", "math_verify"],
)
```

### 4. Storage & Caching

Themis caches results to resume runs and avoid repeated API calls.

```python
report = evaluate(
    "gsm8k",
    model="gpt-4",
    run_id="experiment-1",
    resume=True,
)
```

### 5. Comparison Engine

Compare multiple runs with statistical tests:

```python
from themis.comparison import compare_runs
from themis.comparison import StatisticalTest

report = compare_runs(
    run_ids=["run-gpt4", "run-claude"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.BOOTSTRAP,
    alpha=0.05,
)

print(report.summary())
```

# API Reference Overview

Complete API documentation for Themis.

## Core API

### Main Functions

::: themis.evaluate
    options:
      show_source: false
      heading_level: 3

The primary entry point for running evaluations. See [evaluate() documentation](evaluate.md) for details.


## Modules

### Comparison

Statistical comparison of multiple runs.

- **[compare_runs()](comparison.md)** - Compare experiment runs
- **Statistical Tests** - T-test, bootstrap, permutation
- **Reports** - ComparisonReport, WinLossMatrix

### Presets

Built-in benchmark configurations.

- **[list_benchmarks()](presets.md)** - List available benchmarks
- **[get_benchmark_preset()](presets.md)** - Get benchmark configuration
- **BenchmarkPreset** - Preset data structure

### Metrics

Evaluation metrics for different domains.

- **Core**: ExactMatch, ResponseLength
- **Math**: MathVerifyAccuracy
- **NLP**: BLEU, ROUGE, BERTScore, METEOR
- **Code**: PassAtK, CodeBLEU, ExecutionAccuracy

### Backends

Pluggable backend interfaces.

- **Storage**: `StorageBackend`, `ExperimentStorage`
- **Execution**: `ExecutionBackend`, `LocalExecutionBackend`, `SequentialExecutionBackend`

## Package Structure

```
themis/
├── __init__.py              # Main exports
├── api.py                   # evaluate() function

├── presets/                 # Benchmark presets
│   ├── benchmarks.py
│   └── models.py
├── comparison/              # Statistical comparison
│   ├── engine.py
│   ├── statistics.py
│   └── reports.py
├── evaluation/              # Metrics & evaluation
│   └── metrics/
├── backends/                # Pluggable backends
│   ├── storage.py
│   └── execution.py
├── generation/              # LLM generation
├── experiment/              # Experiment orchestration
├── storage/                 # Storage adapters
├── server/                  # API server
└── cli/                     # Command-line interface
```

## Type Hints

Themis is fully typed with Python type hints:

```python
from themis import evaluate
from themis.core.entities import ExperimentReport

report: ExperimentReport = evaluate(
    "gsm8k",
    model="gpt-4",
)
```

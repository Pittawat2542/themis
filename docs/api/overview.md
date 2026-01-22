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

[View Comparison API →](comparison.md)

### Presets

Built-in benchmark configurations.

- **[list_benchmarks()](presets.md)** - List available benchmarks
- **[get_benchmark_preset()](presets.md)** - Get benchmark configuration
- **BenchmarkPreset** - Preset data structure

[View Presets API →](presets.md)

### Metrics

Evaluation metrics for different domains.

#### Math Metrics
- **ExactMatch** - String matching
- **MathVerify** - Symbolic & numeric verification

#### NLP Metrics
- **BLEU** - N-gram overlap
- **ROUGE** - Recall-oriented matching
- **BERTScore** - Semantic similarity
- **METEOR** - Alignment-based scoring

#### Code Metrics
- **PassAtK** - Pass rate at K samples
- **CodeBLEU** - Code similarity
- **ExecutionAccuracy** - Functional correctness

[View Metrics API →](metrics.md)

### Backends

Pluggable backend interfaces.

#### Storage Backends
- **StorageBackend** - Abstract interface
- **LocalFileStorageBackend** - File-based storage

#### Execution Backends
- **ExecutionBackend** - Abstract interface
- **LocalExecutionBackend** - Multi-threaded execution
- **SequentialExecutionBackend** - Single-threaded execution

[View Backends API →](backends.md)

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
│       ├── math/
│       ├── nlp/
│       └── code/
├── backends/                # Pluggable backends
│   ├── storage.py
│   └── execution.py
├── generation/              # LLM generation
├── experiment/              # Experiment orchestration
├── server/                  # API server
└── cli/                     # Command-line interface
```

## Type Hints

Themis is fully typed with Python type hints:

```python
from themis import evaluate
from themis.core.entities import ExperimentReport

# Type hints are available
result: ExperimentReport = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
)
```

## Common Patterns

### Basic Evaluation

```python
from themis import evaluate

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    limit=100,
)
```

### With Type Annotations

```python
from typing import Dict, Any, List
from themis import evaluate
from themis.core.entities import ExperimentReport

def run_evaluation(
    benchmark: str,
    model: str,
    **kwargs: Any
) -> ExperimentReport:
    return evaluate(benchmark=benchmark, model=model, **kwargs)
```

### Custom Dataset

```python
from typing import Sequence, Dict, Any
from themis import evaluate

def evaluate_custom(
    dataset: Sequence[Dict[str, Any]],
    model: str
) -> ExperimentReport:
    return evaluate(dataset, model=model)
```

### Comparison

```python
from themis.comparison import compare_runs
from themis.comparison.reports import ComparisonReport
from pathlib import Path

def compare_experiments(
    run_ids: List[str],
    storage_path: Path
) -> ComparisonReport:
    return compare_runs(run_ids, storage_path=storage_path)
```

## Error Handling

Themis raises standard Python exceptions:

```python
from themis import evaluate

try:
    result = evaluate(
        benchmark="invalid-benchmark",
        model="gpt-4",
    )
except ValueError as e:
    print(f"Invalid benchmark: {e}")
except FileNotFoundError as e:
    print(f"Storage error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration

All configuration is done through function parameters:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    
    # Model config
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    
    # Execution config
    num_samples=1,
    workers=8,
    
    # Storage config
    storage=".cache",
    run_id="my-experiment",
    resume=True,
    
    # Output config
    output="results.json",
)
```

## Next Steps

- **[evaluate() API](evaluate.md)** - Detailed evaluate() documentation
- **[Comparison API](comparison.md)** - Statistical comparison
- **[Presets API](presets.md)** - Built-in benchmarks
- **[Metrics API](metrics.md)** - Evaluation metrics
- **[Backends API](backends.md)** - Extensibility interfaces

## Need Help?

- Check [User Guide](../guides/evaluation.md) for usage patterns
- See [Examples](../tutorials/examples.md) for working code
- Visit [GitHub Discussions](https://github.com/pittawat2542/themis/discussions) for questions

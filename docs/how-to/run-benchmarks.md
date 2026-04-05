---
title: Run benchmarks
diataxis: how-to
audience: users evaluating against catalog entries
goal: Show how to execute named benchmark entries and understand their constraints.
---

# Run benchmarks

Goal: execute a named benchmark from the catalog instead of wiring the dataset yourself.

When to use this:

Use this guide when a shipped benchmark entry already matches the task you want to run.

## Procedure

Run the shortest benchmark workflow:

```bash
themis quick-eval benchmark --name mmlu_pro
```

Or run the same named benchmark from Python through the catalog API using
`themis.catalog.run(...)`.

Then inspect the benchmark catalog for prerequisites such as optional dataset dependencies or adapter-specific execution constraints.

When you want to inspect or filter the benchmark dataset before running it:

```python
from themis.catalog import load

benchmark = load("mmlu_pro")
dataset = benchmark.materialize_dataset()
```

Benchmark slicing and downsampling are code-authored today. When you need a subset of a shipped benchmark, load or materialize a `Dataset`, then filter or sample its `cases` before compiling the experiment. Themis treats that filtered dataset as the benchmark you asked it to run.

One concrete pattern is:

```python
from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Dataset

source_dataset = Dataset(...)
filtered_dataset = source_dataset.model_copy(
    update={
        "cases": [
            case
            for case in source_dataset.cases
            if case.metadata.get("category") == "hard"
        ][:100]
    }
)

experiment = Experiment(
    generation=GenerationConfig(...),
    evaluation=EvaluationConfig(...),
    storage=StorageConfig(store="sqlite", parameters={"path": "runs.sqlite3"}),
    datasets=[filtered_dataset],
)
```

This is the current supported way to run just a slice or downsample of a benchmark.

## Variants

- quick local check: `quick-eval benchmark`
- Python catalog execution: `themis.catalog.run(...)`
- custom experiment around the same dataset: load the definition first and move to Python or config-driven experiments
- filtered benchmark slice: construct a `Dataset(cases=[...])` from the original benchmark input
- benchmark downsample: sample cases before `Experiment.compile()`

## Expected result

You should get a completed run keyed by the named benchmark entry and know whether the benchmark requires extra setup.

## Troubleshooting

- [Benchmark catalog](../reference/benchmark-catalog.md)
- [Benchmark adapters](../explanation/benchmark-adapters.md)

## Local smoke checks

Use these optional commands when you want to validate benchmark wiring against
local services instead of the demo generator.

Generation-model smoke check against your local OpenAI-compatible endpoint:

```python
from themis.adapters.openai import openai
from themis.catalog import run
from themis.core.stores import InMemoryRunStore

result = run(
    "frontierscience",
    model=openai(
        "google/gemma-4-26b-a4b",
        base_url="http://127.0.0.1:1234/v1",
    ),
    store=InMemoryRunStore(),
)
print(result.status)
```

Code-benchmark smoke check with local sandbox services:

```bash
export THEMIS_CODE_SANDBOX_FUSION_URL=http://localhost:8080
export THEMIS_CODE_PISTON_URL=http://localhost:2000
themis quick-eval benchmark --name codeforces
```

These smoke checks are optional local verification only. The automated test
suite should continue to use fixture-backed datasets and fake or demo
components.

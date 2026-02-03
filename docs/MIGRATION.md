# Migration Guide: v1.x to v2.0

This guide helps you migrate from Themis v1.x to v2.0.

## Overview

Themis v2.0 is a complete rewrite focusing on simplicity and usability. The core changes:

- ✅ Simple `themis.evaluate()` API replaces `ExperimentBuilder`
- ✅ Built-in benchmarks (no config files needed)
- ✅ Simplified CLI (5 commands instead of 20+)
- ✅ Statistical comparison engine
- ✅ Web dashboard and REST API
- ❌ Breaking changes to configuration format

## Quick Migration

### Old API (v1.x)

```python
from themis.experiment.builder import ExperimentBuilder
from themis.generation.prompt_template import PromptTemplate
from themis.evaluation.metrics.math import ExactMatch

# Build experiment
experiment = (
    ExperimentBuilder()
    .with_dataset_name("gsm8k")
    .with_model("gpt-4")
    .with_prompt_template(PromptTemplate(template="{prompt}"))
    .with_metrics([ExactMatch()])
    .with_workers(8)
    .build()
)

# Run
report = experiment.run()
```

### New API (v2.0)

```python
from themis import evaluate

# Simple one-liner!
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    workers=8,
)
```

## Detailed Changes

### 1. Evaluation API

| v1.x | v2.0 |
|------|------|
| `ExperimentBuilder()` | `themis.evaluate()` |
| `.with_dataset_name()` | `benchmark=` parameter |
| `.with_model()` | `model=` parameter |
| `.with_prompt_template()` | `prompt=` parameter |
| `.with_metrics()` | `metrics=` parameter |
| `.build().run()` | Direct call |

**Before:**
```python
experiment = (
    ExperimentBuilder()
    .with_dataset_name("gsm8k")
    .with_model("gpt-4")
    .build()
)
report = experiment.run()
```

**After:**
```python
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### 2. Configuration

| v1.x | v2.0 |
|------|------|
| JSON/YAML config files | Function parameters |
| Hydra configs | Python API |
| `config.sample.json` | Direct parameters |

**Before:**
```json
{
  "dataset": {"name": "gsm8k"},
  "model": {"name": "gpt-4"},
  "metrics": [{"type": "exact_match"}]
}
```

**After:**
```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch"],
)
```

### 3. CLI Commands

| v1.x | v2.0 |
|------|------|
| `themis run experiment` | `themis eval <benchmark>` |
| `themis list experiments` | `themis list runs` |
| `themis export results` | `--output` parameter |
| 20+ commands | 5 commands |

**Before:**
```bash
themis run experiment --config config.json
themis list experiments
themis export results --format json
```

**After:**
```bash
themis eval gsm8k --model gpt-4
themis list runs
themis eval gsm8k --model gpt-4 --output results.json
```

### 4. Metrics

| v1.x | v2.0 |
|------|------|
| Import individual metric classes | String names or class instances |
| Manual metric instantiation | Automatic from preset |

**Before:**
```python
from themis.evaluation.metrics.math import ExactMatch, MathVerify

metrics = [ExactMatch(), MathVerify()]
```

**After:**
```python
# Use string names
metrics = ["exact_match", "math_verify"]

# Or use instances (still supported)
from themis.evaluation.metrics import ExactMatch, MathVerifyAccuracy
metrics = [ExactMatch(), MathVerifyAccuracy()]
```

### 5. Comparison

| v1.x | v2.0 |
|------|------|
| Manual comparison scripts | `themis.comparison.compare_runs()` |
| No statistical tests | T-test, bootstrap, permutation |
| Basic CSV export | HTML, JSON, Markdown export |

**Before:**
```python
# Manual comparison
report1 = experiment1.run()
report2 = experiment2.run()

# Compare manually
diff = report1.metrics["accuracy"] - report2.metrics["accuracy"]
print(f"Difference: {diff}")
```

**After:**
```python
from themis.comparison import compare_runs

report = compare_runs(
    ["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test="bootstrap",
)

print(report.summary())
```

## Breaking Changes

### Removed Features

- ❌ Hydra configuration system
- ❌ Complex builder pattern
- ❌ `ExperimentBuilder` class
- ❌ Config file requirement for simple tasks

### Changed Behavior

- ⚠️ Default storage location changed to `.cache/experiments`
- ⚠️ Run IDs are auto-generated (timestamp-based)
- ⚠️ Caching is enabled by default (`resume=True`)

### Renamed

- `with_dataset_name()` → `benchmark=` parameter
- `with_model()` → `model=` parameter
- `with_prompt_template()` → `prompt=` parameter
- `exact_match` metric → `ExactMatch`

## Migration Strategy

### Step 1: Update Imports

```python
# Old
from themis.experiment.builder import ExperimentBuilder

# New
from themis import evaluate
```

### Step 2: Replace Builder Pattern

```python
# Old
experiment = (
    ExperimentBuilder()
    .with_dataset_name("gsm8k")
    .with_model("gpt-4")
    .with_temperature(0.7)
    .with_max_tokens(512)
    .build()
)
report = experiment.run()

# New
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.7,
    max_tokens=512,
)
```

### Step 3: Update CLI Usage

```bash
# Old
themis run experiment --config config.json

# New
themis eval gsm8k --model gpt-4
```

### Step 4: Update Comparison

```python
# Old: Manual comparison
# (no built-in support)

# New: Built-in comparison
from themis.comparison import compare_runs

report = compare_runs(
    ["run-1", "run-2"],
    storage_path=".cache/experiments",
)
```

## Compatibility

### v1.x Examples Still Work

Old examples in `examples/` still work:

```bash
# Old API still functional
python -m examples.getting_started.cli run
python -m examples.config_file.cli run
```

But **new code should use v2.0 API**.

### Gradual Migration

You can migrate gradually:

1. Keep old experiments running
2. Start new experiments with v2.0 API
3. Migrate old code when convenient

## Common Migration Patterns

### Pattern 1: Simple Evaluation

**Before:**
```python
from themis.experiment.builder import ExperimentBuilder

experiment = (
    ExperimentBuilder()
    .with_dataset_name("gsm8k")
    .with_model("gpt-4")
    .build()
)
report = experiment.run()
```

**After:**
```python
from themis import evaluate

result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### Pattern 2: Custom Dataset

**Before:**
```python
experiment = (
    ExperimentBuilder()
    .with_dataset(my_dataset)
    .with_model("gpt-4")
    .with_prompt_template(PromptTemplate(template="{prompt}"))
    .build()
)
```

**After:**
```python
result = evaluate(
    my_dataset,
    model="gpt-4",
    prompt="{prompt}",
)
```

### Pattern 3: Multiple Models

**Before:**
```python
models = ["gpt-4", "claude-3"]

for model in models:
    experiment = (
        ExperimentBuilder()
        .with_dataset_name("gsm8k")
        .with_model(model)
        .build()
    )
    report = experiment.run()
    print(f"{model}: {report.metrics}")
```

**After:**
```python
for model in ["gpt-4", "claude-3"]:
    result = evaluate(
        benchmark="gsm8k",
        model=model,
        run_id=f"gsm8k-{model}",
    )
    print(f"{model}: {result.evaluation_report.metrics}")

# Then compare
from themis.comparison import compare_runs
report = compare_runs(
    ["gsm8k-gpt-4", "gsm8k-claude-3"],
    storage_path=".cache/experiments",
)
```

## FAQ

### Can I still use config files?

Yes, but they're optional:

```python
import json

with open("config.json") as f:
    config = json.load(f)

result = evaluate(**config)
```

### Will old storage work?

No, v2.0 uses Storage V2 architecture. You'll need to:
- Re-run evaluations, or
- Migrate storage (manual process)

### Can I use both APIs?

Yes, but not recommended:
- Old examples still work
- New code should use v2.0 API
- Don't mix in the same project

### What about custom metrics?

They still work! The `Metric` interface is unchanged:

```python
from themis.evaluation.metrics import Metric

class MyMetric(Metric):
    @property
    def name(self) -> str:
        return "MyMetric"
    
    def evaluate(self, response, reference):
        # Your logic
        pass

# Use it
result = evaluate(
    dataset=my_dataset,
    model="gpt-4",
    metrics=[MyMetric()],
)
```

## Getting Help

Having trouble migrating?

- Check the [documentation](index.md)
- Ask on [GitHub Discussions](https://github.com/pittawat2542/themis/discussions)
- Open an [issue](https://github.com/pittawat2542/themis/issues)

## Summary

**Key takeaway**: v2.0 is simpler! Most migrations involve:
- Removing builder pattern
- Removing config files
- Using function parameters instead

**Example migration:**
```python
# Before (10+ lines)
experiment = (
    ExperimentBuilder()
    .with_dataset_name("gsm8k")
    .with_model("gpt-4")
    .with_metrics([ExactMatch()])
    .with_temperature(0.7)
    .with_max_tokens(512)
    .build()
)
report = experiment.run()

# After (1 line!)
result = evaluate(benchmark="gsm8k", model="gpt-4", temperature=0.7, max_tokens=512)
```

Welcome to v2.0! 🎉

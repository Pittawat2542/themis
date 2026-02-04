# Comparison Guide

Use `themis.comparison.compare_runs` to compare two or more run IDs.

## Basic

```python
from themis.comparison import compare_runs
from themis.comparison.statistics import StatisticalTest

report = compare_runs(
    run_ids=["run-a", "run-b"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.BOOTSTRAP,
    alpha=0.05,
)

print(report.summary())
```

## Metric-Scoped Comparison

```python
report = compare_runs(
    run_ids=["run-a", "run-b", "run-c"],
    storage_path=".cache/experiments",
    metrics=["ExactMatch"],
)
```

## CLI

```bash
themis compare run-a run-b --output comparison.html
```

## Notes

- Run IDs must exist in the same storage root.
- Metrics compared are the intersection available across selected runs.

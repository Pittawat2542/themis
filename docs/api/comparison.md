# Comparison API

Statistical comparison of experiment runs.

## Functions

### compare_runs()

Compare multiple experiment runs with statistical significance testing.

```python
def compare_runs(
    run_ids: Sequence[str],
    *,
    storage_path: str | Path,
    metrics: Sequence[str] | None = None,
    statistical_test: StatisticalTest = StatisticalTest.BOOTSTRAP,
    alpha: float = 0.05,
) -> ComparisonReport:
```

#### Parameters

- **`run_ids`** : `Sequence[str]` - List of run IDs to compare (minimum 2)
- **`storage_path`** : `str | Path` - Path to experiment storage
- **`metrics`** : `Sequence[str] | None` - Metrics to compare (None = all)
- **`statistical_test`** : `StatisticalTest` - Type of statistical test
- **`alpha`** : `float` - Significance level (default: 0.05 for 95% confidence)

#### Returns

**`ComparisonReport`** - Contains pairwise results, win/loss matrices, and rankings

#### Example

```python
from themis.experiment.comparison import compare_runs
from themis.experiment.comparison import StatisticalTest

report = compare_runs(
    run_ids=["gpt4-run", "claude-run"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.BOOTSTRAP,
    alpha=0.05,
)

print(report.summary())
```

## Classes

### ComparisonReport

Comprehensive comparison report for multiple runs.

#### Attributes

- **`run_ids`** : `list[str]` - All run IDs being compared
- **`metrics`** : `list[str]` - Metrics being compared
- **`pairwise_results`** : `list[ComparisonResult]` - All pairwise comparisons
- **`win_loss_matrices`** : `dict[str, WinLossMatrix]` - Matrix for each metric
- **`best_run_per_metric`** : `dict[str, str]` - Best run for each metric
- **`overall_best_run`** : `str | None` - Overall best run

#### Methods

**`get_comparison(run_a: str, run_b: str, metric: str) -> ComparisonResult | None`**

Get specific pairwise comparison.

**`get_metric_results(metric: str) -> list[ComparisonResult]`**

Get all comparisons for a metric.

**`summary(include_details: bool = False) -> str`**

Generate human-readable summary.

**`to_dict() -> dict`**

Convert to dictionary for serialization.

#### Example

```python
report = compare_runs(["run-1", "run-2"], storage_path=".cache")

# Access results
best = report.overall_best_run
print(f"Winner: {best}")

# Get specific comparison
result = report.get_comparison("run-1", "run-2", "ExactMatch")
if result and result.is_significant():
    print(f"Significant difference: {result.delta:.3f}")
```

### ComparisonResult

Result of comparing two runs on a single metric.

#### Attributes

- **`metric_name`** : `str` - Metric being compared
- **`run_a_id`** : `str` - First run identifier
- **`run_b_id`** : `str` - Second run identifier
- **`run_a_mean`** : `float` - Mean score for run A
- **`run_b_mean`** : `float` - Mean score for run B
- **`delta`** : `float` - Difference (run_a - run_b)
- **`delta_percent`** : `float` - Percentage difference
- **`winner`** : `str` - Winner ID or "tie"
- **`test_result`** : `TestResult | None` - Statistical test result

#### Methods

**`is_significant() -> bool`**

Check if difference is statistically significant.

**`summary() -> str`**

Generate human-readable summary.

### WinLossMatrix

Win/loss/tie matrix for comparing multiple runs.

#### Attributes

- **`run_ids`** : `list[str]` - Run IDs in the matrix
- **`metric_name`** : `str` - Metric being compared
- **`matrix`** : `list[list[str]]` - 2D matrix of results
- **`win_counts`** : `dict[str, int]` - Wins for each run
- **`loss_counts`** : `dict[str, int]` - Losses for each run
- **`tie_counts`** : `dict[str, int]` - Ties for each run

#### Methods

**`get_result(run_a: str, run_b: str) -> str`**

Get comparison result between two runs.

**`rank_runs() -> list[tuple[str, int, int, int]]`**

Rank runs by performance (wins, then losses).

**`to_table() -> str`**

Generate formatted table representation.

#### Example

```python
report = compare_runs(
    ["run-1", "run-2", "run-3"],
    storage_path=".cache",
)

matrix = report.win_loss_matrices["ExactMatch"]

# View table
print(matrix.to_table())

# Get rankings
for run_id, wins, losses, ties in matrix.rank_runs():
    print(f"{run_id}: {wins}W-{losses}L-{ties}T")
```

## Statistical Tests

### StatisticalTest

Enum of available statistical tests.

```python
from themis.experiment.comparison import StatisticalTest

# Available tests
StatisticalTest.T_TEST        # Student's t-test
StatisticalTest.BOOTSTRAP     # Bootstrap confidence intervals
StatisticalTest.PERMUTATION   # Permutation test
StatisticalTest.NONE          # No testing
```

### t_test()

Perform Student's t-test.

```python
from themis.experiment.comparison import t_test

result = t_test(
    samples_a=[0.8, 0.85, 0.82],
    samples_b=[0.7, 0.75, 0.72],
    alpha=0.05,
    paired=True,
)

print(f"p-value: {result.p_value}")
print(f"Significant: {result.significant}")
print(f"Effect size: {result.effect_size}")
```

### bootstrap_confidence_interval()

Compute bootstrap confidence interval.

```python
from themis.experiment.comparison import bootstrap_confidence_interval

result = bootstrap_confidence_interval(
    samples_a=[0.8, 0.85, 0.82],
    samples_b=[0.7, 0.75, 0.72],
    n_bootstrap=10000,
    confidence_level=0.95,
    seed=42,
)

print(f"CI: {result.confidence_interval}")
print(f"Significant: {result.significant}")
```

Bootstrap mode in Themis is CI-only inference and does not return p-values.

### permutation_test()

Perform permutation test.

```python
from themis.experiment.comparison import permutation_test

result = permutation_test(
    samples_a=[0.8, 0.85, 0.82],
    samples_b=[0.7, 0.75, 0.72],
    n_permutations=10000,
    alpha=0.05,
    seed=42,
)

print(f"p-value: {result.p_value}")
```

## Complete Example

```python
from themis import evaluate
from themis.experiment.comparison import compare_runs
from themis.experiment.comparison import StatisticalTest

# Run two experiments
result1 = evaluate("gsm8k",
    model="gpt-4",
    temperature=0.0,
    run_id="gpt4-temp0",
    limit=100,
)

result2 = evaluate("gsm8k",
    model="gpt-4",
    temperature=0.7,
    run_id="gpt4-temp07",
    limit=100,
)

# Compare with bootstrap test
report = compare_runs(
    run_ids=["gpt4-temp0", "gpt4-temp07"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.BOOTSTRAP,
    alpha=0.05,
)

# View results
print(report.summary(include_details=True))

# Check significance
for result in report.pairwise_results:
    if result.is_significant():
        print(f"{result.metric_name}: {result.winner} wins (p={result.test_result.p_value:.4f})")
    else:
        print(f"{result.metric_name}: no significant difference")

# Export
import json
Path("comparison.json").write_text(json.dumps(report.to_dict(), indent=2))
```

## See Also

- [Comparison Guide](../guides/comparison.md) - Detailed usage guide
- [CLI Reference](../guides/cli.md#themis-compare) - Command-line usage

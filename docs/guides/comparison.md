# Statistical Comparison Guide

This guide covers how to compare experiment runs in Themis using statistical tests, win/loss matrices, and effect sizes.

## Table of Contents

- [Quick Start](#quick-start)
- [Statistical Tests](#statistical-tests)
- [Interpreting Results](#interpreting-results)
- [Win/Loss Matrices](#winloss-matrices)
- [Export and Visualization](#export-and-visualization)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### Compare Two Runs

```python
from themis.comparison import compare_runs

report = compare_runs(
    run_ids=["run-gpt4", "run-claude"],
    storage_path=".cache/experiments",
)

print(report.summary())
```

### CLI Comparison

```bash
# Basic comparison
themis compare run-1 run-2

# With statistical test
themis compare run-1 run-2 --test bootstrap --alpha 0.05

# Export to HTML
themis compare run-1 run-2 --output comparison.html
```

---

## Statistical Tests

Themis supports multiple statistical tests to determine if differences between runs are significant.

### Bootstrap Confidence Intervals (Default)

Bootstrap resampling provides robust confidence intervals:

```python
from themis.comparison import compare_runs
from themis.comparison.statistics import StatisticalTest

report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.BOOTSTRAP,
    alpha=0.05,  # 95% confidence level
)
```

**Advantages:**
- ✅ No assumptions about data distribution
- ✅ Works with small sample sizes
- ✅ Provides confidence intervals
- ✅ Robust to outliers

**When to use:**
- Default choice for most comparisons
- When you have paired data (same test set)
- When sample sizes are small (n < 30)

### T-Test

Student's t-test for comparing means:

```python
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.T_TEST,
    alpha=0.05,
)
```

**Advantages:**
- ✅ Fast computation
- ✅ Well-understood statistics
- ✅ Provides effect sizes (Cohen's d)

**Limitations:**
- ⚠️ Assumes normal distribution
- ⚠️ Sensitive to outliers

**When to use:**
- When data is approximately normal
- When you want effect sizes
- When you need fast computation

### Permutation Test

Non-parametric test based on random permutations:

```python
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.PERMUTATION,
    alpha=0.01,  # 99% confidence
)
```

**Advantages:**
- ✅ No distributional assumptions
- ✅ Exact p-values
- ✅ Works with any test statistic

**Limitations:**
- ⚠️ Slower than t-test
- ⚠️ Requires sufficient samples

**When to use:**
- When data is non-normal
- When you need exact p-values
- When you have computational resources

### No Testing

Compare without statistical tests:

```python
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.NONE,
)
```

**When to use:**
- Exploratory analysis
- When sample sizes are too small
- When you just want descriptive statistics

---

## Interpreting Results

### P-Values

The p-value represents the probability of observing the difference by chance:

- **p < 0.05**: Statistically significant (95% confidence)
- **p < 0.01**: Highly significant (99% confidence)
- **p ≥ 0.05**: Not statistically significant

```python
report = compare_runs(["run-1", "run-2"], storage_path=".cache")

for result in report.pairwise_results:
    if result.is_significant():
        print(f"✓ {result.metric_name}: {result.winner} wins (p={result.test_result.p_value:.4f})")
    else:
        print(f"  {result.metric_name}: No significant difference")
```

### Effect Sizes

Effect size (Cohen's d) measures the magnitude of the difference:

- **d < 0.2**: Small effect
- **d ≈ 0.5**: Medium effect
- **d > 0.8**: Large effect

```python
report = compare_runs(
    ["run-1", "run-2"],
    storage_path=".cache",
    statistical_test=StatisticalTest.T_TEST,  # Provides effect sizes
)

for result in report.pairwise_results:
    if result.test_result and result.test_result.effect_size:
        effect = result.test_result.effect_size
        if effect > 0.8:
            magnitude = "large"
        elif effect > 0.5:
            magnitude = "medium"
        elif effect > 0.2:
            magnitude = "small"
        else:
            magnitude = "negligible"
        
        print(f"{result.metric_name}: d={effect:.2f} ({magnitude})")
```

### Confidence Intervals

Confidence intervals show the range of likely values for the difference:

```python
report = compare_runs(
    ["run-1", "run-2"],
    storage_path=".cache",
    statistical_test=StatisticalTest.BOOTSTRAP,
)

for result in report.pairwise_results:
    if result.test_result and result.test_result.confidence_interval:
        low, high = result.test_result.confidence_interval
        print(f"{result.metric_name}: 95% CI = [{low:.3f}, {high:.3f}]")
        
        if low > 0:
            print(f"  → {result.run_a_id} is consistently better")
        elif high < 0:
            print(f"  → {result.run_b_id} is consistently better")
        else:
            print(f"  → No clear winner")
```

---

## Win/Loss Matrices

For comparing 3+ runs, win/loss matrices show pairwise comparisons:

```python
report = compare_runs(
    run_ids=["gpt-4", "claude-3", "gemini-pro"],
    storage_path=".cache/experiments",
)

# Get matrix for a specific metric
matrix = report.win_loss_matrices["ExactMatch"]

# Display table
print(matrix.to_table())

# Get rankings
print("\nRankings:")
for run_id, wins, losses, ties in matrix.rank_runs():
    print(f"{run_id}: {wins}W - {losses}L - {ties}T")
```

**Output:**
```
Run                  | gpt-4        | claude-3     | gemini-pro   
------------------------------------------------------------------------
gpt-4                | —            | win          | win          
claude-3             | loss         | —            | tie          
gemini-pro           | loss         | tie          | —            

Rankings:
  gpt-4: 2W - 0L - 0T
  claude-3: 0W - 1L - 1T
  gemini-pro: 0W - 2L - 0T
```

### Accessing Matrix Results

```python
# Get specific comparison
result = matrix.get_result("gpt-4", "claude-3")
print(result)  # "win", "loss", or "tie"

# Check if run A beats run B
if matrix.get_result("gpt-4", "claude-3") == "win":
    print("GPT-4 beats Claude-3")

# Get all wins for a run
wins = matrix.win_counts["gpt-4"]
print(f"GPT-4 won {wins} comparisons")
```

---

## Export and Visualization

### Export to JSON

```python
import json
from pathlib import Path

report = compare_runs(["run-1", "run-2"], storage_path=".cache")

# Convert to dictionary
data = report.to_dict()

# Save to file
output = Path("comparison.json")
output.write_text(json.dumps(data, indent=2))
```

### Export to HTML

```bash
# CLI export
themis compare run-1 run-2 --output comparison.html
```

Or programmatically:

```python
from themis.comparison import compare_runs

report = compare_runs(["run-1", "run-2"], storage_path=".cache")

# Generate HTML (from CLI code)
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .win {{ background-color: #d4edda; }}
        .loss {{ background-color: #f8d7da; }}
        .tie {{ background-color: #fff3cd; }}
    </style>
</head>
<body>
    <h1>Comparison Report</h1>
    <p><strong>Runs:</strong> {', '.join(report.run_ids)}</p>
    <p><strong>Overall Best:</strong> {report.overall_best_run}</p>
    
    <h2>Results</h2>
    <ul>
        {''.join(f'<li>{result.summary()}</li>' for result in report.pairwise_results)}
    </ul>
</body>
</html>
"""

Path("comparison.html").write_text(html)
```

### Export to Markdown

```bash
themis compare run-1 run-2 --output comparison.md
```

---

## Advanced Usage

### Compare Specific Metrics

```python
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache",
    metrics=["ExactMatch", "BLEU"],  # Only compare these
)
```

### Custom Significance Level

```python
# 99% confidence (α = 0.01)
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache",
    alpha=0.01,
)

# 90% confidence (α = 0.10)
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache",
    alpha=0.10,
)
```

### Programmatic Statistical Tests

Use the statistics module directly:

```python
from themis.comparison.statistics import t_test, bootstrap_confidence_interval

# Your data
model_a_scores = [0.85, 0.87, 0.83, 0.90, 0.82]
model_b_scores = [0.78, 0.80, 0.79, 0.82, 0.77]

# T-test
result = t_test(model_a_scores, model_b_scores, paired=True, alpha=0.05)
print(f"T-test: p={result.p_value:.4f}, significant={result.significant}")

# Bootstrap
result = bootstrap_confidence_interval(
    model_a_scores,
    model_b_scores,
    n_bootstrap=10000,
    confidence_level=0.95,
    seed=42,  # For reproducibility
)
print(f"Bootstrap: CI={result.confidence_interval}, significant={result.significant}")
```

### Custom Test Statistics

```python
from themis.comparison.statistics import permutation_test

# Custom statistic: median difference
def median_diff(a, b):
    return sorted(a)[len(a)//2] - sorted(b)[len(b)//2]

result = permutation_test(
    model_a_scores,
    model_b_scores,
    statistic_fn=median_diff,
    n_permutations=10000,
)

print(f"Permutation test on medians: p={result.p_value:.4f}")
```

### McNemar's Test for Binary Outcomes

For comparing two models on binary outcomes (correct/incorrect):

```python
from themis.comparison.statistics import mcnemar_test

# Contingency table: (both wrong, A wrong B correct, A correct B wrong, both correct)
contingency = (10, 5, 15, 70)  # A correct more often than B

result = mcnemar_test(contingency, alpha=0.05)
print(f"McNemar's test: χ²={result.statistic:.2f}, p={result.p_value:.4f}")

if result.significant:
    print("Models perform significantly differently")
```

---

## Best Practices

### 1. Use Paired Comparisons

When comparing on the same test set, use paired tests:

```python
# Evaluate both models on same data
result_a = evaluate(benchmark="gsm8k", model="gpt-4", run_id="run-a")
result_b = evaluate(benchmark="gsm8k", model="claude-3", run_id="run-b")

# Compare with paired test (default)
report = compare_runs(["run-a", "run-b"], storage_path=".cache")
```

### 2. Report Multiple Metrics

Don't rely on a single metric:

```python
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache",
    # Will compare all available metrics
)

# Check consistency across metrics
for metric in report.metrics:
    winner = report.best_run_per_metric[metric]
    print(f"{metric}: {winner}")
```

### 3. Consider Practical Significance

A statistically significant difference may not be practically important:

```python
for result in report.pairwise_results:
    if result.is_significant():
        # Check if the difference is meaningful
        if abs(result.delta_percent) < 1.0:
            print(f"{result.metric_name}: Significant but small difference ({result.delta_percent:.2f}%)")
        else:
            print(f"{result.metric_name}: Significant and meaningful ({result.delta_percent:.2f}%)")
```

### 4. Use Appropriate Sample Sizes

Statistical power depends on sample size:

- **n < 10**: Results may be unreliable
- **10 ≤ n < 30**: Use bootstrap or permutation tests
- **n ≥ 30**: Any test should work
- **n ≥ 100**: High statistical power

```python
# For small datasets, use bootstrap
if num_samples < 30:
    statistical_test = StatisticalTest.BOOTSTRAP
else:
    statistical_test = StatisticalTest.T_TEST

report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache",
    statistical_test=statistical_test,
)
```

### 5. Correct for Multiple Comparisons

When comparing many runs, apply Bonferroni correction:

```python
# Comparing 5 runs = 10 pairwise comparisons
num_runs = 5
num_comparisons = (num_runs * (num_runs - 1)) // 2  # 10

# Bonferroni correction
adjusted_alpha = 0.05 / num_comparisons  # 0.005

report = compare_runs(
    run_ids=[f"run-{i}" for i in range(num_runs)],
    storage_path=".cache",
    alpha=adjusted_alpha,
)
```

---

## Examples

### Example 1: Compare Temperature Settings

```python
from themis import evaluate
from themis.comparison import compare_runs

# Run with different temperatures
for temp in [0.0, 0.3, 0.7, 1.0]:
    evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        temperature=temp,
        limit=100,
        run_id=f"temp-{temp}",
    )

# Compare all
report = compare_runs(
    run_ids=["temp-0.0", "temp-0.3", "temp-0.7", "temp-1.0"],
    storage_path=".cache/experiments",
)

print(report.summary(include_details=True))
```

### Example 2: Model Comparison

```python
# Compare different models
models = {
    "gpt-4": "gpt-4",
    "gpt-3.5": "gpt-3.5-turbo",
    "claude-3": "claude-3-opus-20240229",
}

for name, model in models.items():
    evaluate(
        benchmark="gsm8k",
        model=model,
        limit=100,
        run_id=f"model-{name}",
    )

# Compare
report = compare_runs(
    run_ids=[f"model-{name}" for name in models.keys()],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.BOOTSTRAP,
)

# Print rankings
for metric, matrix in report.win_loss_matrices.items():
    print(f"\n{metric} Rankings:")
    for run_id, wins, losses, ties in matrix.rank_runs():
        print(f"  {run_id}: {wins}W-{losses}L-{ties}T")
```

### Example 3: Prompt Engineering

```python
prompts = {
    "zero-shot": "Solve: {prompt}",
    "cot": "Let's think step by step to solve: {prompt}",
    "pal": "Write Python code to solve: {prompt}",
}

for name, prompt in prompts.items():
    evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        prompt=prompt,
        limit=100,
        run_id=f"prompt-{name}",
    )

# Compare prompts
report = compare_runs(
    run_ids=[f"prompt-{name}" for name in prompts.keys()],
    storage_path=".cache/experiments",
)

print(f"Best prompt: {report.overall_best_run}")
```

---

## Troubleshooting

### Issue: "Need at least 2 runs to compare"

```python
# Make sure you have at least 2 runs
from themis.experiment.storage import ExperimentStorage

storage = ExperimentStorage(".cache/experiments")
runs = storage.list_runs()
print(f"Available runs: {runs}")

if len(runs) < 2:
    print("Run more evaluations first!")
```

### Issue: "Run not found"

```python
# Check run IDs are correct
print(f"Available runs: {storage.list_runs()}")

# Use correct run IDs
report = compare_runs(
    run_ids=storage.list_runs()[:2],  # First 2 runs
    storage_path=".cache/experiments",
)
```

### Issue: "No common metrics found"

```python
# Make sure runs use the same metrics
# Or specify metrics explicitly
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache",
    metrics=["ExactMatch"],  # Only compare this metric
)
```

---

## Further Reading

- [Comparison API Reference](api/comparison.md) - Complete API documentation
- [Statistical Tests](api/comparison.md#statistical-tests) - Available tests
- [User Guide](guides/evaluation.md) - Practical usage

---

**Ready to compare your models? Start with `themis compare run-1 run-2`! 🔬**

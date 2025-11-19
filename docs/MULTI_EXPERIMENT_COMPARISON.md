# Multi-Experiment Comparison Guide

This guide explains how to compare multiple experiment runs using Themis' comparison tools.

## Overview

Themis provides powerful tools for comparing multiple experiment runs across different configurations, models, or prompts. This is essential for:
- **Model selection**: Compare performance across different LLMs
- **Hyperparameter tuning**: Identify optimal temperature, top_p settings
- **Prompt engineering**: Evaluate different prompt strategies
- **Cost-performance tradeoffs**: Find Pareto-optimal configurations

## Quick Start

### 1. Run Multiple Experiments

First, run several experiments with different configurations:

```bash
# Run experiment with GPT-4
uv run python -m themis.cli run-config \
  --config experiments/gpt4_config.yaml

# Run experiment with GPT-3.5
uv run python -m themis.cli run-config \
  --config experiments/gpt35_config.yaml

# Run experiment with different temperature
uv run python -m themis.cli run-config \
  --config experiments/high_temp_config.yaml
```

### 2. Compare Results

```bash
# Compare all three runs
uv run python -m themis.cli compare \
  --run-ids run-gpt4 run-gpt35 run-high-temp \
  --storage .cache/runs \
  --output comparison.csv
```

Output:
```
================================================================================
Experiment Comparison
================================================================================
Run ID               | accuracy | f1_score | Samples | Failures
--------------------------------------------------------------------------------
run-gpt4            | 0.8500   | 0.8200   | 100     | 2
run-gpt35           | 0.7800   | 0.7500   | 100     | 5
run-high-temp       | 0.8200   | 0.7900   | 100     | 3
================================================================================
```

## Core Commands

### `compare` - Compare Multiple Runs

Compare experiments across multiple metrics:

```bash
uv run python -m themis.cli compare \
  --run-ids run-1 run-2 run-3 \
  --storage .cache/runs \
  --metrics accuracy cost latency \
  --output comparison.md \
  --highlight-best accuracy
```

**Parameters:**
- `--run-ids`: List of run IDs to compare (required)
- `--storage`: Storage directory (default: `.cache/runs`)
- `--metrics`: Specific metrics to compare (default: all)
- `--output`: Output file (.csv, .md, or .json)
- `--format`: Output format if extension doesn't indicate (csv, markdown, json)
- `--highlight-best`: Metric to highlight best performer

### `diff` - Configuration Differences

Show what changed between two experiment configurations:

```bash
uv run python -m themis.cli diff \
  --run-id-a run-baseline \
  --run-id-b run-optimized \
  --storage .cache/runs
```

Output:
```
================================================================================
Configuration Diff: run-baseline → run-optimized
================================================================================

📝 Changed Fields:

  temperature:
    - run-baseline: 0.0
    + run-optimized: 0.7

  max_tokens:
    - run-baseline: 512
    + run-optimized: 1024

➕ Added Fields (in run_id_b):
  top_p: 0.95

================================================================================
```

### `pareto` - Pareto Frontier Analysis

Find experiments on the Pareto frontier (optimal tradeoffs):

```bash
uv run python -m themis.cli pareto \
  --run-ids run-1 run-2 run-3 run-4 run-5 \
  --objectives accuracy cost \
  --maximize true false \
  --storage .cache/runs
```

**Example:** Find experiments with best accuracy/cost tradeoff
- `--objectives`: Metrics to optimize
- `--maximize`: For each objective, whether higher is better (true/false)

Output:
```
================================================================================
Pareto Frontier Analysis
================================================================================

⭐ Found 3 Pareto-optimal experiment(s):

  • run-2
      accuracy: 0.9000
      cost: 0.1500

  • run-3
      accuracy: 0.8500
      cost: 0.0800

  • run-4
      accuracy: 0.8200
      cost: 0.0500

📊 Dominated experiments (2):
  • run-1
  • run-5
================================================================================
```

## Programmatic API

### Basic Comparison

```python
from themis.experiment.comparison import compare_experiments
from pathlib import Path

# Load and compare experiments
comparison = compare_experiments(
    run_ids=["run-1", "run-2", "run-3"],
    storage_dir=Path(".cache/runs"),
    metrics=["accuracy", "f1_score"],  # Optional: specific metrics
    include_metadata=True
)

# Access results
print(f"Loaded {len(comparison.experiments)} experiments")
print(f"Metrics: {comparison.metrics}")

# Rank by metric
ranked = comparison.rank_by_metric("accuracy", ascending=False)
for exp in ranked:
    print(f"{exp.run_id}: {exp.get_metric('accuracy'):.3f}")
```

### Find Best Performer

```python
# Find experiment with highest accuracy
best = comparison.highlight_best("accuracy", higher_is_better=True)
if best:
    print(f"Best accuracy: {best.run_id} ({best.get_metric('accuracy'):.3f})")

# Find experiment with lowest cost
cheapest = comparison.highlight_best("cost", higher_is_better=False)
if cheapest:
    print(f"Lowest cost: {cheapest.run_id} (${cheapest.get_metric('cost'):.2f})")
```

### Pareto Frontier

```python
# Find Pareto-optimal experiments (maximize accuracy, minimize cost)
pareto_ids = comparison.pareto_frontier(
    objectives=["accuracy", "cost"],
    maximize=[True, False]
)

print(f"Pareto-optimal experiments: {pareto_ids}")
```

### Export Results

```python
# Export to CSV
comparison.to_csv("comparison.csv", include_metadata=True)

# Export to Markdown
comparison.to_markdown("comparison.md")

# Export to JSON
import json
data = comparison.to_dict()
with open("comparison.json", "w") as f:
    json.dump(data, f, indent=2)

# Export to LaTeX (for research papers)
comparison.to_latex(
    "results.tex",
    style="booktabs",  # or "basic"
    caption="Comparison of experiment results",
    label="tab:results"
)
```

### Configuration Diff

```python
from themis.experiment.comparison import diff_configs

diff = diff_configs("run-baseline", "run-optimized", ".cache/runs")

if diff.has_differences():
    print("Changed fields:")
    for field, (old, new) in diff.changed_fields.items():
        print(f"  {field}: {old} → {new}")

    print("\nAdded fields:")
    for field, value in diff.added_fields.items():
        print(f"  {field}: {value}")
```

## Export Formats

### CSV Format

Wide format with one row per experiment:

```csv
run_id,accuracy,f1_score,model,temperature,sample_count,failure_count
run-1,0.8500,0.8200,gpt-4,0.0,100,2
run-2,0.7800,0.7500,gpt-3.5-turbo,0.0,100,5
```

### Markdown Format

Formatted table for README files:

```markdown
# Experiment Comparison

| Run ID | accuracy | f1_score | Samples | Failures |
| --- | --- | --- | --- | --- |
| run-1 | 0.8500 | 0.8200 | 100 | 2 |
| run-2 | 0.7800 | 0.7500 | 100 | 5 |
```

### JSON Format

Structured data for downstream tooling:

```json
{
  "experiments": [
    {
      "run_id": "run-1",
      "metric_values": {
        "accuracy": 0.85,
        "f1_score": 0.82
      },
      "metadata": {
        "model": "gpt-4",
        "temperature": 0.0
      },
      "sample_count": 100,
      "failure_count": 2
    }
  ],
  "metrics": ["accuracy", "f1_score"]
}
```

### LaTeX Format

Publication-ready tables for research papers:

**Booktabs Style** (recommended for academic papers):

```latex
\begin{table}[htbp]
\centering
\caption{Comparison of experiment results}
\label{tab:results}
\begin{tabular}{lrrr}
\toprule
Run ID & accuracy & f1\_score & Samples \\
\midrule
run-1 & 0.8500 & 0.8200 & 100 \\
run-2 & 0.7800 & 0.7500 & 100 \\
\bottomrule
\end{tabular}
\end{table}
```

**Basic Style** (with borders):

```latex
\begin{table}[htbp]
\centering
\caption{Comparison of experiment results}
\label{tab:results}
\begin{tabular}{|l|r|r|r|}
\hline
Run ID & accuracy & f1\_score & Samples \\
\hline
run-1 & 0.8500 & 0.8200 & 100 \\
\hline
run-2 & 0.7800 & 0.7500 & 100 \\
\hline
\end{tabular}
\end{table}
```

**Features:**
- Automatic underscore escaping in run IDs
- Cost column included if cost data available
- Caption and label support for cross-referencing
- Two styles: `booktabs` (professional) and `basic` (simple)

**Usage in LaTeX documents:**

```latex
\documentclass{article}
\usepackage{booktabs}  % For booktabs style

\begin{document}
\section{Experimental Results}

Table~\ref{tab:results} shows the comparison of different models.

\input{results.tex}  % Include generated table

\end{document}
```

## Common Workflows

### Workflow 1: Model Comparison

```bash
# Run experiments with different models
for model in gpt-4 gpt-3.5-turbo claude-3-sonnet; do
  uv run python -m themis.cli run-config \
    --config base_config.yaml \
    generation.model_identifier=$model \
    run_id=run-$model
done

# Compare results
uv run python -m themis.cli compare \
  --run-ids run-gpt-4 run-gpt-3.5-turbo run-claude-3-sonnet \
  --highlight-best accuracy \
  --output model_comparison.md
```

### Workflow 2: Temperature Sweep

```bash
# Run experiments with different temperatures
for temp in 0.0 0.3 0.5 0.7 1.0; do
  uv run python -m themis.cli run-config \
    --config base_config.yaml \
    generation.sampling.temperature=$temp \
    run_id=run-temp-$temp
done

# Find Pareto frontier (accuracy vs diversity)
uv run python -m themis.cli pareto \
  --run-ids run-temp-0.0 run-temp-0.3 run-temp-0.5 run-temp-0.7 run-temp-1.0 \
  --objectives accuracy diversity \
  --maximize true true
```

### Workflow 3: Prompt Engineering

```bash
# Run experiments with different prompts
uv run python -m themis.cli run-config --config zero_shot.yaml
uv run python -m themis.cli run-config --config few_shot.yaml
uv run python -m themis.cli run-config --config chain_of_thought.yaml

# Compare prompts
uv run python -m themis.cli compare \
  --run-ids run-zero-shot run-few-shot run-cot \
  --storage .cache/runs \
  --output prompt_comparison.csv

# Check what changed
uv run python -m themis.cli diff \
  --run-id-a run-zero-shot \
  --run-id-b run-few-shot
```

## Best Practices

### 1. Use Descriptive Run IDs

```yaml
# Good: Descriptive run ID
run_id: "gpt4-temp07-math500-2024-11-18"

# Bad: Auto-generated timestamp
run_id: "run-20241118-123456"
```

### 2. Keep Storage Organized

```bash
# Separate storage by experiment type
--storage .cache/model-comparison
--storage .cache/temperature-sweep
--storage .cache/prompt-engineering
```

### 3. Always Export Results

```bash
# Export for later analysis
uv run python -m themis.cli compare \
  --run-ids run-1 run-2 run-3 \
  --output results.csv

# Can analyze with pandas, Excel, etc.
```

### 4. Document Configuration Differences

```bash
# Save diffs for reproducibility
uv run python -m themis.cli diff \
  --run-id-a baseline \
  --run-id-b optimized \
  > diffs/baseline-vs-optimized.txt
```

## Troubleshooting

### "No valid experiments found"

**Cause:** Experiments don't have `report.json` files.

**Solution:** Experiments must be run with storage enabled:

```yaml
# In config file
storage:
  path: .cache/runs
```

Or from CLI:
```bash
uv run python -m themis.cli run-config \
  --config my_config.yaml \
  --json-output .cache/runs/my-run/report.json
```

### Missing Metrics

**Cause:** Some experiments may not have all metrics.

**Solution:** Use `--metrics` to specify common metrics:

```bash
uv run python -m themis.cli compare \
  --run-ids run-1 run-2 \
  --metrics accuracy  # Only compare accuracy
```

### Large Number of Experiments

For comparing many experiments (10+), use programmatic API:

```python
from themis.experiment.comparison import compare_experiments
import pandas as pd

# Load all experiments
comparison = compare_experiments(
    run_ids=[f"run-{i}" for i in range(100)],
    storage_dir=".cache/runs"
)

# Convert to DataFrame for analysis
data = []
for exp in comparison.experiments:
    row = {"run_id": exp.run_id, **exp.metric_values}
    data.append(row)

df = pd.DataFrame(data)
print(df.describe())
```

## Next Steps

- See [COOKBOOK.md](../COOKBOOK.md) for more experiment patterns
- See [examples/prompt_engineering](../examples/prompt_engineering) for full comparison examples
- See [IMPROVEMENT_PLAN.md](../IMPROVEMENT_PLAN.md) for upcoming features (interactive visualizations, cost tracking, etc.)

## Related Documentation

- [Configuration Guide](./CONFIGURATION.md)
- [Statistical Analysis](./STATISTICS.md)
- [Export Formats](./EXPORT.md)

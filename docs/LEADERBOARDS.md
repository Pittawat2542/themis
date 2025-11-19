# Benchmark Leaderboards

Generate ranked leaderboards from experiment results for README files, documentation, and research papers.

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Output Formats](#output-formats)
- [Customization](#customization)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Quick Start

```bash
# Basic leaderboard
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 run-4 \
  --metric accuracy \
  --output LEADERBOARD.md

# With model names and costs
uv run python -m themis.cli leaderboard \
  --run-ids run-gpt4 run-claude run-gemini \
  --metric accuracy \
  --include-metadata model \
  --include-cost \
  --title "Math500 Benchmark" \
  --output leaderboard.md
```

## CLI Usage

### Basic Command

```bash
uv run python -m themis.cli leaderboard \
  --run-ids RUN_ID [RUN_ID ...] \
  --metric METRIC_NAME \
  [OPTIONS]
```

### Parameters

**Required:**
- `--run-ids`: List of experiment run IDs to include
- `--metric`: Primary metric for ranking (e.g., "accuracy", "f1", "cost")

**Optional:**
- `--storage`: Storage directory (default: `.cache/runs`)
- `--format`: Output format - `markdown`, `latex`, or `csv` (default: `markdown`)
- `--output`: Output file path (prints to terminal if omitted)
- `--title`: Leaderboard title (default: "Leaderboard")
- `--ascending`: Rank in ascending order for metrics where lower is better (default: `false`)
- `--include-cost`: Include cost column (default: `true`)
- `--include-metadata`: Metadata fields to include (can be specified multiple times)

### Output Formats

#### Markdown Format

Perfect for README files and documentation:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --format markdown \
  --output LEADERBOARD.md
```

**Output:**

```markdown
# Leaderboard

*Generated: 2024-11-19 10:30:00*

| Rank | Run ID | Accuracy | Cost ($) | Samples | Failures |
| --- | --- | --- | --- | --- | --- |
| 1 | run-gpt4 | **0.9200** | 0.1000 | 500 | 2 |
| 2 | run-claude | **0.9000** | 0.0500 | 500 | 3 |
| 3 | run-gemini | **0.8800** | 0.0200 | 500 | 5 |
```

**Features:**
- Primary metric values in **bold**
- Automatic ranking
- Cost column when available
- Clean Markdown table format

#### LaTeX Format

Publication-ready tables for research papers:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --format latex \
  --title "Benchmark Results" \
  --output leaderboard.tex
```

**Output:**

```latex
\begin{table}[htbp]
\centering
\caption{Benchmark Results}
\label{tab:leaderboard}
\begin{tabular}{clrrrr}
\toprule
Rank & Run ID & \textbf{accuracy} & Cost (\$) & Samples & Failures \\
\midrule
1 & run-gpt4 & \textbf{0.9200} & 0.1000 & 500 & 2 \\
2 & run-claude & \textbf{0.9000} & 0.0500 & 500 & 3 \\
3 & run-gemini & \textbf{0.8800} & 0.0200 & 500 & 5 \\
\bottomrule
\end{tabular}
\end{table}
```

**Features:**
- Booktabs style for professional appearance
- Primary metric in **bold**
- Automatic underscore escaping
- Caption and label for cross-referencing

**Usage in LaTeX:**

```latex
\documentclass{article}
\usepackage{booktabs}

\begin{document}
\section{Results}

Table~\ref{tab:leaderboard} presents our benchmark results.

\input{leaderboard.tex}

\end{document}
```

#### CSV Format

Structured data for analysis:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --format csv \
  --output leaderboard.csv
```

**Output:**

```csv
rank,run_id,accuracy,cost,sample_count,failure_count
1,run-gpt4,0.92,0.1,500,2
2,run-claude,0.9,0.05,500,3
3,run-gemini,0.88,0.02,500,5
```

**Features:**
- Easy to import into pandas, Excel, etc.
- Compatible with data analysis tools
- Machine-readable format

## Customization

### Including Metadata

Add custom columns from experiment metadata:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --include-metadata model \
  --include-metadata temperature \
  --include-metadata max_tokens \
  --output leaderboard.md
```

**Output includes additional columns:**

| Rank | Run ID | Accuracy | model | temperature | max_tokens |
| --- | --- | --- | --- | --- | --- |
| 1 | run-1 | **0.9200** | gpt-4 | 0.0 | 512 |

### Ascending Order

For metrics where lower is better (e.g., cost, latency):

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric cost \
  --ascending true \
  --output cost_leaderboard.md
```

### Custom Title

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --title "MATH-500 Official Leaderboard" \
  --output MATH500_LEADERBOARD.md
```

### Without Cost Column

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --include-cost false \
  --output leaderboard.md
```

## Examples

### Example 1: Model Comparison

Compare different models on the same benchmark:

```bash
# Run experiments with different models
for model in gpt-4 claude-3-5-sonnet gemini-1.5-pro; do
  uv run python -m themis.cli math500 \
    --model $model \
    --limit 100 \
    --run-id run-$model
done

# Generate leaderboard
uv run python -m themis.cli leaderboard \
  --run-ids run-gpt-4 run-claude-3-5-sonnet run-gemini-1.5-pro \
  --metric accuracy \
  --include-metadata model \
  --title "MATH-500: Model Comparison" \
  --output MATH500_LEADERBOARD.md
```

### Example 2: Cost-Performance Analysis

Find the most cost-effective configuration:

```bash
# Generate leaderboard sorted by cost
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 run-4 run-5 \
  --metric cost \
  --ascending true \
  --include-metadata model \
  --include-metadata accuracy \
  --title "Cost-Performance Analysis" \
  --output cost_analysis.md
```

**Result shows cheapest runs first, with accuracy for comparison:**

| Rank | Run ID | Cost ($) | model | accuracy |
| --- | --- | --- | --- | --- |
| 1 | run-5 | **0.0200** | gpt-3.5-turbo | 0.78 |
| 2 | run-3 | **0.0500** | claude-3-haiku | 0.82 |
| 3 | run-2 | **0.1000** | claude-3-sonnet | 0.90 |

### Example 3: Temperature Sweep

Compare different temperature settings:

```bash
# Run experiments with varying temperatures
for temp in 0.0 0.3 0.5 0.7 1.0; do
  uv run python -m themis.cli math500 \
    --model gpt-4 \
    --temperature $temp \
    --limit 50 \
    --run-id run-temp-$temp
done

# Generate leaderboard
uv run python -m themis.cli leaderboard \
  --run-ids run-temp-0.0 run-temp-0.3 run-temp-0.5 run-temp-0.7 run-temp-1.0 \
  --metric accuracy \
  --include-metadata temperature \
  --title "Temperature Sweep: GPT-4 on MATH-500" \
  --output temperature_sweep.md
```

### Example 4: LaTeX for Papers

Create publication-ready table:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids baseline few-shot cot self-consistency \
  --metric accuracy \
  --format latex \
  --title "Prompting Strategy Comparison on MATH-500" \
  --include-metadata strategy \
  --output table_results.tex
```

Include in your paper:

```latex
\section{Experimental Results}

We compare four prompting strategies on the MATH-500 benchmark.
Table~\ref{tab:leaderboard} shows that self-consistency achieves
the highest accuracy at 92\%, followed by chain-of-thought at 88\%.

\input{table_results.tex}
```

### Example 5: Multi-Metric Leaderboard

Show multiple metrics side-by-side:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --format csv \
  --output leaderboard.csv
```

Then manually edit to include additional metrics from the CSV, or use the comparison command for a complete view:

```bash
uv run python -m themis.cli compare \
  --run-ids run-1 run-2 run-3 \
  --metrics accuracy f1 precision recall \
  --output comparison.md
```

## Best Practices

### 1. Consistent Run IDs

Use descriptive, consistent naming for runs:

```bash
# Good: Descriptive and consistent
run-gpt4-temp0-math500
run-claude3-temp0-math500
run-gemini-temp0-math500

# Bad: Unclear and inconsistent
run-1
experiment-tuesday
test_v2
```

### 2. Include Metadata

Always include relevant metadata for context:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --include-metadata model \
  --include-metadata temperature \
  --include-metadata prompt_strategy \
  --output leaderboard.md
```

### 3. Update Regularly

Keep leaderboards up-to-date:

```bash
# Add to your CI/CD or run script
#!/bin/bash

# Run all benchmark variants
for model in gpt-4 claude-3 gemini; do
  uv run python -m themis.cli math500 \
    --model $model \
    --run-id run-$model
done

# Regenerate leaderboard
uv run python -m themis.cli leaderboard \
  --run-ids run-gpt-4 run-claude-3 run-gemini \
  --metric accuracy \
  --include-metadata model \
  --title "MATH-500 Leaderboard" \
  --output LEADERBOARD.md

# Commit changes
git add LEADERBOARD.md
git commit -m "Update leaderboard"
```

### 4. Document Benchmark Details

Include experimental details in the README:

```markdown
# MATH-500 Leaderboard

**Benchmark:** MATH-500
**Evaluation Metric:** Exact match accuracy
**Temperature:** 0.0
**Max Tokens:** 512
**Last Updated:** 2024-11-19

[Full leaderboard](LEADERBOARD.md)

## Top Results

| Rank | Model | Accuracy | Cost |
| --- | --- | --- | --- |
| 1 | GPT-4 | 92.0% | $0.10 |
| 2 | Claude 3 Sonnet | 90.0% | $0.05 |
| 3 | Gemini 1.5 Pro | 88.0% | $0.02 |
```

### 5. Version Control

Track leaderboard changes over time:

```bash
# Create dated leaderboards
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --output "leaderboards/$(date +%Y-%m-%d)_leaderboard.md"

# Keep a latest symlink
ln -sf "$(date +%Y-%m-%d)_leaderboard.md" leaderboards/LATEST.md
```

### 6. Cost-Performance Tradeoffs

Show both accuracy and cost to help users make informed decisions:

```bash
uv run python -m themis.cli leaderboard \
  --run-ids run-1 run-2 run-3 \
  --metric accuracy \
  --include-cost true \
  --include-metadata model \
  --output leaderboard.md
```

Annotate with cost-effectiveness notes:

```markdown
# Leaderboard

| Rank | Model | Accuracy | Cost | Notes |
| --- | --- | --- | --- | --- |
| 1 | GPT-4 | 92.0% | $0.10 | 🏆 Best accuracy |
| 2 | Claude | 90.0% | $0.05 | ⚖️ Best balance |
| 3 | Gemini | 88.0% | $0.02 | 💰 Most economical |
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Update Leaderboard

on:
  push:
    paths:
      - 'experiments/**'
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  update-leaderboard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install themis

      - name: Generate leaderboard
        run: |
          python -m themis.cli leaderboard \
            --run-ids run-1 run-2 run-3 \
            --metric accuracy \
            --output LEADERBOARD.md

      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add LEADERBOARD.md
          git commit -m "Update leaderboard" || exit 0
          git push
```

## Troubleshooting

### No Cost Data

If cost column is empty, ensure experiments tracked costs:

1. Use LiteLLM provider (cost tracking automatic)
2. Or manually implement cost tracking in custom providers

### Incorrect Ranking

Check metric name:

```bash
# List available metrics
uv run python -m themis.cli compare \
  --run-ids run-1 \
  --storage .cache/runs
```

### Missing Metadata

Ensure metadata is captured during experiments:

```python
# In your experiment config
metadata = {
    "model": "gpt-4",
    "temperature": 0.0,
    "prompt_strategy": "chain-of-thought",
}
```

## See Also

- [Multi-Experiment Comparison](MULTI_EXPERIMENT_COMPARISON.md) - Compare experiments in detail
- [Cost Tracking](COST_TRACKING.md) - Track and optimize experiment costs
- [CLAUDE.md](../CLAUDE.md) - Complete CLI reference

# Analyze a Stored Run

This tutorial starts from the shipped paired-comparison example and walks
through one complete analysis loop: run the example, inspect the report, compare
the models, and confirm the stored SQLite summaries.

## Before You Start

Install the stats extra and run the example from the repository root:

```bash
uv add "themis-eval[stats]"
uv run python examples/04_compare_models.py
```

Expected output:

```text
delta_mean= 0.5 adjusted_p_value= 0.25 pairs= 6
Report written to: .cache/themis-examples/04-compare-models/report.md
```

The example writes its SQLite store to:

```text
.cache/themis-examples/04-compare-models/themis.sqlite3
```

## Step 1: Open the report path

The example already exported a Markdown report:

```text
.cache/themis-examples/04-compare-models/report.md
```

That file is the quickest handoff artifact when you want the paired comparison
and leaderboard in one place.

## Step 2: Rebuild the comparison in Python

Open a Python session or a notebook in the repository and recreate the same run
state:

```python
import runpy

namespace = runpy.run_path("examples/04_compare_models.py")

orchestrator = namespace["Orchestrator"].from_project_spec(
    namespace["build_project"](),
    registry=namespace["build_registry"](),
    dataset_loader=namespace["ArithmeticDatasetLoader"](),
)
result = orchestrator.run(namespace["build_experiment"]())
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])

comparison = evaluation_result.compare(
    metric_id="exact_match",
    baseline_model_id="baseline",
    treatment_model_id="candidate",
    p_value_correction="holm",
)
row = comparison.rows[0]
print(row.delta_mean, row.adjusted_p_value, row.pair_count)
```

Expected output:

```text
0.5 0.25 6
```

## Step 3: Build a quick leaderboard

```python
leaderboard = evaluation_result.leaderboard(metric_id="exact_match")
for row in leaderboard:
    print(row["model_id"], row["task_id"], row["mean"])
```

Expected output:

```text
baseline paired-math 0.5
candidate paired-math 1.0
```

## Step 4: Inspect the SQLite summaries

Use the operator CLI when you want a lightweight check without hydrating the
full trial records:

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/04-compare-models/themis.sqlite3 \
  --metric exact_match
```

Expected output:

```text
ev:fc7ad3e8b3e2	baseline	paired-math	exact_match	0.5000	6
ev:fc7ad3e8b3e2	candidate	paired-math	exact_match	1.0000	6
```

## Step 5: Inspect one concrete trial

```python
trial = evaluation_result.get_trial(evaluation_result.trial_hashes[0])
candidate_view = evaluation_result.view_timeline(trial.candidates[0].candidate_id)

print(candidate_view.inference.raw_text)
print(candidate_view.evaluation.aggregate_scores)
```

This is the point where you switch from aggregate comparison back to one stored
example.

## Next Steps

- Use [Analyze Results](../guides/analyze-results.md) for the task-oriented guide.
- Use [Compare and Export Results](../guides/compare-and-export.md) when you
  want JSON, CSV, or report exports.
- Use [Use the Quickcheck CLI](../guides/quickcheck.md) for more operator
  queries against the same SQLite store.

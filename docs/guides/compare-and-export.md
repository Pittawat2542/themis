# Compare and Export Results

Install the `stats` extra first:

```bash
uv add "themis-eval[stats]"
```

Prerequisites:

- start from `examples/04_compare_models.py` or a similar run with at least one
  evaluation overlay
- use `.cache/themis-examples/04-compare-models` if you want paths and output
  that match the shipped example

## Paired Comparison

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
comparison = evaluation_result.compare(
    metric_id="exact_match",
    baseline_model_id="baseline",
    treatment_model_id="candidate",
    p_value_correction="holm",
)
row = comparison.rows[0]
print("delta_mean=", row.delta_mean, "adjusted_p_value=", row.adjusted_p_value, "pairs=", row.pair_count)
```

Expected output from `examples/04_compare_models.py`:

```text
delta_mean= 0.5 adjusted_p_value= 0.25 pairs= 6
```

Comparison rows are paired by `task_id`, `metric_id`, and `item_id`.

If a run has multiple evaluation overlays, pick the one you want first with
`result.for_evaluation(...)` so the comparison reads from the intended score set.
To inspect the exact overlay hash first:

```python
print(result.evaluation_hashes[0])
```

Each comparison row includes:

- baseline and treatment means
- delta mean
- bootstrap confidence interval bounds
- the raw and adjusted p-values

Use `compare()` when you want a direct answer to "is this model better on the
same items?" rather than a report-sized artifact.

## Build a Leaderboard

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
leaderboard = evaluation_result.leaderboard(metric_id="exact_match")

for row in leaderboard:
    print(row["model_id"], row["task_id"], row["mean"])
```

`leaderboard()` aggregates the active score rows into one summary per
`model_id`, `task_id`, and `metric_id`. Use it when you want a quick table
without building a full report object first.

For larger runs, this is often the best first read because it uses projected
score rows instead of materializing every stored artifact.

## Build and Export a Report

```python
builder = result.report()
builder.build(p_value_correction="holm")
builder.to_markdown(".cache/themis-examples/04-compare-models/report.md")
builder.to_csv(".cache/themis-examples/04-compare-models/report.csv")
```

`compare()` and `report()` read from the active `ExperimentResult` overlay. Use
`result.for_transform(...)` or `result.for_evaluation(...)` first when you want
overlay-specific summaries, comparisons, or exports.

Available convenience exporters:

- Markdown
- CSV
- LaTeX

These exporters are good for handoff artifacts. Use the leaderboard or JSON path
when you need programmatic downstream processing.

## Export the Active Overlay as JSON

```python
payload = evaluation_result.export_json(
    ".cache/themis-examples/04-compare-models/result.json"
)
print(payload["overlay"])
print(len(payload["trial_summaries"]))
```

The JSON export keeps the active overlay selection, trial summaries, score rows,
and optionally the fully hydrated trials in one portable payload.

## Database-Shaped Outputs

Themis does not ship a separate "export to database" command because the storage
backend is already the database-shaped output.

For downstream reporting or dashboarding, you have two main options:

- read the configured SQLite or Postgres projections directly
- export CSV or JSON from the active overlay for an external warehouse step

The most useful stored tables for downstream reporting are usually:

- `trial_summary`
- `candidate_summary`
- `metric_scores`

Use [Storage and Resume](../concepts/storage-and-resume.md) when you need the
broader storage model behind those tables.

## Drill Down into Edge Cases

```python
for row in evaluation_result.iter_invalid_extractions():
    print(row["candidate_id"], row["failure_reason"])

for row in evaluation_result.iter_failures():
    print(row["level"], row["message"])

for row in evaluation_result.iter_tagged_examples(tag="hallucination"):
    print(row["candidate_id"], row["tags"])
```

These helpers hydrate the affected trials and give you a small, explicit view of
bad parses, failed candidates, and tagged examples without writing a custom
projection query.

For one candidate or trial, switch from aggregate helpers to timeline views:

```python
trial = evaluation_result.get_trial(evaluation_result.trial_hashes[0])
candidate_id = trial.candidates[0].candidate_id
candidate_view = evaluation_result.view_timeline(candidate_id)

print(candidate_view.inference.raw_text)
print(candidate_view.evaluation.aggregate_scores)
```

Use this when you want to inspect the concrete response, extraction chain,
evaluation payload, or judge audit tied to a bad example.

Use [Statistical Comparisons](../concepts/statistical-comparisons.md) when you
need the interpretation behind paired outputs.

Use `examples/04_compare_models.py` for a full runnable script, and the
[Reporting & Stats API reference](../api-reference/reporting-and-stats.md) when
you need the lower-level builder, exporter, or statistical engine details.

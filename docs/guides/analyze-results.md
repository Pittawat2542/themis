# Analyze Results

Use this guide after running `examples/04_compare_models.py` or any comparable
evaluation run when you want to move from aggregate deltas into concrete failure
cases and shareable artifacts.

## Before You Start

This page assumes:

- you installed `themis-eval[stats]`
- you ran `examples/04_compare_models.py`
- you have a `result` object with at least one evaluation overlay

The example script writes its store to:

```text
.cache/themis-examples/04-compare-models/themis.sqlite3
```

## Build a Report

```python
report_builder = result.report()
report = report_builder.build(p_value_correction="holm")
print([table.id for table in report.tables])
report_builder.to_markdown(".cache/themis-examples/04-compare-models/report.md")
```

Expected output:

```text
['main_results', 'paired_comparisons']
```

## Inspect the Paired Comparison

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
comparison = evaluation_result.compare(
    metric_id="exact_match",
    p_value_correction="holm",
)
for row in comparison.rows:
    print(row.baseline_model_id, row.treatment_model_id, row.delta_mean)
```

Use [Statistical Comparisons](../concepts/statistical-comparisons.md) when you
need the interpretation behind these numbers.

## Build a Lightweight Leaderboard

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
leaderboard = evaluation_result.leaderboard(metric_id="exact_match")
print(leaderboard)
```

## Surface Invalid or Null Extractions

```python
for row in evaluation_result.iter_invalid_extractions():
    print(
        row["candidate_id"],
        row["extractor_id"],
        row["failure_reason"],
        row["warnings"],
    )
```

## Inspect Tagged Failure Cases

```python
for row in evaluation_result.iter_tagged_examples(tag="hallucination"):
    print("tagged", row["candidate_id"], row["tags"])
```

## Drill Into One Concrete Example

```python
trial = evaluation_result.get_trial(evaluation_result.trial_hashes[0])
candidate_id = trial.candidates[0].candidate_id
candidate_view = evaluation_result.view_timeline(candidate_id)

print(candidate_view.inference.raw_text)
print(candidate_view.evaluation.aggregate_scores)
```

## Inspect SQLite Summaries

Use the same DB produced by `examples/04_compare_models.py`:

```bash
themis-quickcheck failures --db .cache/themis-examples/04-compare-models/themis.sqlite3
themis-quickcheck scores --db .cache/themis-examples/04-compare-models/themis.sqlite3 --metric exact_match
themis-quickcheck latency --db .cache/themis-examples/04-compare-models/themis.sqlite3
```

When you need one specific overlay, get the hash from Python first:

```python
print(result.evaluation_hashes[0])
print(result.transform_hashes[0] if result.transform_hashes else None)
```

## Related Guides

- [Compare and Export Results](compare-and-export.md)
- [Resume and Inspect Runs](resume-and-inspect.md)
- [Use the Quickcheck CLI](quickcheck.md)

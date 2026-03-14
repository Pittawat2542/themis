# Analyze Results

In this tutorial you will extend a stored comparison run into five concrete
outputs:

- a paired comparison table in Python
- a lightweight leaderboard and drilldown view
- an exported Markdown report
- a targeted pass over invalid, null, and tagged examples
- a quick operator check from SQLite summaries

This tutorial assumes you installed the `stats` extra:

```bash
uv add "themis-eval[stats]"
```

## Before You Start

Run `examples/04_compare_models.py` once, or start from a script that already
produces a `result` object with two comparable models.

## Step 1: Build a report

```python
report_builder = result.report()
report = report_builder.build(p_value_correction="holm")
print([table.id for table in report.tables])
report_builder.to_markdown("report.md")
```

You should see tables such as `main_results` and `paired_comparisons`, and a new
`report.md` file on disk.

`ReportBuilder` assembles:

- aggregate metric tables grouped by model, task, and metric
- optional paired comparison tables when the run includes comparable trial sets
- report metadata including stored provenance summaries

## Step 2: Inspect the paired comparison

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
comparison = evaluation_result.compare(
    metric_id="exact_match",
    p_value_correction="holm",
)
for row in comparison.rows:
    print(row.baseline_model_id, row.treatment_model_id, row.delta_mean)
```

Comparisons are paired by `item_id`, so both models need score rows for the same
task/item pairs.

Each row also carries bootstrap confidence intervals and p-values, which makes
this the best place to answer "how large is the effect?" and "is the delta
statistically meaningful on the paired items?".

## Step 3: Build a lightweight leaderboard

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
leaderboard = evaluation_result.leaderboard(metric_id="exact_match")
print(leaderboard)
```

This gives you a fast aggregate view without needing to build a report first.

## Step 4: Surface invalid or null extractions

```python
for row in evaluation_result.iter_invalid_extractions():
    print(
        row["candidate_id"],
        row["extractor_id"],
        row["failure_reason"],
        row["warnings"],
    )
```

`iter_invalid_extractions()` includes both hard extraction failures and
successful extractions whose `parsed_answer` ended up null. Use it when you want
to tighten extractor edge cases without trawling the full event log.

## Step 5: Inspect tagged failure cases

```python
for row in evaluation_result.iter_tagged_examples(tag="hallucination"):
    print("tagged", row["candidate_id"], row["tags"])
```

This only works if your metrics or candidate payloads emit structured tags. It
is the recommended path for qualitative categories such as refusals,
hallucinations, or formatting problems.

## Step 6: Drill into one concrete example

```python
trial = evaluation_result.get_trial(evaluation_result.trial_hashes[0])
candidate_id = trial.candidates[0].candidate_id
candidate_view = evaluation_result.view_timeline(candidate_id)

print(candidate_view.inference.raw_text)
print(candidate_view.evaluation.aggregate_scores)
```

Switch to timeline views when aggregate helpers tell you *which* example is bad
and you now need to inspect the actual response, extraction chain, evaluation
payload, or judge audit behind it.

## Step 7: Inspect the SQLite summaries

The operator CLI reads only SQLite summaries, which makes it fast even when the
artifact store is large.

```bash
themis-quickcheck failures --db .cache/themis-docs/compare/themis.sqlite3
themis-quickcheck scores --db .cache/themis-docs/compare/themis.sqlite3 --metric exact_match --evaluation-hash <evaluation_hash>
themis-quickcheck latency --db .cache/themis-docs/compare/themis.sqlite3 --evaluation-hash <evaluation_hash>
```

## Summary

This walkthrough uses the same stored run in five different ways:

1. high-level statistical comparison through `ExperimentResult.compare()`
2. aggregate ranking through `ExperimentResult.leaderboard()`
3. qualitative drilldown through invalid extractions, tags, and timeline views
4. shareable report export through `ReportBuilder`
5. lightweight operations inspection through `themis-quickcheck`

Use the guide pages when you already know which of those tasks you want to do
and do not need the walkthrough.

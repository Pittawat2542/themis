# Analyze Results

In this tutorial you will extend a stored comparison run into three concrete
outputs:

- a paired comparison table in Python
- an exported Markdown report
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
comparison = result.compare(metric_id="exact_match", p_value_correction="holm")
for row in comparison.rows:
    print(row.baseline_model_id, row.treatment_model_id, row.delta_mean)
```

Comparisons are paired by `item_id`, so both models need score rows for the same
task/item pairs.

## Step 3: Inspect the SQLite summaries

The operator CLI reads only SQLite summaries, which makes it fast even when the
artifact store is large.

```bash
themis-quickcheck failures --db .cache/themis-docs/compare/themis.sqlite3
themis-quickcheck scores --db .cache/themis-docs/compare/themis.sqlite3 --metric exact_match
themis-quickcheck latency --db .cache/themis-docs/compare/themis.sqlite3
```

## What You Learned

You used the same stored run in three different ways:

1. high-level statistical comparison through `ExperimentResult.compare()`
2. shareable report export through `ReportBuilder`
3. lightweight operations inspection through `themis-quickcheck`

Use the guide pages when you already know which of those tasks you want to do
and do not need the walkthrough.

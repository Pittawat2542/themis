# Compare and Export Results

Install the `stats` extra first:

```bash
uv add "themis-eval[stats]"
```

## Paired Comparison

```python
comparison = result.compare(
    metric_id="exact_match",
    baseline_model_id="baseline",
    treatment_model_id="candidate",
    p_value_correction="holm",
)
```

Comparison rows are paired by `task_id`, `metric_id`, and `item_id`.

## Build and Export a Report

```python
builder = result.report()
builder.build(p_value_correction="holm")
builder.to_markdown("report.md")
builder.to_csv("report.csv")
```

Available convenience exporters:

- Markdown
- CSV
- LaTeX

Use `examples/04_compare_models.py` for a full runnable script, and the
[Reporting & Stats API reference](../api-reference/reporting-and-stats.md) when
you need the lower-level builder, exporter, or statistical engine details.

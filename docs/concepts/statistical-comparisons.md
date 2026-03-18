# Statistical Comparisons

The benchmark-first read side starts with semantic aggregation.

## Aggregate First

```python
rows = result.aggregate(
    group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
)
```

Use dimensions directly when they matter:

```python
rows = result.aggregate(group_by=["model_id", "source", "metric_id"])
```

## Pair By Benchmark Semantics

```python
comparison = result.paired_compare(
    metric_id="exact_match",
    group_by="slice_id",
)
```

This compares models on shared benchmark items within the requested grouping
key. It replaces the old public habit of thinking in `task_id` tables first.

## CLI Analogs

Use `themis-quickcheck scores` with:

- `--slice qa`
- `--dimension source=synthetic`

Those filters read the same benchmark summary fields stored in SQLite.

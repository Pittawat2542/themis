# Compare and Export Results

Use `BenchmarkResult` for benchmark-native aggregation and paired comparisons.

## Aggregate

```python
rows = result.aggregate(
    group_by=["model_id", "slice_id", "metric_id", "prompt_variant_id"]
)
```

## Pair

```python
comparison = result.paired_compare(
    metric_id="exact_match",
    group_by="slice_id",
)
```

Example output from `examples/04_compare_models.py`:

```text
[{'slice_id': 'qa', 'metric_id': 'exact_match', 'baseline_model_id': 'baseline', 'treatment_model_id': 'candidate', 'pair_count': 4, 'baseline_mean': 0.5, 'treatment_mean': 1.0, 'delta_mean': 0.5, 'p_value': 0.5, 'adjusted_p_value': 0.5, 'adjustment_method': <PValueCorrection.NONE: 'none'>, 'ci_lower': 0.0, 'ci_upper': 1.0, 'ci_level': 0.95, 'method': 'bootstrap_BCa_wilcoxon'}]
```

## Persist Operator Artifacts

```python
bundle = result.persist_artifacts(storage_root=project.storage.root_dir)
print(bundle.aggregate_json_path)
print(bundle.summary_markdown_path)
```

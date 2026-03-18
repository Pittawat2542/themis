# Analyze a Stored Run

This tutorial uses the comparison example as the stored benchmark.

## Run It

```bash
uv run python examples/04_compare_models.py
```

Output:

```text
[{'slice_id': 'qa', 'metric_id': 'exact_match', 'baseline_model_id': 'baseline', 'treatment_model_id': 'candidate', 'pair_count': 4, 'baseline_mean': 0.5, 'treatment_mean': 1.0, 'delta_mean': 0.5}]
ArtifactBundle(aggregate_json_path=PosixPath('.cache/themis-examples/04-compare-models-benchmark-first/benchmark-aggregate.json'), summary_markdown_path=PosixPath('.cache/themis-examples/04-compare-models-benchmark-first/benchmark-summary.md'))
```

## Python Read-Side

Use `BenchmarkResult` for the benchmark-native surface:

```python
from themis import BenchmarkResult

# `project` is the ProjectSpec for this workspace.
result: BenchmarkResult = orchestrator.run_benchmark(benchmark)

rows = result.aggregate(group_by=["model_id", "slice_id", "metric_id"])
comparison = result.paired_compare(metric_id="exact_match", group_by="slice_id")
bundle = result.persist_artifacts(storage_root=project.storage.root_dir)
```

## CLI Read-Side

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/04-compare-models-benchmark-first/themis.sqlite3 \
  --metric exact_match \
  --slice qa
```

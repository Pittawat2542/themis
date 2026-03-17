# Quick Start

Run the smallest shipped benchmark:

```bash
uv run python examples/01_hello_world.py
```

Expected output:

```text
{'model_id': 'demo-model', 'slice_id': 'arithmetic', 'metric_id': 'exact_match', 'source': 'synthetic', 'prompt_variant_id': 'qa-default', 'mean': 1.0, 'count': 1}
```

The run writes a SQLite database to:

```text
.cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3
```

## What The Script Covers

- one `DatasetProvider`
- one inference engine
- one metric
- one `ProjectSpec`
- one `BenchmarkSpec`
- one `BenchmarkResult.aggregate(...)` call

## Full Example

--8<-- "examples/01_hello_world.py"

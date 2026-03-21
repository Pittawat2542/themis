# Quick Start

Start with a direct smoke evaluation from the CLI:

```bash
themis quick-eval inline \
  --model demo-model \
  --provider demo \
  --input "2 + 2" \
  --expected "4" \
  --format json
```

Expected output:

```json
{
  "metric": "exact_match",
  "mode": "inline",
  "model": "demo-model",
  "prompt": "{item.input}",
  "provider": "demo",
  "rows": [
    {
      "count": 1,
      "mean": 1.0,
      "metric_id": "exact_match",
      "model_id": "demo-model",
      "prompt_variant_id": "inline-demo-model-exact-match-default",
      "slice_id": "quick-eval"
    }
  ],
  "sqlite_db": ".cache/themis/quick-eval/inline-demo-model-exact-match/themis.sqlite3",
  "storage_root": ".cache/themis/quick-eval/inline-demo-model-exact-match"
}
```

Create a catalog project when you are ready to edit a real benchmark:

```bash
themis init starter-eval
```

Or use one of the built-in benchmark definitions when you want a benchmark-aware
starter right away:

```bash
themis quick-eval benchmark \
  --benchmark mmlu_pro \
  --model demo-model \
  --provider demo \
  --preview \
  --format json
```

```bash
themis init starter-mmlu --benchmark mmlu_pro
```

That scaffold includes:

- `project.toml` for storage and execution policy
- a runnable package under `starter_eval/`
- `data/sample.jsonl`
- README examples for `themis quickcheck` and `themis report`

To inspect a dataset before wiring a new built-in, use:

```bash
uv run python scripts/inspect_huggingface_dataset.py TIGER-Lab/MMLU-Pro --split test
```

## Go Deeper

Run the smallest shipped Python benchmark when you want to study the full code-first authoring loop:

```bash
uv run python examples/01_hello_world.py
```

Expected output:

```text
{'model_id': 'demo-model', 'slice_id': 'arithmetic', 'metric_id': 'exact_match', 'source': 'synthetic', 'prompt_variant_id': 'qa-default', 'mean': 1.0, 'count': 1}
```

The example run writes a SQLite database to:

```text
.cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3
```

What the example script covers:

- one `DatasetProvider`
- one inference engine
- one metric
- one `ProjectSpec`
- one `BenchmarkSpec`
- one `BenchmarkResult.aggregate(...)` call

## Full Example Script

--8<-- "examples/01_hello_world.py"

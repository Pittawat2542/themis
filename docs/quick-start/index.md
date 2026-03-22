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

That default scaffold includes:

- `project.toml` for storage and execution policy
- a runnable package under `starter_eval/`
- `data/sample.jsonl`
- README examples for `themis quickcheck` and `themis report`

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

That builtin scaffold includes:

- `project.toml` for storage and execution policy
- `.env.example` with provider, model, benchmark, and judge settings
- a runnable package under `starter_mmlu/`
- builtin benchmark wiring via `themis.catalog.get_catalog_benchmark(...)`
- README examples for `themis quickcheck` and `themis report`

The highlighted benchmarks above are examples, not the full catalog. Discover
all shipped ids from Python with `list_catalog_benchmarks()`, or see the
[Builtin Benchmarks guide](../guides/builtin-benchmarks.md):

```python
from themis.catalog import list_catalog_benchmarks

print(list_catalog_benchmarks())
```

The built-in catalog also includes short-answer math benchmarks such as
`aime_2026`, `aime_2025`, `hmmt_feb_2025`, `hmmt_nov_2025`, `apex_2025`,
`beyond_aime`, and `imo_answerbench`:

```bash
themis quick-eval benchmark \
  --benchmark aime_2026 \
  --model demo-model \
  --provider demo \
  --preview \
  --format json
```

Use `build_catalog_benchmark_project(...)` when you want the Python entry point
for builtin benchmarks instead of the CLI. The full catalog list and related
examples live in the [Builtin Benchmarks guide](../guides/builtin-benchmarks.md).

Minimal example:

```python
from pathlib import Path

from themis import Orchestrator
from themis.catalog import build_catalog_benchmark_project


def load_fixture_rows(dataset_id: str, split: str, revision: str | None):
    del revision
    assert dataset_id == "TIGER-Lab/MMLU-Pro"
    assert split == "test"
    return [
        {
            "item_id": "mmlu-pro-1",
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Mercury"],
            "answer": "B",
            "answer_index": 1,
            "category": "astronomy",
            "src": "fixture",
        }
    ]


project, benchmark, registry, dataset_provider, definition = (
    build_catalog_benchmark_project(
        benchmark_id="mmlu_pro",
        model_id="demo-model",
        provider="demo",
        storage_root=Path(".cache/themis-examples/catalog-snippet"),
        huggingface_loader=load_fixture_rows,
    )
)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_provider=dataset_provider,
)
result = orchestrator.run_benchmark(benchmark)
print(result.aggregate(group_by=["model_id", "slice_id", "metric_id"]))
print(definition.summarize_result(result))
```

To inspect a dataset before wiring a new built-in, use:

```bash
uv run python scripts/inspect_huggingface_dataset.py TIGER-Lab/MMLU-Pro --split test
```

You can inspect multiple datasets in one pass:

```bash
uv run python scripts/inspect_huggingface_dataset.py \
  MathArena/aime_2026:train \
  ByteDance-Seed/BeyondAIME:test \
  Hwilner/imo-answerbench:train
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

For the catalog-specific library path, run:

```bash
uv run python examples/13_catalog_builtin_benchmark.py
```

## Full Example Script

--8<-- "examples/01_hello_world.py"

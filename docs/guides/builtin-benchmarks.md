# Builtin Benchmarks

Use the builtin catalog when you want a shipped benchmark definition instead of
authoring one from scratch.

The public Python entry points are:

- `themis.catalog.list_catalog_benchmarks()`
- `themis.catalog.get_catalog_benchmark(...)`
- `themis.catalog.build_catalog_benchmark_project(...)`

Discover all shipped ids from Python:

```python
from themis.catalog import list_catalog_benchmarks

print(list_catalog_benchmarks())
```

Current builtin benchmarks:

| Benchmark | Shape | Metric | Judge | Dataset | Split |
| --- | --- | --- | --- | --- | --- |
| `aime_2025` | Short-answer math | `math_equivalence` | No | `MathArena/aime_2025` | `train` |
| `aime_2026` | Short-answer math | `math_equivalence` | No | `MathArena/aime_2026` | `train` |
| `apex_2025` | Short-answer math | `math_equivalence` | No | `MathArena/apex_2025` | `train` |
| `beyond_aime` | Short-answer math | `math_equivalence` | No | `ByteDance-Seed/BeyondAIME` | `test` |
| `encyclo_k` | Multiple choice | `choice_accuracy` | No | `m-a-p/Encyclo-K` | `test` |
| `healthbench` | Rubric-scored response | `healthbench_score` | Yes | `openai/healthbench` | `test` |
| `hle:<variant>[,<variant>...]` | Judge-backed QA | `hle_accuracy` | Yes | `cais/hle` | `test` |
| `hmmt_feb_2025` | Short-answer math | `math_equivalence` | No | `MathArena/hmmt_feb_2025` | `train` |
| `hmmt_nov_2025` | Short-answer math | `math_equivalence` | No | `MathArena/hmmt_nov_2025` | `train` |
| `imo_answerbench` | Short-answer math | `math_equivalence` | No | `Hwilner/imo-answerbench` | `train` |
| `lpfqa` | Judge-backed free-form QA | `lpfqa_score` | Yes | `m-a-p/LPFQA` | `train` |
| `mmlu_pro` | Multiple choice | `choice_accuracy` | No | `TIGER-Lab/MMLU-Pro` | `test` |
| `simpleqa_verified` | Judge-backed short answer | `simpleqa_verified_score` | Yes | `google/simpleqa-verified` | `eval` |
| `supergpqa` | Multiple choice | `choice_accuracy` | No | `m-a-p/SuperGPQA` | `train` |

Use `get_catalog_benchmark(...)` when you want to inspect or render one
definition directly:

```python
from themis.catalog import get_catalog_benchmark

definition = get_catalog_benchmark("mmlu_pro")
preview = definition.render_preview(model_id="demo-model", provider="demo")
print(preview[0]["messages"][0]["content"])
```

HLE requires explicit variants in the benchmark id. Example ids:

- `hle:text_only`
- `hle:no_tool`
- `hle:text_only,no_tool`

Use `build_catalog_benchmark_project(...)` when you want the full runnable
Python path for a builtin benchmark:

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

For a runnable script version, see `examples/13_catalog_builtin_benchmark.py`.

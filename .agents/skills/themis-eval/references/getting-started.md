# Getting Started

## Install The Package

```bash
uv add themis-eval
```

Add extras only when the workflow needs them:

- `stats`
- `compression`
- `extractors`
- `datasets`
- `providers-openai`, `providers-litellm`, `providers-vllm`
- `telemetry`
- `storage-postgres`

## Pick The Fastest Entry Point

Start with the smallest path that answers the user:

- smoke test one prompt: `themis quick-eval inline ...`
- scaffold a local project: `themis init starter-eval`
- start from a shipped benchmark definition:
  `themis quick-eval benchmark ...` or `themis init ... --benchmark <id>`
- write Python directly when the benchmark or runtime needs custom logic

For example:

```bash
themis quick-eval inline \
  --model demo-model \
  --provider demo \
  --input "2 + 2" \
  --expected "4" \
  --format json
```

```bash
themis quick-eval benchmark \
  --benchmark mmlu_pro \
  --model demo-model \
  --provider demo \
  --preview \
  --format json
```

If the user does not want to run `themis init` but still wants the same ideal
project shape, either:

- read `references/project-structure.md`
- run `scripts/generate_project_structure.py` to materialize the layout in a target folder

## Use The Core Mental Model

The normal user workflow is:

1. implement a `DatasetProvider`
2. register the minimum plugin set
3. define `ProjectSpec`
4. define `BenchmarkSpec`
5. build `Orchestrator`
6. run and inspect `BenchmarkResult`

When the benchmark already exists in the shipped catalog, prefer that over
rebuilding it from scratch.

## Start From This Bundled Pattern

```python
from pathlib import Path

from themis import (
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    StorageSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore
from themis.specs import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class DemoProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [{"item_id": "item-1", "question": "2 + 2", "answer": "4"}]


class DemoEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=context["answer"],
            )
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
        )


registry = PluginRegistry()
registry.register_inference_engine("demo", DemoEngine())
registry.register_metric("exact_match", ExactMatchMetric())

project = ProjectSpec(
    project_name="hello-world",
    researcher_id="docs",
    global_seed=7,
    storage=StorageSpec(
        root_dir=str(Path(".cache/themis-examples/01-hello-world-benchmark-first")),
        compression=CompressionCodec.NONE,
    ),
    execution_policy=ExecutionPolicySpec(),
)

benchmark = BenchmarkSpec(
    benchmark_id="hello-world",
    models=[ModelSpec(model_id="demo-model", provider="demo")],
    slices=[
        SliceSpec(
            slice_id="arithmetic",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            dataset_query=DatasetQuerySpec.subset(1, seed=7),
            dimensions={"source": "synthetic", "format": "qa"},
            generation=GenerationSpec(),
            prompt_variant_ids=["baseline"],
            scores=[ScoreSpec(name="default", metrics=["exact_match"])],
        )
    ],
    prompt_variants=[
        PromptVariantSpec(
            id="baseline",
            family="qa",
            messages=[
                PromptMessage(
                    role=PromptRole.USER,
                    content="Solve the arithmetic problem.",
                )
            ],
        )
    ],
    inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_provider=DemoProvider(),
)
result = orchestrator.run_benchmark(benchmark)

for row in result.aggregate(
    group_by=["model_id", "slice_id", "metric_id", "source", "prompt_variant_id"]
):
    print(row)
```

Use an explicit `seed=` when you want a reproducible sampled subset. If you
omit the seed for count-based sampling, Themis keeps deterministic order-based
selection from the provider instead of randomizing.

## Use The Quick Authoring Helpers

`BenchmarkSpec.simple(...)` is the fastest Python path for a small benchmark:

```python
benchmark = BenchmarkSpec.simple(
    benchmark_id="arithmetic-quick",
    model=ModelSpec(model_id="demo-model", provider="demo"),
    dataset_rows=[{"item_id": "item-1", "question": "2 + 2", "answer": "4"}],
    prompt_template="What is {question}?",
    metric_id="exact_match",
)
```

Use `BenchmarkSpec.preview(item)` before running when the user wants to inspect
rendered prompts:

```python
preview = benchmark.preview({"question": "2 + 2", "answer": "4"})
print(preview["arithmetic-quick-default"]["messages"][0]["content"])
```

## Add Built-In Progress Logging When Needed

```python
from themis.progress import ProgressConfig, ProgressRendererType

result = orchestrator.run_benchmark(
    benchmark,
    progress=ProgressConfig(renderer=ProgressRendererType.LOG),
)
```

## Use Built-In Benchmark Catalog Projects When Possible

Reach for the shipped benchmark catalog when the user wants standard benchmark
definitions such as `mmlu_pro`, `aime_2026`, `codeforces`, or `livecodebench`.

```python
from pathlib import Path

from themis.catalog import build_catalog_benchmark_project

project, benchmark, registry, dataset_loader = build_catalog_benchmark_project(
    benchmark_id="mmlu_pro",
    model_id="demo-model",
    provider="demo",
    storage_root=Path(".cache/themis-examples/catalog-snippet"),
)
```

## Move To The Agent Pattern Only When Needed

If the user needs bootstrap `system` or `developer` messages, scripted
follow-up turns, first-class local tool passing, or OpenAI-hosted MCP server
selection, keep the same benchmark-first flow and then switch to
`references/agent-evals-and-tools.md`.

The canonical advanced examples are `examples/10_agent_eval.py` for local tools
and `examples/14_mcp_openai.py` for OpenAI-hosted MCP servers.

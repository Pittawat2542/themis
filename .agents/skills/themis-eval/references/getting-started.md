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

## Use The Core Mental Model

The normal user workflow is:

1. implement a `DatasetProvider`
2. register the minimum plugin set
3. define `ProjectSpec`
4. define `BenchmarkSpec`
5. build `Orchestrator`
6. run and inspect `BenchmarkResult`

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
    storage=StorageSpec(root_dir=str(Path(".cache/themis/hello-world"))),
    execution_policy=ExecutionPolicySpec(),
)

benchmark = BenchmarkSpec(
    benchmark_id="hello-world",
    models=[ModelSpec(model_id="demo-model", provider="demo")],
    slices=[
        SliceSpec(
            slice_id="arithmetic",
            dataset=DatasetSpec(source="memory"),
            dataset_query=DatasetQuerySpec.subset(1, seed=7),
            generation=GenerationSpec(),
            prompt_variant_ids=["baseline"],
            scores=[ScoreSpec(name="default", metrics=["exact_match"])],
        )
    ],
    prompt_variants=[
        PromptVariantSpec(
            id="baseline",
            family="qa",
            messages=[PromptMessage(role="user", content="Solve the arithmetic problem.")],
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
```

## Add Built-In Progress Logging When Needed

```python
from themis.progress import ProgressConfig, ProgressRendererType

result = orchestrator.run_benchmark(
    benchmark,
    progress=ProgressConfig(renderer=ProgressRendererType.LOG),
)
```

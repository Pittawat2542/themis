# Getting Started

## Install The Package

Use the base package for core orchestration:

```bash
uv add themis-eval
```

Add extras only when the workflow needs them:

- `stats`: paired comparisons, report tables, leaderboard helpers
- `compression`: compressed artifact storage such as judge-audit examples
- `extractors`: built-in `json_schema` extractor
- `datasets`: Hugging Face datasets integrations
- `providers-openai`, `providers-litellm`, `providers-vllm`: SDKs for your own
  engine implementations
- `telemetry`: Langfuse or Weights & Biases callbacks; not required for the
  built-in progress/logging surface
- `storage-postgres`: Postgres-backed storage

## Use The Core Mental Model

The normal user workflow is:

1. implement a dataset loader
2. implement or register the minimum plugin set
3. define `ProjectSpec`
4. define `ExperimentSpec`
5. build `Orchestrator`
6. run and inspect `ExperimentResult`

The smallest useful object set is:

- `ProjectSpec`
- `ExperimentSpec`
- `PluginRegistry`
- `Orchestrator`
- `ExperimentResult`

Operator-facing progress logging is part of the base package through
`themis.progress`; treat it as separate from optional telemetry integrations.

## Start From This Bundled Pattern

Reuse this shape even when the user does not have the Themis source tree,
examples directory, or local docs available:

```python
from pathlib import Path

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    GenerationSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptTemplateSpec,
    StorageSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore


class DemoLoader:
    def load_task_items(self, task):
        del task
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
        expected = str(context["answer"])
        return MetricScore(
            metric_id="exact_match",
            value=float(actual.strip() == expected),
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

experiment = ExperimentSpec(
    models=[ModelSpec(model_id="demo-model", provider="demo")],
    tasks=[
        TaskSpec(
            task_id="arithmetic",
            dataset=DatasetSpec(source="memory"),
            generation=GenerationSpec(),
            evaluations=[EvaluationSpec(name="default", metrics=["exact_match"])],
        )
    ],
    prompt_templates=[
        PromptTemplateSpec(
            id="baseline",
            messages=[PromptMessage(role="user", content="Solve the arithmetic problem.")],
        )
    ],
    inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_loader=DemoLoader(),
)
result = orchestrator.run(experiment)
```

## Add Built-In Progress Logging When Needed

Use the `progress=` entrypoint argument when the user wants live status,
callback snapshots, or stdlib logging:

```python
from themis.progress import ProgressConfig, ProgressRendererType, ProgressVerbosity

result = orchestrator.run(
    experiment,
    progress=ProgressConfig(
        renderer=ProgressRendererType.LOG,
        verbosity=ProgressVerbosity.DEBUG,
    ),
)
```

Use `callback=...` to collect `RunProgressSnapshot` values in memory. When you
set only a callback, Themis does not attach the Rich terminal renderer unless
you explicitly request one.

## Pick The Nearest Pattern

- Use `references/advanced-workflows.md` when project policy should live in
  TOML.
- Use `references/plugins-and-specs.md` when raw model text is not the final
  scoring surface.
- Use `references/results-and-ops.md` when the user needs statistical
  comparison and reports.
- Use `references/results-and-ops.md` when the task is reruns, incremental
  work, live progress updates, or persisted run snapshots.
- Use `references/plugins-and-specs.md` when the task is prompt mutation,
  instrumentation, or judge-backed metrics.
- Use `references/advanced-workflows.md` when generation or scoring happens
  outside Themis, when the user is adding metrics, prompts, or models later, or
  when they need telemetry sinks such as Langfuse.

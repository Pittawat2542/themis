# Hello World Walkthrough

In this tutorial you will build a complete Themis script from scratch and end up
with:

- a local SQLite-backed experiment store
- two executed trials
- one `ExperimentResult` object you can inspect in Python

This is the teaching version of the [Quick Start](../quick-start/index.md). The
Quick Start is optimized for speed; this page is optimized for understanding.

## Before You Start

Make sure `themis-eval` is installed and create a new file, for example
`hello_world.py`.

## Step 1: Add imports and a dataset loader

Start with the write-side building blocks and a tiny in-memory dataset loader:

```python
from pathlib import Path

from themis import (
    DatasetSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
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
from themis.records.evaluation import MetricScore
from themis.records.inference import InferenceRecord


class DemoDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item-1", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-2", "question": "6 * 7", "answer": "42"},
        ]
```

At this point you have not configured any runtime behavior yet. You have only
defined the data that trial planning will expand over.

## Step 2: Add an engine and a metric

Now add the smallest useful plugin set:

```python
class DemoEngine:
    def infer(self, trial, context, runtime):
        del runtime
        answer = "4" if context["question"] == "2 + 2" else "42"
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inference_{trial.item_id}",
                raw_text=answer,
                latency_ms=2,
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
            details={"actual": actual, "expected": expected},
        )
```

The engine turns one planned trial plus one dataset item into an
`InferenceRecord`. The metric turns the resulting candidate into a scalar score.

## Step 3: Register the plugins

`PluginRegistry` is the runtime lookup table for engines, metrics, extractors,
judges, and hooks:

```python
registry = PluginRegistry()
registry.register_inference_engine("demo", DemoEngine())
registry.register_metric("exact_match", ExactMatchMetric())
```

The registry is instance-scoped, so test suites and separate experiments can
carry different plugin sets without leaking global state.

## Step 4: Declare project-level policy

`ProjectSpec` is where you put:

- the experiment storage root
- retry and circuit-breaker policy
- researcher and project identity
- global deterministic seed

```python
project = ProjectSpec(
    project_name="hello-world",
    researcher_id="docs",
    global_seed=7,
    storage=StorageSpec(
        root_dir=str(Path(".cache/themis-docs/hello-world")),
        compression="none",
    ),
    execution_policy=ExecutionPolicySpec(),
)
```

This keeps storage and execution rules out of the experiment matrix.

## Step 5: Declare the experiment matrix

`ExperimentSpec` describes:

- the models to try
- the tasks to run
- the prompt templates to attach
- the inference parameter grid
- how many candidates to generate per trial

```python
experiment = ExperimentSpec(
    models=[ModelSpec(model_id="demo-model", provider="demo")],
    tasks=[
        TaskSpec(
            task_id="arithmetic",
            dataset=DatasetSpec(source="memory"),
            default_metrics=["exact_match"],
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
```

When execution begins, the planner combines this matrix with the dataset items to
produce deterministic `TrialSpec` objects.

## Step 6: Run the experiment

Now connect the write-side pieces and run them:

```python
orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_loader=DemoDatasetLoader(),
)
result = orchestrator.run(experiment)

for trial in result.iter_trials():
    print(trial.trial_spec.item_id)
    print(trial.candidates[0].evaluation.aggregate_scores)
```

Expected output:

```text
item-1 {'exact_match': 1.0}
item-2 {'exact_match': 1.0}
```

## Step 7: Inspect a stored projection

`ExperimentResult` is the read-side facade over stored projections:

```python
trial = result.get_trial(result.trial_hashes[0])
timeline = result.view_timeline(trial.spec_hash, record_type="trial")
```

You should now have:

- stored specs and events on disk
- hydrated `TrialRecord` projections
- a timeline view for later inspection

## What You Learned

You used the full Themis loop:

1. declare project policy
2. declare an experiment matrix
3. register runtime plugins
4. run through the orchestrator
5. inspect stored results from the read side

Move on to [Load a Project File](project-files.md) once you want a reusable
configuration file instead of inline `ProjectSpec` construction.

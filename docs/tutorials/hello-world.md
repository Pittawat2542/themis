# Hello World Walkthrough

This tutorial builds a complete Themis script from scratch. The finished script
includes:

- a local SQLite-backed experiment store
- two executed trials
- one `ExperimentResult` object you can inspect in Python

For a faster path, use the [Quick Start](../quick-start/index.md). This page
walks through the same workflow step by step.

## Before You Start

Make sure `themis-eval` is installed and create a new file, for example
`hello_world.py`.

You will run it with:

```bash
uv run python hello_world.py
```

## Step 1: Add imports and a dataset loader

Start with the write-side building blocks and a tiny in-memory dataset loader:

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
    SqliteBlobStorageSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore


class DemoDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item-1", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-2", "question": "6 * 7", "answer": "42"},
        ]
```

This step defines the data that trial planning expands over.

## Step 2: Add an engine and a metric

Add the smallest useful plugin set:

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
    storage=SqliteBlobStorageSpec(root_dir=str(Path(".cache/themis-examples/01-hello-world")), compression="none"),
    execution_policy=ExecutionPolicySpec(),
)
```

This keeps storage and execution rules out of the experiment matrix. `StorageSpec`
remains the SQLite compatibility alias, but the concrete
`SqliteBlobStorageSpec` name is the preferred teaching path for new code.

## Step 5: Declare the experiment matrix

`ExperimentSpec` describes:

- the models to try
- the tasks to run
- the generation / transform / evaluation stages inside each task
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
```

When execution begins, the planner combines this matrix with the dataset items to
produce deterministic `TrialSpec` objects.

## Step 6: Run the experiment

Connect the write-side pieces and run them:

```python
orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_loader=DemoDatasetLoader(),
)
result = orchestrator.run(experiment)

print("Stored SQLite database:", ".cache/themis-examples/01-hello-world/themis.sqlite3")
for trial in result.iter_trials():
    score = trial.candidates[0].evaluation.aggregate_scores["exact_match"]
    print(f"{trial.trial_spec.item_id}: exact_match={score:.1f}")
```

`run()` executes the full three-stage flow: generation first, then any declared
output transforms, then any declared evaluations.

Expected output:

```text
Stored SQLite database: .cache/themis-examples/01-hello-world/themis.sqlite3
item-1: exact_match=1.0
item-2: exact_match=1.0
```

## Step 7: Inspect a stored projection

`ExperimentResult` is the read-side facade over stored projections:

```python
trial = result.get_trial(result.trial_hashes[0])
timeline = result.view_timeline(trial.spec_hash, record_type="trial")
```

The run produces:

- stored specs and events on disk
- hydrated `TrialRecord` projections
- a timeline view for later inspection

## Summary

This walkthrough covers the full Themis loop:

1. declare project policy
2. declare an experiment matrix
3. register runtime plugins
4. run through the orchestrator
5. inspect stored results from the read side

Move on to [Load a Project File](project-files.md) once you want a reusable
configuration file instead of inline `ProjectSpec` construction.

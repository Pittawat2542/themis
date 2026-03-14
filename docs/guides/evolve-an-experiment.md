# Evolve an Experiment

Use this guide when a run already exists and you want to add more work without
starting over from scratch.

## The Core Rule

Themis hashes the full experiment matrix into deterministic stage identities.
Unchanged trial, transform, and evaluation hashes are reused; only new hashes
remain pending.

That means:

- add a model or prompt: new trial hashes
- add inference params: new trial hashes for the new parameter combinations
- add a metric: existing generation can stay, but new evaluation overlays appear
- add an output transform: existing generation can stay, but new transform and
  evaluation overlays appear
- change the dataset slice: the planned trial set changes deterministically

## See What Will Change Before Running

Use `diff_specs()` to compare a baseline experiment with a new one:

```python
diff = orchestrator.diff_specs(baseline_experiment, treatment_experiment)

print(diff.changed_experiment_fields)
print(len(diff.added_trial_hashes))
print(len(diff.added_evaluation_hashes))
```

This is the fastest way to confirm that a change only affects the stage you
intended.

## Add a New Model or Prompt Later

Models and prompt templates live directly in `ExperimentSpec`, so adding either
one expands the experiment matrix with new trial hashes.

```python
expanded = experiment.model_copy(
    update={
        "models": [
            *experiment.models,
            ModelSpec(model_id="new-model", provider="demo"),
        ],
        "prompt_templates": [
            *experiment.prompt_templates,
            PromptTemplateSpec(
                id="few-shot",
                messages=[
                    PromptMessage(role="system", content="Answer with one integer."),
                    PromptMessage(role="user", content="Q: 1 + 1\nA: 2"),
                    PromptMessage(role="user", content="Now solve the next problem."),
                ],
            ),
        ],
    }
)
```

On rerun, only the newly introduced model/prompt combinations stay pending.

## Add a New Metric Without Regenerating

Metrics sit inside `EvaluationSpec`, so you can extend evaluation after
generation already finished:

```python
expanded = experiment.model_copy(
    update={
        "tasks": [
            experiment.tasks[0].model_copy(
                update={
                    "evaluations": [
                        EvaluationSpec(
                            name="default",
                            metrics=["exact_match", "answer_length"],
                        )
                    ]
                }
            )
        ]
    }
)
```

If the generation overlay is already in storage, Themis reuses it and only
materializes the missing evaluation overlay.

## Slice a Benchmark Deterministically

`ItemSamplingSpec` lets you change the benchmark slice without editing the
dataset loader itself:

```python
from themis import ItemSamplingSpec


hard_only = experiment.model_copy(
    update={
        "item_sampling": ItemSamplingSpec(
            kind="subset",
            count=100,
            seed=7,
            metadata_filters={"difficulty": "hard"},
        )
    }
)
```

Use:

- `item_ids=[...]` for a fixed regression list
- `metadata_filters={...}` for a named category such as `difficulty=hard`
- `kind="subset"` or `kind="stratified"` for deterministic sampling on top of
  those filters

## Sweep Prompt Templates and Zero/Few-Shot Variants

Zero-shot and few-shot live naturally as different `PromptTemplateSpec`
instances. You do not need to rewrite benchmark logic to compare them.

```python
experiment = ExperimentSpec(
    ...,
    prompt_templates=[
        PromptTemplateSpec(
            id="zero-shot",
            messages=[PromptMessage(role="user", content="Solve the problem.")],
        ),
        PromptTemplateSpec(
            id="few-shot",
            messages=[
                PromptMessage(role="user", content="Q: 2 + 2\nA: 4"),
                PromptMessage(role="user", content="Solve the next problem."),
            ],
        ),
    ],
)
```

Each prompt template becomes part of the deterministic trial identity, so
existing zero-shot trials stay reusable while the few-shot trials are new work.

## Sweep Generation Parameters Without Repeating Unchanged Work

Use `InferenceGridSpec` to expand one or more base parameter sets over a grid of
overrides:

```python
experiment = experiment.model_copy(
    update={
        "inference_grid": InferenceGridSpec(
            params=[InferenceParamsSpec(max_tokens=64, seed=17)],
            overrides={"temperature": [0.0, 0.7], "top_p": [0.9, 1.0]},
        )
    }
)
```

Only the new parameter combinations produce new trial hashes. Unchanged prompt /
model / task / params combinations still resume normally.

## Code as Config, with Optional Project Files

Themis is intentionally code-first for the experiment matrix:

- `ExperimentSpec` is authored in Python
- `ProjectSpec` can live in Python or in `project.toml` / `project.json`

That split is deliberate:

- project files are good for shared storage, retry, and backend policy
- Python is the intended surface for prompt sweeps, custom metrics, and dynamic
  experiment composition

Use [Author Project Files](project-files.md) when you want a reusable
project-level config file.

## Version for Reproducibility

To reproduce the same run months later, version all inputs that affect the
hashes:

- the Python code that builds `ExperimentSpec`
- the `project.toml` or inline `ProjectSpec`
- dataset identity and `DatasetSpec.revision` when applicable
- model/provider IDs and any provider `extras`
- prompt templates and inference params

Themis also persists a `RunManifest` snapshot, so the stored run keeps the exact
project and experiment payload used for planning.

## Seeds and Determinism

Use both project-level and inference-level seeds when your engine supports them:

```python
project = ProjectSpec(..., global_seed=7, ...)
params = InferenceParamsSpec(max_tokens=64, temperature=0.0, seed=7)
```

`global_seed` keeps Themis-side planning and sampling deterministic. Provider-
level determinism still depends on whether your engine or API honors
`InferenceParamsSpec.seed` and `temperature=0.0`.

## Detect Completed Work Instead of Recomputing It

Use `plan()` or `submit()` to inspect what is still pending:

```python
manifest = orchestrator.plan(expanded)
pending = [item for item in manifest.work_items if item.status.value == "pending"]
print(len(pending))
```

If the same `(model, task, prompt, params, slice)` combination was already
materialized, Themis skips it. The built-in behavior is skip-on-match, not a
separate warning-only mode.

Use `examples/09_experiment_evolution.py` for a runnable example that adds
metrics, prompts, and models across repeated runs.

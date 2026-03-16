# Results And Operations

## Inspect Stored Results In Python

Start with the returned `ExperimentResult`:

```python
for trial in result.iter_trials():
    print(trial.trial_spec.item_id)
    print(trial.candidates[0].evaluation.aggregate_scores)
```

For one concrete example:

```python
trial = result.get_trial(result.trial_hashes[0])
candidate_id = trial.candidates[0].candidate_id
candidate_view = result.view_timeline(candidate_id)

print(candidate_view.inference.raw_text)
print(candidate_view.evaluation.aggregate_scores)
```

Use the trial view for stage timing and the candidate view for concrete
inference, extraction, evaluation, judge, and observability payloads.

## Compare Models And Export Reports

These features need the `stats` extra:

```bash
uv add "themis-eval[stats]"
```

Pick the active overlay first when multiple evaluations exist:

```python
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
```

Paired comparison:

```python
comparison = evaluation_result.compare(
    metric_id="exact_match",
    baseline_model_id="baseline",
    treatment_model_id="candidate",
    p_value_correction="holm",
)
```

Leaderboard:

```python
leaderboard = evaluation_result.leaderboard(metric_id="exact_match")
```

Report export:

```python
builder = result.report()
builder.build(p_value_correction="holm")
builder.to_markdown("report.md")
builder.to_csv("report.csv")
```

Use `compare()` for a direct answer, `leaderboard()` for a quick aggregate
table, and `report()` for a handoff artifact.

## Export Project And Experiment Config Reports

Use config reports when the user wants a reproducibility snapshot of the exact
project and experiment setup, wants to inspect defaults and source metadata, or
asks for a paper-ready config appendix.

Prefer the root helper when both objects are already in memory:

```python
from pathlib import Path

from themis import generate_config_report

bundle = {"project": project, "experiment": experiment}

markdown_report = generate_config_report(bundle, format="markdown")
full_json_report = generate_config_report(
    bundle,
    format="json",
    verbosity="full",
)
generate_config_report(
    bundle,
    format="latex",
    output=Path("config-report.tex"),
)
```

Supported built-in formats are `json`, `yaml`, `markdown`, and `latex`.
`verbosity="default"` keeps the paper-facing subset. `verbosity="full"` keeps
the complete collected tree for audits and debugging.

Use the CLI when the user wants an artifact from a config factory or a
persisted run:

```bash
themis report \
  --factory my_package.evals.paper_run:build_config_bundle \
  --format markdown \
  --output config-report.md

themis report \
  --project-file project.json \
  --run-id run_123 \
  --format latex \
  --verbosity full \
  --output config-report.tex
```

Use factory mode when config lives in Python. Use `--project-file` plus
`--run-id` when the user wants the persisted manifest snapshot for an existing
run.

## Surface Failures And Tagged Examples

Use aggregate helpers before drilling into timelines:

```python
for row in evaluation_result.iter_invalid_extractions():
    print(row["candidate_id"], row["failure_reason"])

for row in evaluation_result.iter_failures():
    print(row["level"], row["message"])

for row in evaluation_result.iter_tagged_examples(tag="hallucination"):
    print(row["candidate_id"], row["tags"])
```

These helpers are the fastest way to inspect weak parses, failed candidates,
and qualitative categories emitted by metric details.

## Resume, Plan, Submit, And Estimate

Themis skips completed work when storage, specs, and stage identity still
match.

Inspect pending work first:

```python
manifest = orchestrator.plan(experiment)
print(manifest.run_id)
print(sum(item.status == "pending" for item in manifest.work_items))
```

Async-friendly flow:

```python
handle = orchestrator.submit(experiment, runtime=runtime)
print(handle.run_id, handle.status, handle.pending_work_items)

resumed = orchestrator.resume(handle.run_id, runtime=runtime)
```

Estimate work:

```python
estimate = orchestrator.estimate(experiment)
print(estimate.total_work_items)
print(estimate.work_items_by_stage)
print(estimate.estimated_total_tokens)
```

Diff two experiment specs before running:

```python
diff = orchestrator.diff_specs(baseline_experiment, treatment_experiment)
print(diff.changed_experiment_fields)
print(diff.added_trial_hashes[:3])
```

## Use The Quickcheck CLI For Fast SQLite Inspection

The CLI surface is the SQLite summary inspector:

```bash
themis-quickcheck failures --db .cache/themis/hello-world/themis.sqlite3 --limit 20
themis-quickcheck scores --db .cache/themis/hello-world/themis.sqlite3 --metric exact_match
themis-quickcheck latency --db .cache/themis/hello-world/themis.sqlite3
```

Use `--evaluation-hash` or `--transform-hash` when the user needs one overlay
instead of the default generation view.

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

## Track Live Progress And Log Run State

Use the public `themis.progress` surface when the user asks for operator-facing
logging, callback snapshots, or terminal progress:

```python
from themis.progress import ProgressConfig, ProgressRendererType, ProgressVerbosity

snapshots = []
result = orchestrator.run(
    experiment,
    runtime=runtime,
    progress=ProgressConfig(
        renderer=ProgressRendererType.LOG,
        verbosity=ProgressVerbosity.DEBUG,
        callback=snapshots.append,
    ),
)

print(snapshots[0].remaining_items)
print(snapshots[-1].processed_items)
```

`run()`, `generate()`, `transform()`, `evaluate()`, `submit()`, and `resume()`
all accept `progress=`. The callback receives `RunProgressSnapshot` values.
Built-in progress logging does not require the `telemetry` extra.

If the user wants terminal rendering instead of stdlib logging, switch the
renderer:

```python
from themis.progress import ProgressConfig, ProgressRendererType

result = orchestrator.run(
    experiment,
    progress=ProgressConfig(renderer=ProgressRendererType.RICH),
)
```

When a callback is provided without `renderer=...`, Themis stays callback-only
and does not attach the Rich renderer automatically.

## Inspect Persisted Run Progress

Use `get_run_progress(run_id)` when the user needs the canonical stored
snapshot for a run handle:

```python
from themis.types.enums import RunStage

handle = orchestrator.submit(experiment, runtime=runtime)
snapshot = orchestrator.get_run_progress(handle.run_id)

print(snapshot.active_stage)
print(snapshot.processed_items, snapshot.remaining_items)
print(snapshot.stage_counts[RunStage.EVALUATION].failed_items)
```

The returned snapshot covers the full run for that `run_id`, even if the work
was started through `generate()`, `transform()`, or `evaluate()`.

## Use The Quickcheck CLI For Fast SQLite Inspection

The CLI surface is the SQLite summary inspector:

```bash
themis-quickcheck failures --db .cache/themis/hello-world/themis.sqlite3 --limit 20
themis-quickcheck scores --db .cache/themis/hello-world/themis.sqlite3 --metric exact_match
themis-quickcheck latency --db .cache/themis/hello-world/themis.sqlite3
```

Use `--evaluation-hash` or `--transform-hash` when the user needs one overlay
instead of the default generation view.

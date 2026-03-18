# Resume and Inspect Runs

## Resume a Run

Reuse the same:

- `ProjectSpec.storage.root_dir`
- benchmark or experiment source spec
- active stage configuration

When a completed projection already exists for that trial hash and overlay, the
executor skips the work instead of recomputing it. Generation resume is keyed by
`trial_hash`; transform and evaluation resume are keyed by deterministic
`transform_hash` and `evaluation_hash`.

Only successful overlays are skipped. If a transform or evaluation overlay
fails, rerunning the same experiment executes that overlay again while reusing
any successful generation projection.

## Submit and Resume Through a Run Handle

```python
from themis.orchestration.run_manifest import RunHandle
from themis.runtime import BenchmarkResult


handle = orchestrator.submit(benchmark, runtime=runtime)
print(handle.run_id, handle.status, handle.pending_work_items)

resumed = orchestrator.resume(handle.run_id, runtime=runtime)

if isinstance(resumed, RunHandle):
    print(resumed.run_id, resumed.status, resumed.pending_work_items)
elif isinstance(resumed, BenchmarkResult):
    print(resumed.trial_hashes[:3])
```

For the local backend, `submit()` executes immediately and returns a completed
handle once the missing work items finish. For batch or worker-pool backends,
the handle remains pending until external workers or imports complete the
manifested work items. `resume()` reuses the persisted canonical manifest and
returns either a new `RunHandle` or a final `BenchmarkResult` when nothing is
left to do.

## Inspect Runtime Progress

```python
from themis.progress import ProgressConfig
from themis.types.enums import RunStage

progress = orchestrator.get_run_progress(handle.run_id)

print(progress.active_stage)
print(progress.processed_items, progress.remaining_items)
print(progress.stage_counts[RunStage.TRANSFORM].failed_items)
```

`get_run_progress()` always reports the full run for that `run_id`, not just
the stage-specific entry point that happened to execute most recently. Failed
work items remain visible in the snapshot until they are retried and complete.

If you want live updates while a run is executing, pass a callback:

```python
snapshots = []
result = orchestrator.run(
    benchmark,
    runtime=runtime,
    progress=ProgressConfig(callback=snapshots.append),
)
```

When a callback is provided, Themis does not attach the Rich terminal renderer
unless you explicitly set `renderer=...`.

## Inspect the Planned Run

```python
manifest = orchestrator.plan(benchmark)

print(manifest.run_id)
print(manifest.backend_kind)
print(len(manifest.work_items))
print(sum(item.status == "pending" for item in manifest.work_items))
```

The manifest gives you the explicit list of deterministic work items that still
need to run. That same snapshot is what Themis uses for external generation or
evaluation handoff.

## Inspect a Trial

```python
trial = result.get_trial(result.trial_hashes[0])
print(trial.status)
print(trial.candidates[0].evaluation.aggregate_scores)
```

## Inspect a Timeline

```python
view = result.view_timeline(result.trial_hashes[0], record_type="trial")
for stage in view.timeline.stages:
    print(stage.stage, stage.status.value, stage.duration_ms)
```

Candidate views expose the richer read-side payloads:

```python
trial = result.get_trial(result.trial_hashes[0])
candidate_view = result.view_timeline(trial.candidates[0].candidate_id)

print(candidate_view.inference.raw_text)

if candidate_view.conversation is not None:
    print(len(candidate_view.conversation.events))

if candidate_view.judge_audit is not None:
    print(len(candidate_view.judge_audit.judge_calls))

if candidate_view.observability is not None:
    print(candidate_view.observability.langfuse_url)
```

Use the trial view when you care about run-level stage timing. Use the candidate
view when you need the concrete inference, extraction, evaluation, judge, or
observability payloads tied to one candidate.

## Inspect Specific Overlays

```python
transform_result = result.for_transform(result.transform_hashes[0])
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
```

Use transform views when you want normalized outputs before scoring. Use
evaluation views when you want metric-backed candidate records.

## Diff Two Benchmark Specs

```python
diff = orchestrator.diff_specs(baseline_benchmark, treatment_benchmark)

print(diff.changed_experiment_fields)
print(diff.added_trial_hashes[:3])
```

This is useful when you add a model, a benchmark slice, a prompt template, or a
metric and want to confirm that only the expected trial set changed.

## Estimate Work Before Running

```python
estimate = orchestrator.estimate(benchmark)

print(estimate.total_work_items)
print(estimate.work_items_by_stage)
print(estimate.estimated_total_tokens)
print(estimate.notes)
```

`estimate()` is intentionally approximate. It uses prompt templates, dataset
payload size, and the `max_tokens` budget to estimate work. It does not include
provider-side caching, unpublished pricing, or provider-specific routing logic,
so treat token and cost values as planning bounds rather than billable truth.

## Export Work for an External System

```python
generation_bundle = orchestrator.export_generation_bundle(benchmark)
evaluation_bundle = orchestrator.export_evaluation_bundle(benchmark)
```

Generation bundles carry the `TrialSpec`, dataset context, and deterministic
candidate IDs for each missing generation item. Evaluation bundles carry the
same trial context plus the candidate payload that should be scored externally.

## Suppress Item Payload Storage

If dataset payloads are sensitive, disable them at the project level:

```python
# `project.model_copy(...)` returns a new `ProjectSpec`, and
# `project.storage.model_copy(...)` replaces the nested storage config with a
# copy where `store_item_payloads` is disabled.
project = project.model_copy(
    update={"storage": project.storage.model_copy(update={"store_item_payloads": False})}
)
```

Timeline views remain available, but `item_payload` is omitted.

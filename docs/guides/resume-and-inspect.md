# Resume and Inspect Runs

## Resume a Run

Reuse the same:

- `ProjectSpec.storage.root_dir`
- `ExperimentSpec`
- `eval_revision`

When a completed projection already exists for that trial hash and evaluation
revision, `TrialExecutor` skips the work instead of recomputing it.

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

## Suppress Item Payload Storage

If dataset payloads are sensitive, disable them at the project level:

```python
project = project.model_copy(
    update={"storage": project.storage.model_copy(update={"store_item_payloads": False})}
)
```

Timeline views still work, but `item_payload` will be omitted.

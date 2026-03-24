# Analyze Results

Use the aggregate APIs for the macro view and timelines for the micro view.

## Candidate Score Aggregates

```python
for row in result.aggregate(group_by=["model_id", "slice_id", "metric_id"]):
    print(row)
```

Use this for the normal per-candidate score rows produced by `ScoreSpec.metrics`.

## Trace Score Aggregates

```python
for row in result.aggregate_trace(
    group_by=["model_id", "slice_id", "metric_id"],
    metric_id="tool_presence",
):
    print(row)
```

Use this for persisted trace metrics declared through `SliceSpec.trace_scores`.
If you need the raw persisted rows first, iterate them directly:

```python
for row in result.iter_trace_scores(metric_id="tool_presence"):
    print(row.trace_scope, row.record_id, row.score)
```

## Corpus Aggregates

```python
for row in result.aggregate_corpus(
    group_by=["model_id", "slice_id"],
    metric_id="f1_macro",
    candidate_selector="anchor_candidate",
):
    print(row)
```

Use this for post-hoc classification metrics computed from persisted benchmark
predictions instead of execution-time plugin scores.

## Trial and Timeline View

```python
trial = result.get_trial(result.trial_hashes[0])
candidate_id = trial.candidates[0].candidate_id
timeline = result.view_timeline(candidate_id)

print(timeline.inference.raw_text)
```

Worked example: `examples/06_hooks_and_timeline.py`

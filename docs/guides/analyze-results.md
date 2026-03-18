# Analyze Results

Use the aggregate APIs for the macro view and timelines for the micro view.

## Aggregate View

```python
for row in result.aggregate(group_by=["model_id", "slice_id", "metric_id"]):
    print(row)
```

## Trial and Timeline View

```python
trial = result.get_trial(result.trial_hashes[0])
candidate_id = trial.candidates[0].candidate_id
timeline = result.view_timeline(candidate_id)

print(timeline.inference.raw_text)
```

Worked example: `examples/06_hooks_and_timeline.py`

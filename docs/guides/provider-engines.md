# Build a Provider Engine

Provider engines still own final request construction. For benchmark-native
runs, Themis renders prompt templates before `infer(...)` is called.

## Use the Prepared Prompt

```python
class MyEngine:
    def infer(self, trial, context, runtime):
        messages = [message.model_dump(mode="json") for message in trial.prompt.messages]
        ...
```

## What Themis Preserves

- `trial.prompt.messages` contains the rendered messages sent to the model
- `trial.prompt.id`, `trial.prompt.family`, and `trial.prompt.variables` stay available for routing and logging
- `trial.task.dimensions`, `trial.task.slice_id`, and `trial.task.benchmark_id` stay available for request metadata and reporting

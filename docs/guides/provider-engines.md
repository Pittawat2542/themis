# Build a Provider Engine

Provider engines still own final request construction. For benchmark-native
runs, Themis renders prompt templates before `infer(...)` is called.

For the full benchmark authoring flow for agent evals, scripted turns, and tool
selection, see [Author Agent Evaluations and Tools](agent-evals-and-tools.md).

## Use the Prepared Prompt

```python
class MyEngine:
    def infer(self, trial, context, runtime):
        bootstrap = [message.model_dump(mode="json") for message in trial.prompt.messages]
        follow_up_turns = [
            [message.model_dump(mode="json") for message in turn.messages]
            for turn in trial.prompt.follow_up_turns
        ]
        tools = [tool.model_dump(mode="json") for tool in trial.tools]
        tool_handlers = runtime.tool_handlers
        seed = trial.params.seed
        ...
```

Use `trial.params.seed` as the authoritative seed to forward to provider
requests when the backend supports deterministic seeding. `runtime.candidate_seed`
mirrors the same effective value for backwards compatibility with older engines.
The persisted trial-level spec remains the planned `TrialSpec`; executed
generation params are exposed on candidate projections and candidate timeline
views instead.

## What Themis Preserves

- `trial.prompt.messages` contains the rendered bootstrap message sequence
- `trial.prompt.follow_up_turns` contains rendered scripted follow-up turns, when configured
- `trial.tools` contains the selected serializable tool specs for the trial
- `runtime.tool_handlers` contains the matching opaque runtime handlers registered for those tool IDs
- `trial.prompt.id`, `trial.prompt.family`, and `trial.prompt.variables` stay available for routing and logging
- `trial.task.dimensions`, `trial.task.slice_id`, and `trial.task.benchmark_id` stay available for request metadata and reporting
- `trial.params.seed` carries the effective deterministic seed for the current candidate when generation seeding is available
- projected candidates expose `effective_seed` and `effective_inference_params_hash` for the executed generation request
- `InferenceRecord.raw_text` should contain the terminal non-tool assistant answer when the engine runs an internal agent loop
- `InferenceResult.conversation` can carry the full tool and node trace back into Themis timelines

Themis validates selected tool coverage before `infer(...)` runs. If a trial
selects a tool and no matching handler is available from the registry or
`RuntimeContext.tool_handlers`, execution fails before the engine is called.

## Declaring Tools

- Put reusable tool specs on `ProjectSpec.tools`
- Override or add benchmark-local tools on `BenchmarkSpec.tools`
- Select the tools each slice should expose with `SliceSpec.tool_ids`
- Code-first experiments use `ExperimentSpec.tools` and `TaskSpec.tool_ids` with the same merge rules
- When compiling or planning a benchmark directly, pass `project_tools=` so those merge rules match `Orchestrator`

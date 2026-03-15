# Attach Telemetry & Observability

Use this guide when you want to observe runtime events in-process or forward
them to Langfuse.

## 1. Subscribe to runtime events

```python
from themis import Orchestrator
from themis.telemetry import TelemetryBus

telemetry_bus = TelemetryBus()
seen_events: list[tuple[str, dict[str, object]]] = []

telemetry_bus.subscribe(
    lambda event: seen_events.append((event.name, dict(event.payload)))
)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_loader=dataset_loader,
    telemetry_bus=telemetry_bus,
)
result = orchestrator.run(experiment)
```

The runtime emits event names such as:

- `trial_start`
- `trial_end`
- `error`
- `conversation_event`
- `tool_call`
- `tool_result`
- `extractor_attempt`
- `metric_start`
- `metric_end`

## 2. Forward the same bus to Langfuse

Install the optional extra first:

```bash
uv add "themis-eval[telemetry]"
```

Then subscribe a callback:

```python
from themis import Orchestrator
from themis.telemetry import LangfuseCallback, TelemetryBus

telemetry_bus = TelemetryBus()
LangfuseCallback(
    public_key="pk_live_...",
    secret_key="sk_live_...",
    base_url="https://cloud.langfuse.com",
).subscribe(telemetry_bus)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_loader=dataset_loader,
    telemetry_bus=telemetry_bus,
)
```

This forwards trial and candidate activity to Langfuse. If you only need
external traces, this high-level path is enough.

## 3. Persist external trace links into timeline views

`TelemetryBus` does not store external URLs on its own. To hydrate
`result.view_timeline(...).observability`, share the storage bundle's
observability store between the callback and the orchestrator.

```python
from themis.orchestration.orchestrator import Orchestrator
from themis.storage import build_storage_bundle
from themis.telemetry import LangfuseCallback, TelemetryBus

storage_bundle = build_storage_bundle(project.storage)

telemetry_bus = TelemetryBus()
LangfuseCallback(
    public_key="pk_live_...",
    secret_key="sk_live_...",
    base_url="https://cloud.langfuse.com",
    observability_store=storage_bundle.observability_store,
).subscribe(telemetry_bus)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    storage_bundle=storage_bundle,
    dataset_loader=dataset_loader,
    telemetry_bus=telemetry_bus,
)
```

After the run:

```python
trial = result.get_trial(result.trial_hashes[0])
candidate_view = result.view_timeline(trial.candidates[0].candidate_id)

if candidate_view.observability is not None:
    print(candidate_view.observability.url_for("langfuse"))
```

Use [Resume and Inspect Runs](resume-and-inspect.md) for the broader
timeline-inspection workflow.

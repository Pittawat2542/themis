# Themis v4

This repository contains the clean-slate Phase 3 execution engine for Themis v4.

Phase 3 keeps the immutable model layer from Phase 1 and extends the Phase 2 engine with workflow-backed evaluation: planning, generation fan-out, reduction, parsing, pure-metric scoring, LLM-backed metrics, judge fan-out, resume, rejudge, and bundle export/import for both generation and evaluation.

## Current scope

- immutable core domain models and typed execution contexts
- extension protocols for generators, reducers, parsers, metrics, and workflows
- `Experiment.compile()` to a reproducible `RunSnapshot`
- `Experiment.run()` / `Experiment.run_async()` for end-to-end execution
- typed `RuntimeConfig` for concurrency, rate limiting, and store retry control
- lazy planning plus event-backed resume state
- workflow-backed evaluation with persisted judge artifacts
- evaluation bundle export/import and `Experiment.rejudge()` / `rejudge_async()`
- in-memory and SQLite run stores
- OpenAI, vLLM, and LangGraph generator adapters
- generation bundle export/import helpers
- typed package distribution via `py.typed`

## What affects `run_id`

`run_id` is derived from `RunSnapshot.identity` only. The following inputs are identity-bearing and change the compiled `run_id`:

- dataset refs and dataset fingerprints
- generator, reducer, parser, and metric component refs
- generation `candidate_policy`
- evaluation `judge_config`
- evaluation `workflow_overrides`
- experiment `seeds`

These provenance fields do not affect `run_id`:

- Themis version
- Python version
- platform
- storage backend configuration
- runtime execution settings
- environment metadata

## Component inputs

Phase 2 accepts two component styles:

1. Builtin string components, resolved through a canonical registry.
   Current demo entries used by tests and examples:
   `generator/demo`, `reducer/demo`, `parser/demo`, `metric/demo`
2. Custom component objects that expose `component_id`, `version`, and `fingerprint()`.

Builtin component metadata is intentional identity. If a builtin component's version or fingerprint changes, compiled snapshots and `run_id` values change too.

## Event schema behavior

Run events use an additive, forward-compatible read model:

- unknown event types are skipped by the SQLite reader
- known event types accept newer `schema_version` values
- additive future fields on known event types are preserved on deserialization
- malformed known event payloads still fail validation

## Examples

Builtin component example:

```python
from themis import Experiment, RunStatus, RuntimeConfig
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset

experiment = Experiment(
    generation=GenerationConfig(generator="generator/demo", reducer="reducer/demo"),
    evaluation=EvaluationConfig(metrics=["metric/demo"], parsers=["parser/demo"]),
    storage=StorageConfig(store="memory"),
    datasets=[
        Dataset(
            dataset_id="dataset-1",
            cases=[Case(case_id="case-1", input={"question": "2+2"})],
        )
    ],
)

result = experiment.run(
    runtime=RuntimeConfig(
        max_concurrent_tasks=8,
        stage_concurrency={"generation": 4},
    )
)
assert result.status is RunStatus.COMPLETED
```

Custom component example:

```python
from themis import Experiment, RunStatus
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset, GenerationResult


class CustomGenerator:
    component_id = "generator/custom"
    version = "1.0"

    def fingerprint(self) -> str:
        return "custom-generator-fingerprint"

    async def generate(self, case: Case, ctx) -> GenerationResult:
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate",
            final_output={"answer": "4"},
        )


experiment = Experiment(
    generation=GenerationConfig(generator=CustomGenerator()),
    evaluation=EvaluationConfig(),
    storage=StorageConfig(store="memory"),
    datasets=[
        Dataset(
            dataset_id="dataset-1",
            cases=[Case(case_id="case-1", input={"question": "2+2"})],
        )
    ],
)

result = experiment.run()
assert result.status is RunStatus.COMPLETED
```

Adapter example:

```python
from openai import AsyncOpenAI

from themis import Experiment, RunStatus
from themis.adapters import openai
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset

experiment = Experiment(
    generation=GenerationConfig(
        generator=openai(
            "gpt-5.4-mini",
            client=AsyncOpenAI(),
            instructions="Answer directly.",
        )
    ),
    evaluation=EvaluationConfig(),
    storage=StorageConfig(store="memory"),
    datasets=[Dataset(dataset_id="dataset-1", cases=[Case(case_id="case-1", input="2+2?")])],
)

result = experiment.run()
assert result.status is RunStatus.COMPLETED
```

vLLM extra note:

- `themis-eval[vllm]` targets Linux installs, where the `vllm` package is available and pulls in the OpenAI-compatible client path used by the adapter.
- On macOS, use an injected client or a separate vLLM environment if you need to exercise the adapter locally.

Generation bundle example:

```python
from themis import InMemoryRunStore
from themis.core import export_generation_bundle, import_generation_bundle

source_store = InMemoryRunStore()
source_store.initialize()
snapshot = experiment.compile()
source_store.persist_snapshot(snapshot)
bundle = export_generation_bundle(source_store, snapshot.run_id)

target_store = InMemoryRunStore()
target_store.initialize()
import_generation_bundle(target_store, bundle)

assert target_store.resume(snapshot.run_id) is not None
```

## Runtime controls

- `RuntimeConfig.max_concurrent_tasks` bounds all in-flight work. Default: `32`.
- `RuntimeConfig.stage_concurrency["generation"]` limits generation fan-out separately from the global cap.
- `RuntimeConfig.provider_concurrency` limits concurrent requests per provider key.
- `RuntimeConfig.provider_rate_limits` sets requests-per-minute token buckets. Default per provider: `60`.
- `RuntimeConfig.store_retry_attempts` and `store_retry_delay` control event/blob persistence retries.

## Resume and bundle behavior

- Resume skips completed generation, reduction, parsing, and successful scoring work.
- Failed scoring is retried on resume.
- Generation bundle export/import round-trips `GenerationCompletedEvent` payloads without changing `run_id`.
- Evaluation bundle export/import round-trips `EvaluationCompletedEvent` payloads and restores inspectable scores when a candidate id is available.
- `Experiment.rejudge()` and `rejudge_async()` re-run workflow-backed metrics from stored upstream artifacts without regenerating candidates.

## Not included yet

- CLI
- reporting and read models beyond the execution snapshot and bundle helpers
- rebuilt end-user documentation

## Optional extras

- `pip install themis-eval[openai]`
- `pip install themis-eval[vllm]` on Linux
- `pip install themis-eval[langgraph]`

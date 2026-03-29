# Themis v4

This repository contains the clean-slate Phase 3 execution engine for Themis v4.

Phase 3 keeps the immutable model layer from Phase 1 and extends the core engine with workflow-backed evaluation: generation fan-out, reduction, parsing, pure-metric scoring, LLM-backed metrics, judge-call planning, resume, rejudge, and bundle export/import for both generation and evaluation.

## Current scope

- immutable core domain models and typed execution contexts
- extension protocols for generators, reducers, parsers, metrics, and workflows
- `Experiment.compile()` to a reproducible `RunSnapshot`
- `Experiment.run()` / `Experiment.run_async()` for end-to-end execution, with optional explicit `store=...`
- typed `RuntimeConfig` for concurrency, rate limiting, and store retry control
- lazy planning plus event-backed resume state
- workflow-backed evaluation with persisted judge artifacts
- evaluation bundle export/import and `Experiment.rejudge()` / `rejudge_async()`
- public inspection helpers: `get_execution_state()` and `get_evaluation_execution()`
- in-memory and SQLite run stores
- OpenAI, vLLM, and LangGraph generator adapters
- generation bundle export/import helpers from the root package
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

The current runtime accepts two component styles:

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

Judge workflow example:

```python
from themis.core import build_prompt_template_context
from themis.core.models import Score
from themis.core.workflows import AggregationResult, JudgeCall, ParsedJudgment, RenderedJudgePrompt


class PassFailWorkflow:
    component_id = "workflow/pass-fail"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-pass-fail"

    def judge_calls(self) -> list[JudgeCall]:
        return [JudgeCall(call_id="call-0", judge_model_id="judge/demo")]

    def render_prompt(self, call, subject, ctx) -> RenderedJudgePrompt:
        values = build_prompt_template_context(subject, ctx, call)
        return RenderedJudgePrompt(
            prompt_id="prompt-0",
            content="Grade this answer: {candidate_output}".format(**values),
        )

    def parse_judgment(self, call, response, ctx) -> ParsedJudgment:
        del call, ctx
        label = response.raw_response.strip().lower()
        return ParsedJudgment(label=label, score=1.0 if label == "pass" else 0.0)

    def score_judgment(self, call, judgment, ctx):
        del call, ctx
        return Score(metric_id="metric/pass-fail", value=float(judgment.score or 0.0))

    def aggregate(self, judgments, scores, ctx) -> AggregationResult | None:
        del judgments, ctx
        if not scores:
            return None
        return AggregationResult(method="mean", value=sum(score.value for score in scores) / len(scores))
```

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
from themis import InMemoryRunStore, export_generation_bundle, import_generation_bundle

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

Inspection and rejudge example:

```python
from themis import (
    InMemoryRunStore,
    get_evaluation_execution,
    get_execution_state,
)

store = InMemoryRunStore()
result = experiment.run(store=store)

state = get_execution_state(store, experiment.compile().run_id)
execution = get_evaluation_execution(store, experiment.compile().run_id, "case-1", "metric/judge")

assert result.status is RunStatus.COMPLETED
assert state.run_id == experiment.compile().run_id
assert execution is None or execution.execution_id
```

## Runtime controls

- `RuntimeConfig.max_concurrent_tasks` bounds all in-flight work. Default: `32`.
- `RuntimeConfig.stage_concurrency["generation"]` limits generation fan-out separately from the global cap.
- `RuntimeConfig.stage_concurrency["evaluation"]` limits in-flight judge calls separately from the global cap.
- `RuntimeConfig.provider_concurrency` limits concurrent requests per provider key.
- `RuntimeConfig.provider_rate_limits` sets requests-per-minute token buckets. Default per provider: `60`.
- `RuntimeConfig.store_retry_attempts` and `store_retry_delay` control event/blob persistence retries.

## Resume and bundle behavior

- Resume skips completed generation, reduction, parsing, and successful scoring work.
- Failed scoring is retried on resume.
- Generation bundle export/import round-trips `GenerationCompletedEvent` payloads, including deterministic blob refs, without changing `run_id`.
- Evaluation bundle export/import round-trips `EvaluationCompletedEvent` payloads, including deterministic blob refs, and restores inspectable scores when a candidate id is available.
- `Experiment.rejudge()` and `rejudge_async()` re-run workflow-backed metrics from stored upstream artifacts without regenerating candidates.
- Memory-backed rejudge requires the original store instance via `store=...`; SQLite-backed runs can be reopened by path.

## Not included yet

- CLI
- reporting and read models beyond the execution snapshot and bundle helpers
- rebuilt end-user documentation

## Optional extras

- `pip install themis-eval[openai]`
- `pip install themis-eval[vllm]` on Linux
- `pip install themis-eval[langgraph]`

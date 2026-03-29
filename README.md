# Themis v4

This repository contains the clean-slate Phase 2 execution engine for Themis v4.

Phase 2 keeps the immutable model layer from Phase 1 and adds a runnable execution engine: planning, generation fan-out, reduction, parsing, pure-metric scoring, tracing hooks, resume, and generation bundle export/import.

## Current scope

- immutable core domain models and typed execution contexts
- extension protocols for generators, reducers, parsers, metrics, and workflows
- `Experiment.compile()` to a reproducible `RunSnapshot`
- `Experiment.run()` / `Experiment.run_async()` for end-to-end execution
- lazy planning plus event-backed resume state
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
- environment metadata

## Component inputs

Phase 1 accepts two component styles:

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
from themis import Experiment, RunStatus
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

result = experiment.run()
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

## Not included yet

- CLI
- LLM-backed evaluation workflows
- judge orchestration and workflow-runner execution
- reporting and read models beyond the execution snapshot and generation bundles
- rebuilt end-user documentation

## Optional extras

- `pip install themis-eval[openai]`
- `pip install themis-eval[vllm]`
- `pip install themis-eval[langgraph]`

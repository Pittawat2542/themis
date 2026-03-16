# Provider-backed Run

This tutorial shows the smallest provider-backed path: install the provider
extra, authenticate with environment variables, register one engine, and run one
tiny task. It is intentionally minimal so you can verify the end-to-end wiring
before scaling up.

## Before You Start

`uv` is required for the documented workflow. Install the provider extra and set
credentials in your shell:

```bash
uv add "themis-eval[providers-openai]"
export OPENAI_API_KEY=...
```

Create a new file named `provider_run.py`. You will run it with:

```bash
uv run python provider_run.py
```

Provider output is not deterministic documentation data. The exact response text,
latency, token counts, and IDs vary by model version, account policy, and the
provider service state at runtime.

## Step 1: Declare one tiny task

```python
from pathlib import Path

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    GenerationSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptTemplateSpec,
    SqliteBlobStorageSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.errors import InferenceError, RetryableProviderError
from themis.records import InferenceRecord, MetricScore
from themis.types.enums import ErrorCode
```

Keep the first run to one item and one metric so a provider or auth problem is
obvious immediately.

## Step 2: Add the engine with explicit failure mapping

```python
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError


class OpenAIEngine:
    def __init__(self) -> None:
        self._client = OpenAI()

    def infer(self, trial, context, runtime):
        del runtime
        try:
            response = self._client.responses.create(
                model=trial.model.model_id,
                input=[{"role": "user", "content": context["question"]}],
                max_output_tokens=trial.params.max_tokens,
            )
        except RateLimitError as exc:
            raise RetryableProviderError(
                code=ErrorCode.PROVIDER_RATE_LIMIT,
                message="Provider rate limited the request.",
                details={"provider": "openai", "model_id": trial.model.model_id},
            ) from exc
        except APIConnectionError as exc:
            raise RetryableProviderError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Provider connection failed.",
                details={"provider": "openai", "model_id": trial.model.model_id},
            ) from exc
        except AuthenticationError as exc:
            raise InferenceError(
                code=ErrorCode.PROVIDER_AUTH,
                message="Provider authentication failed.",
                details={"provider": "openai", "model_id": trial.model.model_id},
            ) from exc

        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.spec_hash}",
                raw_text=response.output_text,
                provider="openai",
                model_id=trial.model.model_id,
            )
        )
```

Known failure pattern:

```text
InferenceError: Provider authentication failed.
```

## Step 3: Add a tiny dataset and metric

```python
class OneItemDatasetLoader:
    def load_task_items(self, task):
        del task
        return [{"item_id": "item-1", "question": "Reply with the word four.", "answer": "four"}]


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text.strip().lower() if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
            details={"actual": actual, "expected": context["answer"]},
        )
```

## Step 4: Register and run

```python
registry = PluginRegistry()
registry.register_inference_engine("openai", OpenAIEngine())
registry.register_metric("exact_match", ExactMatchMetric())

project = ProjectSpec(
    project_name="provider-tutorial",
    researcher_id="docs",
    global_seed=7,
    storage=SqliteBlobStorageSpec(root_dir=str(Path(".cache/themis-examples/provider-tutorial")), compression="none"),
    execution_policy=ExecutionPolicySpec(max_retries=2),
)
experiment = ExperimentSpec(
    models=[ModelSpec(model_id="gpt-4.1-mini", provider="openai")],
    tasks=[
        TaskSpec(
            task_id="one-item-check",
            dataset=DatasetSpec(source="memory"),
            generation=GenerationSpec(),
            evaluations=[EvaluationSpec(name="default", metrics=["exact_match"])],
        )
    ],
    prompt_templates=[
        PromptTemplateSpec(
            id="baseline",
            messages=[PromptMessage(role="user", content="Answer the task directly.")],
        )
    ],
    inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
)

orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_loader=OneItemDatasetLoader(),
)
result = orchestrator.run(experiment)
trial = result.get_trial(result.trial_hashes[0])
print(trial.status)
print(trial.candidates[0].evaluation.aggregate_scores["exact_match"])
```

Expected success output pattern:

```text
RecordStatus.OK
0.0 or 1.0
```

The score is intentionally documented as a pattern rather than a fixed value,
because provider responses can change even when the wiring is correct.

## Next Steps

- Use [Build a Provider Engine](../guides/provider-engines.md) for the
  task-oriented guide.
- Use [Scale Execution](../guides/scaling-execution.md) after the one-item run
  succeeds consistently.
- Use [Resume and Inspect Runs](../guides/resume-and-inspect.md) when you want
  run handles, persisted progress, and timeline inspection.

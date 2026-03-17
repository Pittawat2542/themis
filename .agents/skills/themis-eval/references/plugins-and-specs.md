# Plugins And Specs

## Keep The Boundary Clear

- `ProjectSpec` holds shared runtime policy.
- `BenchmarkSpec` holds benchmark semantics.
- `SliceSpec` holds dataset, dimensions, parse pipelines, and score overlays.
- `PluginRegistry` is the runtime lookup table.

## Build A Dataset Provider

```python
class MyProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {
                "item_id": "item-1",
                "question": "2 + 2",
                "answer": "4",
                "metadata": {"difficulty": "easy"},
            }
        ]
```

Use `DatasetQuerySpec` for selection logic instead of payload conventions.

## Implement The Minimum Plugin Set

Themis renders benchmark prompt templates before your engine runs. In normal
benchmark-native flows, consume `trial.prompt.messages` directly instead of
calling a prompt-render helper inside the engine.

Inference engine:

```python
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord


class MyEngine:
    def infer(self, trial, context, runtime):
        messages = [message.model_dump(mode="json") for message in trial.prompt.messages]
        prompt_family = trial.prompt.family
        del messages, prompt_family, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=str(context["answer"]),
            )
        )
```

Metric:

```python
from themis.records import MetricScore


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
        )
```

## Use Parse Pipelines

```python
SliceSpec(
    ...,
    parses=[ParseSpec(name="parsed", extractors=["boxed_text", "normalized_text"])],
    scores=[ScoreSpec(name="default", parse="parsed", metrics=["exact_match"])],
)
```

## Use Hooks For Small Pipeline Edits

```python
class MyHook:
    def pre_inference(self, trial, prompt):
        return prompt

    def post_inference(self, trial, result):
        return result

    def pre_extraction(self, trial, candidate):
        return candidate

    def post_extraction(self, trial, candidate):
        return candidate

    def pre_eval(self, trial, candidate):
        return candidate

    def post_eval(self, trial, candidate):
        return candidate
```

## Use Judge-Backed Metrics

```python
from themis import ModelSpec, PromptMessage
from themis.records import MetricScore
from themis.specs import JudgeInferenceSpec


class JudgeMetric:
    def score(self, trial, candidate, context):
        judge = context["judge_service"]
        judge_record = judge.judge(
            metric_id="judge_pass",
            parent_candidate=candidate,
            judge_spec=JudgeInferenceSpec(
                model=ModelSpec(model_id="judge-model", provider="demo"),
            ),
            prompt=trial.prompt.model_copy(
                update={
                    "messages": [
                        PromptMessage(
                            role="user",
                            content="Judge whether the answer is correct.",
                        )
                    ]
                }
            ),
            runtime=context,
        )
        return MetricScore(
            metric_id="judge_pass",
            value=float(judge_record.raw_text == "PASS"),
        )
```

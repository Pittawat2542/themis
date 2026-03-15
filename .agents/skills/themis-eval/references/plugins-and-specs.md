# Plugins And Specs

## Keep The Boundary Clear

- `ProjectSpec` holds stable storage, retry, and backend policy.
- `ExperimentSpec` holds models, tasks, prompt templates, sampling, transforms,
  evaluations, and inference params.
- `PluginRegistry` is the runtime lookup table and should be built explicitly.

## Build A Dataset Loader

Implement `load_task_items(task)` and return item payloads with stable IDs when
possible:

```python
class MyLoader:
    def load_task_items(self, task):
        del task
        return [
            {
                "item_id": "item-1",
                "question": "2 + 2",
                "answer": "4",
                "metadata": {"difficulty": "easy"},
            }
        ]
```

Use `metadata` when the user needs deterministic filtering or sampling through
`ItemSamplingSpec`.

## Implement The Minimum Plugin Set

Inference engine:

```python
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord


class MyEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
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
        expected = str(context["answer"])
        return MetricScore(
            metric_id="exact_match",
            value=float(actual.strip() == expected),
        )
```

Registry:

```python
registry = PluginRegistry()
registry.register_inference_engine("demo", MyEngine())
registry.register_metric("exact_match", ExactMatchMetric())
```

The model `provider` in `ModelSpec` must match the inference engine name you
register.

## Use Built-In Extractors Before Writing One

Built-ins available in the documented workflow:

- `regex`
- `json_schema` with the `extractors` extra
- `first_number`
- `choice_letter`

Example:

```python
from themis import EvaluationSpec, ExtractorChainSpec, OutputTransformSpec

task = TaskSpec(
    task_id="qa",
    dataset=DatasetSpec(source="memory"),
    generation=GenerationSpec(),
    output_transforms=[
        OutputTransformSpec(
            name="parsed",
            extractor_chain=ExtractorChainSpec(
                extractors=[
                    "first_number",
                    {"id": "regex", "config": {"pattern": r"score = (\\d+)", "group": 1}},
                ]
            ),
        )
    ],
    evaluations=[EvaluationSpec(name="default", transform="parsed", metrics=["exact_match"])],
)
```

Write a custom extractor only when the built-ins cannot express the parsing
logic.

## Use Hooks For Small Pipeline Edits

Use hooks when the user wants to mutate prompts or records without replacing a
whole stage:

```python
class MyHook:
    def pre_inference(self, trial, prompt):
        return prompt

    def post_inference(self, trial, result):
        return result


registry.register_hook("my_hook", MyHook(), priority=10)
```

## Use Judge-Backed Metrics For Model-Graded Scoring

Judge-backed metrics call `context["judge_service"]` and usually need a judge
engine plus a `JudgeInferenceSpec`:

```python
from themis import ModelSpec, PromptMessage, PromptTemplateSpec
from themis.records import MetricScore
from themis.specs.foundational import JudgeInferenceSpec


class JudgeMetric:
    def score(self, trial, candidate, context):
        judge = context["judge_service"]
        judge_inference = judge.judge(
            metric_id="judge_pass",
            parent_candidate=candidate,
            judge_spec=JudgeInferenceSpec(
                model=ModelSpec(model_id="judge-model", provider="judge"),
            ),
            prompt=PromptTemplateSpec(
                id="judge-prompt",
                messages=[
                    PromptMessage(
                        role="user",
                        content="Does the answer match the reference?",
                    )
                ],
            ),
            runtime={"task_spec": trial.task, "dataset_context": context},
        )
        return MetricScore(
            metric_id="judge_pass",
            value=float(judge_inference.raw_text == "PASS"),
            details={"judge_raw_text": judge_inference.raw_text},
        )
```

If the user needs the full working pattern, combine this reference with
`references/results-and-ops.md` for stored audit inspection. Do not assume a
local examples directory exists.

## Author Specs In The Documented Shape

Use:

- `ProjectSpec` for storage root, researcher ID, seed, execution policy, and
  optional backend
- `ExperimentSpec` for models, tasks, prompt templates, inference grid, and
  item sampling
- `TaskSpec` for dataset, generation, transforms, and evaluations

Prefer `ExperimentSpec.model_copy(update=...)` when evolving a run instead of
mutating pieces in place.

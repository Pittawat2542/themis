# Add a Minimal Plugin Set

Use this guide when you already have a task and dataset loader and need the
smallest set of custom runtime pieces to make it executable.

## 1. Add an inference engine

An inference engine must implement `infer(trial, context, runtime)`.

```python
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord


class DemoEngine:
    def infer(self, trial, context, runtime):
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.item_id}",
                raw_text=str(context["answer"]),
            )
        )
```

## 2. Add an extractor when raw text is not enough

Extractors are optional. Use them when your metrics should score structured
output instead of raw model text.

Before writing a custom extractor, check the built-ins that ship with every
`PluginRegistry()`:

| Extractor ID | Use it for | Config |
| --- | --- | --- |
| `regex` | capture one regex match from raw text | `pattern`, optional `group` |
| `json_schema` | parse JSON and validate its shape | `schema` |
| `first_number` | pull the first integer or float token | none |
| `choice_letter` | parse `A/B/C/...` multiple-choice answers | optional `choices` |

Example:

```python
task = TaskSpec(
    task_id="example-task",
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

Install `themis-eval[extractors]` when you want to use the built-in
`json_schema` extractor.

## Provider SDK extras

Themis does not ship built-in OpenAI, LiteLLM, or vLLM engines in this package.
These extras install the SDKs used by your own `InferenceEngine`
implementations without forcing every user to install them.

| Extra | Install it when your engine imports | Typical provider name |
| --- | --- | --- |
| `providers-openai` | `openai` | `openai` |
| `providers-litellm` | `litellm` | `litellm` |
| `providers-vllm` | `vllm` | `vllm` |

Install one with:

```bash
uv add "themis-eval[providers-openai]"
```

Then register your engine as usual:

```python
registry.register_inference_engine("openai", MyOpenAIEngine())
```

Use [Build a Provider Engine](provider-engines.md) when you want the full
provider-backed skeleton with auth and retry guidance.

## 3. Write a custom extractor only when the built-ins do not fit

Custom extractors use the signature `(trial, candidate, config)`.

```python
from themis.records import ExtractionRecord


class FirstWordExtractor:
    def extract(self, trial, candidate, config):
        del trial, config
        text = candidate.inference.raw_text if candidate.inference else ""
        first_word = text.split()[0] if text else None
        return ExtractionRecord(
            spec_hash=f"ext_{candidate.spec_hash}",
            extractor_id="first_word",
            success=first_word is not None,
            parsed_answer=first_word,
        )
```

## 4. Add a metric

Metrics receive the `TrialSpec`, the current `CandidateRecord`, and a context
mapping derived from the dataset item.

```python
from themis.records import MetricScore


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        if candidate.extractions:
            actual = candidate.extractions[0].parsed_answer
        else:
            actual = candidate.inference.raw_text if candidate.inference else ""
        expected = str(context["answer"])
        return MetricScore(metric_id="exact_match", value=float(actual == expected))
```

## 5. Emit qualitative tags from metrics

Metrics can return structured details alongside the scalar score. Use that for
qualitative labeling you want to inspect later:

```python
class SafetyMetric:
    def score(self, trial, candidate, context):
        del trial, context
        text = candidate.inference.raw_text if candidate.inference else ""
        tags = ["refusal"] if "cannot help" in text.lower() else []
        return MetricScore(
            metric_id="safety",
            value=float(not tags),
            details={"tags": tags, "raw_text": text},
        )
```

Those tags become queryable through `ExperimentResult.iter_tagged_examples()`.

## 6. Run multiple judge-backed metrics on the same candidates

The same evaluation can run more than one metric:

```python
task = TaskSpec(
    task_id="example-task",
    dataset=DatasetSpec(source="memory"),
    generation=GenerationSpec(),
    evaluations=[
        EvaluationSpec(
            name="judge-suite",
            metrics=["helpfulness_judge", "safety_judge"],
        )
    ],
)
```

Each metric can call `context["judge_service"]` with its own prompt, model, and
parsing logic. That is how you apply multiple judges or multiple judge prompts
to the same candidate set without duplicating generation.

## 7. Register the pieces

```python
registry.register_inference_engine("demo", DemoEngine())
registry.register_extractor("first_word", FirstWordExtractor())
registry.register_metric("exact_match", ExactMatchMetric())
```

## 8. Reference them from `TaskSpec`

```python
task = TaskSpec(
    task_id="example-task",
    dataset=DatasetSpec(source="memory"),
    generation=GenerationSpec(),
    output_transforms=[
        OutputTransformSpec(
            name="parsed",
            extractor_chain=ExtractorChainSpec(extractors=["first_word"]),
        )
    ],
    evaluations=[EvaluationSpec(name="default", transform="parsed", metrics=["exact_match"])],
)
```

If you need to change prompts or candidate objects around the pipeline rather
than replacing a whole stage, see [Plugins and Hooks](../concepts/plugins-and-hooks.md).

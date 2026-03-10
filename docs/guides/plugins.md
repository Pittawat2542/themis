# Add a Minimal Plugin Set

Use this guide when you already have a task and dataset loader and need the
smallest set of custom runtime pieces to make it executable.

## 1. Add an inference engine

An inference engine must implement `infer(trial, context, runtime)`.

```python
from themis.contracts.protocols import InferenceResult
from themis.records.inference import InferenceRecord


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
    default_extractor_chain=ExtractorChainSpec(
        extractors=[
            "first_number",
            {"id": "regex", "config": {"pattern": r"score = (\\d+)", "group": 1}},
        ]
    ),
    default_metrics=["exact_match"],
)
```

Install `themis-eval[extractors]` when you want to use the built-in
`json_schema` extractor.

## Provider SDK extras

Themis does not ship built-in OpenAI, LiteLLM, or vLLM engines in this package.
Those extras exist so your own `InferenceEngine` implementations can import the
matching SDKs without forcing every user to install them.

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

## 3. Write a custom extractor only when the built-ins do not fit

```python
from themis.records.extraction import ExtractionRecord


class FirstWordExtractor:
    def extract(self, trial, candidate, config=None):
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
from themis.records.evaluation import MetricScore


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        if candidate.extractions:
            actual = candidate.extractions[0].parsed_answer
        else:
            actual = candidate.inference.raw_text if candidate.inference else ""
        expected = str(context["answer"])
        return MetricScore(metric_id="exact_match", value=float(actual == expected))
```

## 5. Register the pieces

```python
registry.register_inference_engine("demo", DemoEngine())
registry.register_extractor("first_word", FirstWordExtractor())
registry.register_metric("exact_match", ExactMatchMetric())
```

## 6. Reference them from `TaskSpec`

```python
task = TaskSpec(
    task_id="example-task",
    dataset=DatasetSpec(source="memory"),
    default_extractor_chain=ExtractorChainSpec(extractors=["first_word"]),
    default_metrics=["exact_match"],
)
```

If you need to change prompts or candidate objects around the pipeline rather
than replacing a whole stage, see [Plugins and Hooks](../concepts/plugins-and-hooks.md).

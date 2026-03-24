# Author Parse Pipelines and Scores

Keep parsing and scoring separate.

## Parse First

Use a `ParseSpec` when raw model text is not the final scoring surface:

```python
from themis import MetricRefSpec

SliceSpec(
    ...,
    parses=[ParseSpec(name="parsed", extractors=["boxed_text", "normalized_text"])],
    scores=[
        ScoreSpec(
            name="exact",
            parse="parsed",
            metrics=[MetricRefSpec(id="exact_match")],
        )
    ],
)
```

## Metric Rule

Metrics should consume the parsed candidate state, not reparse raw text:

```python
class ParsedExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial, context
        extraction = candidate.best_extraction()
        parsed = extraction.parsed_answer if extraction is not None else None
        return MetricScore(metric_id="exact_match", value=float(parsed == "42"))
```

## Built-Ins

`PluginRegistry` auto-registers built-in extractors including:

- `choice_letter`
- `first_number`
- `boxed_text`
- `normalized_text`
- `regex`

Custom extractor example: `examples/03_custom_extractor_metric.py`

## Metric Surfaces

Themis now has four distinct metric surfaces:

- Candidate metrics: run per candidate via `ScoreSpec.metrics`
- Trial metrics: aggregate one candidate set during evaluation via `ScoreSpec.metrics`
- Trace metrics: score persisted candidate or trial traces via `SliceSpec.trace_scores`
- Corpus metrics: post-hoc analysis helpers on `BenchmarkResult`, not execution-time metrics

Corpus metrics are not valid inside `ScoreSpec.metrics`.

## Trace Scores

Use `trace_scores` for agent or workflow checks that inspect persisted traces:

```python
SliceSpec(
    ...,
    trace_scores=[
        TraceScoreSpec(
            name="workflow",
            scope="candidate_trace",
            metrics=[MetricRefSpec(id="tool_presence", config={"tool_name": "search"})],
        )
    ],
)
```

`tool_stage` depends on conversation events written after this metrics update. Older
stored runs may need to be re-run before `tool_stage` can match correctly.

## Corpus Metrics

Use `BenchmarkResult.aggregate_corpus(...)` for classification-style corpus metrics:

```python
rows = result.aggregate_corpus(
    group_by=["model_id", "slice_id"],
    metric_id="f1_macro",
    candidate_selector="anchor_candidate",
)
```

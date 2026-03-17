# Author Parse Pipelines and Scores

Keep parsing and scoring separate.

## Parse First

Use a `ParseSpec` when raw model text is not the final scoring surface:

```python
SliceSpec(
    ...,
    parses=[ParseSpec(name="parsed", extractors=["boxed_text", "normalized_text"])],
    scores=[ScoreSpec(name="exact", parse="parsed", metrics=["exact_match"])],
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

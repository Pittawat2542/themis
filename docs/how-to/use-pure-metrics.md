---
title: Use pure metrics
diataxis: how-to
audience: users configuring deterministic scoring
goal: Show how to apply builtin and custom pure metrics.
---

# Use pure metrics

Goal: configure deterministic scoring based only on parsed output and case data.

When to use this:

Use this guide when you do not need judge models or workflow execution.

## Procedure

Choose one or more builtin pure metrics such as `builtin/exact_match`, `builtin/f1`, or `builtin/bleu`, and pair them with a parser that normalizes the reduced candidate into the shape the metric expects.

```python
--8<-- "examples/docs/pure_metrics.py"
```

--8<-- "docs/_snippets/how-to/pure-metrics-note.md"

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Exact structured comparison | Parsed output should match the expected value exactly | Too strict for fuzzy or stylistic outputs | `builtin/exact_match` |
| Token-overlap style scoring | Partial lexical overlap is more meaningful than exact equality | Still surface-form based, not semantic judging | `builtin/f1`, `builtin/bleu` |
| Task-specific deterministic logic | The scoring rule is deterministic but domain-specific | Requires custom metric implementation | `PureMetric` |

## Expected result

The run should complete without judge models and produce direct final scores.

## Troubleshooting

- [Metric families and subjects](../explanation/metric-families-and-subjects.md)
- [Data models reference](../reference/data-models.md)

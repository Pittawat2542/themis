---
title: Use reduction strategies
diataxis: how-to
audience: users working with multi-candidate generation
goal: Show when and how to use builtin selectors, reducers, or custom multi-candidate logic.
---

# Use reduction strategies

Goal: choose a selector or reducer for multi-candidate generation.

When to use this:

Use this guide when `num_samples` is greater than one and you need candidate selection or reduction before parsing or scoring.

## Procedure

Use `builtin/majority_vote` when multiple candidates can converge on the same output and a simple majority is sufficient.

Use `builtin/best_of_n` when judge-backed comparison should select the best candidate before any optional reduction step.

Use a custom reducer when the selection rule is domain-specific.

```python
--8<-- "examples/docs/custom_reducer.py"
```

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Deterministic output voting | Multiple candidates often converge on the same normalized answer | Less useful when outputs vary semantically but not textually | `builtin/majority_vote` |
| Judged selection | A judge should pick the strongest candidate before reduction | Requires judge-backed selection logic | `builtin/best_of_n` |
| Domain-specific selection | Selection or synthesis logic belongs entirely to your own task | Requires custom reduction code | Custom reducer, `CandidateReducer` |

--8<-- "docs/_snippets/how-to/reduction-strategies-note.md"

## Expected result

The run should produce either a selected candidate set or a reduced candidate that downstream parsing and scoring can consume.

## Troubleshooting

- [Fan-out, reduction, parsing, and scoring](../explanation/fanout-reduction-parsing-scoring.md)
- [Reducer vs parser vs metric](../explanation/reducer-parser-metric-boundaries.md)

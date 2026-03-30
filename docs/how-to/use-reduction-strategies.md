---
title: Use reduction strategies
diataxis: how-to
audience: users working with multi-candidate generation
goal: Show when and how to use builtin or custom reduction strategies.
---

# Use reduction strategies

Goal: choose a reducer for multi-candidate generation.

When to use this:

Use this guide when `num_samples` is greater than one and you need a reduced candidate before parsing or scoring.

## Procedure

Use `builtin/majority_vote` when multiple candidates can converge on the same output and a simple majority is sufficient.

Use `builtin/best_of_n` when judge-backed comparison should choose the best candidate among alternatives.

Use a custom reducer when the selection rule is domain-specific.

```python
--8<-- "examples/docs/custom_reducer.py"
```

## Variants

- deterministic output voting: `builtin/majority_vote`
- judged selection: `builtin/best_of_n`
- domain-specific selection: custom reducer

--8<-- "docs/_snippets/how-to/reduction-strategies-note.md"

## Expected result

The run should produce a reduced candidate that downstream parsing and scoring can consume.

## Troubleshooting

- [Fan-out, reduction, parsing, and scoring](../explanation/fanout-reduction-parsing-scoring.md)
- [Reducer vs parser vs metric](../explanation/reducer-parser-metric-boundaries.md)

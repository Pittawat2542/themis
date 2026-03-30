---
title: First custom component
diataxis: tutorial
audience: users extending Themis for the first time
goal: Teach the smallest custom extension workflow with a parser, reducer, generator, or metric.
---

# First custom component

## What you will build

You will plug a custom component object into a Themis experiment and verify that it participates in compilation and execution as a first-class component.

## Prerequisites

- familiarity with `Experiment(...)`
- understanding that custom components must expose `component_id`, `version`, and `fingerprint()`

## Steps

1. Review a small custom component example.
2. Run the example to verify it produces a completed result.
3. Inspect the returned `score_ids` or other evidence that the component participated in execution.

```python
--8<-- "examples/docs/custom_parser.py"
```

```python
--8<-- "examples/docs/custom_reducer.py"
```

## Expected results

--8<-- "docs/_snippets/tutorials/first-custom-component-outcome.md"

## Common failure points

- forgetting `fingerprint()`
- overloading the component with orchestration logic that belongs in Themis

## Next steps

- [Author custom components](../how-to/author-custom-components.md)
- [Extension boundaries](../explanation/extension-boundaries.md)

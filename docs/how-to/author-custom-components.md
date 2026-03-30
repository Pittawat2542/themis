---
title: Author custom components
diataxis: how-to
audience: users extending Themis
goal: Show how to write custom generators, parsers, reducers, and metrics correctly.
---

# Author custom components

Goal: plug your own runtime behavior into Themis without changing the orchestration core.

When to use this:

Use this guide when builtin components are close but not sufficient and the gap belongs to a runtime extension point.

## Procedure

Start with the smallest protocol that solves your need:

- `Generator` for candidate production
- `Parser` for reduced-output normalization
- `CandidateReducer` for selection or synthesis after fan-out
- `PureMetric`, `LLMMetric`, `SelectionMetric`, or `TraceMetric` for scoring

Review the runnable examples:

```python
--8<-- "examples/docs/custom_generator.py"
```

```python
--8<-- "examples/docs/custom_parser.py"
```

```python
--8<-- "examples/docs/custom_reducer.py"
```

```python
--8<-- "examples/docs/custom_metric.py"
```

--8<-- "docs/_snippets/how-to/author-components-note.md"

## Variants

- simple deterministic scoring: `PureMetric`
- workflow-backed evaluation: `LLMMetric` or `SelectionMetric`
- artifact-aware generation: `Generator` plus trace/conversation payloads

## Expected result

Your component should compile and run as a first-class Themis component with a stable identity.

## Troubleshooting

- [Protocols reference](../reference/protocols.md)
- [Extension boundaries](../explanation/extension-boundaries.md)

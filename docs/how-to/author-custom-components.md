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

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Simple deterministic scoring | Parsed output alone is enough to decide correctness | Less flexible than workflow-backed judging for subjective tasks | `PureMetric` |
| Workflow-backed evaluation | A judge model or richer workflow should score the output | Higher latency and judge-model dependencies | `LLMMetric`, `SelectionMetric` |
| Artifact-aware generation | Generation should emit trace or conversation artifacts for later inspection | More generator responsibility and more stored artifacts | `Generator`, `GenerationResult.trace`, `GenerationResult.conversation` |

## Expected result

Your component should compile and run as a first-class Themis component with a stable identity.

## Troubleshooting

- [Protocols reference](../reference/protocols.md)
- [Extension boundaries](../explanation/extension-boundaries.md)

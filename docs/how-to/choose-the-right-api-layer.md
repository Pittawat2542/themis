---
title: Choose the right API layer
diataxis: how-to
audience: active Themis users
goal: Help readers pick the smallest API layer that still matches their workflow.
---

# Choose the right API layer

Goal: choose between `evaluate(...)`, `Experiment(...)`, and custom extension protocols.

When to use this:

Use this guide when you already know the problem you want to solve but you are unsure how much of the Themis surface you need to adopt.

## Procedure

Start with `evaluate(model=..., data=..., metric=..., ...)` if all of these are true:

- you are writing a short Python script
- your run can be described inline with a model, data, metrics, and small optional overrides
- you do not need `compile()`, `replay()`, or config-file loading

Choose `Experiment(...)` if any of these are true:

- you need an explicit compiled artifact
- you want a reusable experiment definition
- you need config loading, persisted stores, or replay support

Choose custom protocols if builtin components are not sufficient and your logic belongs inside generation, reduction, parsing, or scoring.

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Shortest possible local script | You want one small Python entry point for a straightforward run | Less explicit control over compile, replay, and config loading | `evaluate(model=..., data=..., metric=..., ...)` |
| Most reusable user workflow | You want a reusable experiment definition with compile, run, replay, and store control | More structure than a one-off script | `Experiment(...)`, `Experiment.compile()`, `Experiment.run()` |
| Advanced extensibility | Builtins are close but not sufficient and custom runtime behavior is required | Highest implementation cost and more protocol knowledge | `Generator`, `Parser`, `CandidateReducer`, metric protocols |

## Expected result

You should have a clear starting layer and know which later docs to follow without translating architecture terms into workflow decisions.

## Troubleshooting

- [Start Here](../start-here/index.md)
- [API layer model](../explanation/api-layer-model.md)
- [Python API reference](../reference/experiment-lifecycle.md)

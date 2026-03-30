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

Start with `evaluate(...)` if all of these are true:

- you are writing a short Python script
- your run can be described inline with config objects and datasets
- you do not need `compile()`, `rejudge()`, or config-file loading

Choose `Experiment(...)` if any of these are true:

- you need an explicit compiled artifact
- you want a reusable experiment definition
- you need config loading, persisted stores, or rejudge support

Choose custom protocols if builtin components are not sufficient and your logic belongs inside generation, reduction, parsing, or scoring.

## Variants

- shortest possible local script: `evaluate(...)`
- most reusable user workflow: `Experiment(...)`
- advanced extensibility: `Generator`, `Parser`, `CandidateReducer`, and metric protocols

## Expected result

You should have a clear starting layer and know which later docs to follow without translating architecture terms into workflow decisions.

## Troubleshooting

- [Start Here](../start-here/index.md)
- [API layer model](../explanation/api-layer-model.md)
- [Python API reference](../reference/experiment-lifecycle.md)

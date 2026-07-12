---
title: Choose the right API layer
diataxis: how-to
audience: active Themis users
goal: Help readers pick the smallest API layer that still matches their workflow.
---

# Choose the right API layer

Goal: choose between `Experiment(...)`, config/CLI, and custom extension protocols.

When to use this:

Use this guide when you already know the problem you want to solve but you are unsure how much of the Themis surface you need to adopt.

## Procedure

Start with `Experiment(...)` if all of these are true:

- you are writing a reviewed Python module or a short local debugging script
- your run can be described inline with a model, data, metrics, and small optional overrides
- you want direct access to `compile()`, `run()`, `replay()`, and store control

Choose config and CLI if any of these are true:

- you want automation or environment-specific overrides around reviewed Python experiment code
- you want shell-friendly automation
- you need worker-pool or batch submission

Choose custom protocols if builtin components are not sufficient and your logic belongs inside generation, reduction, parsing, or scoring.

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Python-authored run | You want a reusable canonical experiment definition with compile, run, replay, and store control | More structure than the removed one-call helper | `Experiment(...)`, `Experiment.compile()`, `Experiment.run()` |
| Config and CLI workflow | You want automation, overrides, or worker submission around importable experiment code | Automation surface only; the Python definition remains canonical | launcher config, `themis run`, `themis submit` |
| Advanced extensibility | Builtins are close but not sufficient and custom runtime behavior is required | Highest implementation cost and more protocol knowledge | `Generator`, `Parser`, `CandidateReducer`, metric protocols |

## Expected result

You should have a clear starting layer and know which later docs to follow without translating architecture terms into workflow decisions.

## Troubleshooting

- [Start Here](../start-here/index.md)
- [API layer model](../explanation/api-layer-model.md)
- [Python API reference](../reference/experiment-lifecycle.md)

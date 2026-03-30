---
title: Choose your API layer
diataxis: landing
audience: users deciding how to start authoring runs
goal: Help readers choose between evaluate, Experiment, and custom extension protocols.
---

# Choose your API layer

Use `evaluate(...)` when you want the shortest path from a dataset and a few config objects to a completed run. It is best for quick scripts and quick local experiments.

Use `Experiment(...)` when you want an explicit compiled object, access to `compile()`, `run()`, `rejudge()`, config-file loading, or long-lived experiment definitions. This is the primary surface for most serious work.

Use custom extension protocols when builtin generators, parsers, reducers, or metrics are not sufficient and you need to plug your own behavior into the runtime.

Decision rule:

- shortest path: `evaluate(...)`
- reusable experiment: `Experiment(...)`
- custom runtime behavior: extension protocols

Next:

- learn by example in [First `evaluate(...)`](../tutorials/first-evaluate.md)
- understand the model in [API layer model](../explanation/api-layer-model.md)

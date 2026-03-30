---
title: First Experiment()
diataxis: tutorial
audience: new Themis users moving beyond the shortest API
goal: Teach explicit experiment authoring and compilation.
---

# First `Experiment(...)`

## What you will build

You will define generation, evaluation, storage, and seeds explicitly, compile them into a `RunSnapshot`, and execute the run.

## Prerequisites

- comfort with the first `evaluate(...)` tutorial
- base Themis install

## Steps

1. Create an explicit `Experiment(...)`.
2. Call `compile()` to inspect the stable `run_id`.
3. Run the experiment with an explicit `RuntimeConfig`.

```python
--8<-- "examples/docs/first_experiment.py"
```

## Expected results

--8<-- "docs/_snippets/tutorials/first-experiment-outcome.md"

## Common failure points

- confusing `runtime` execution controls with identity-bearing inputs
- expecting `compile()` to execute work

## Next steps

- [First persisted run](first-persisted-run.md)
- [Compile vs run](../explanation/compile-vs-run.md)

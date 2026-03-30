---
title: First evaluate()
diataxis: tutorial
audience: new Themis users
goal: Teach the smallest end-to-end Themis evaluation using the Layer 1 Python API.
---

# First `evaluate(...)`

## What you will build

You will run a single deterministic evaluation from Python using builtin generation, parsing, and scoring components.

## Prerequisites

- base Themis install
- no provider extras required
- basic familiarity with running a Python script

## Steps

1. Read the example below.
2. Run it as a standalone script or import `run_example()`.
3. Inspect the returned `run_id` and `status`.

```python
--8<-- "examples/docs/first_evaluate.py"
```

## Expected results

--8<-- "docs/_snippets/tutorials/first-evaluate-outcome.md"

## Common failure points

- using a different expected output than the builtin demo generator returns
- assuming `memory` storage can be reopened from another process

## Next steps

- [First `Experiment(...)`](first-experiment.md)
- [Choose your API layer](../start-here/choose-your-api-layer.md)

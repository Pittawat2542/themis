---
title: First external execution
diataxis: tutorial
audience: users learning deferred execution flows
goal: Teach the worker-pool submission flow from config to worker execution.
---

# First external execution

## What you will build

You will submit an experiment through the worker-pool flow, create a manifest, and execute one worker cycle against the queued request.

## Prerequisites

- familiarity with config-driven experiments
- writable local filesystem

## Steps

1. Create a small config-backed experiment.
2. Submit it into the worker-pool queue.
3. Run a worker cycle and inspect the resulting run status.

```python
--8<-- "examples/docs/external_execution.py"
```

## Expected results

--8<-- "docs/_snippets/tutorials/first-external-execution-outcome.md"

## Common failure points

- trying to submit custom components that are neither builtin ids nor importable config symbols
- using `memory` storage when the workflow must survive process boundaries

## Next steps

- [Use submit, worker, and batch](../how-to/use-submit-worker-and-batch.md)
- [Store backend model](../explanation/store-backend-model.md)

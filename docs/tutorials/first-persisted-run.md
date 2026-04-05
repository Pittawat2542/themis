---
title: First persisted run
diataxis: tutorial
audience: users who need runs to survive process boundaries
goal: Teach persisted local execution with SQLite plus inspection and reporting.
---

# First persisted run

## What you will build

You will run Themis against SQLite, reopen the stored state, and export a report without regenerating the run.

## Prerequisites

- familiarity with `Experiment(...)`
- writable local filesystem

## Steps

1. Create a SQLite-backed experiment.
2. Execute it with an explicit store.
3. Inspect execution state and export a Markdown report.

```python
--8<-- "examples/docs/persisted_run.py"
```

## Expected results

--8<-- "docs/_snippets/tutorials/persisted-run-outcome.md"

## Common failure points

- using `memory` storage when later inspection is required
- forgetting that report/export/compare workflows need persisted state

## Next steps

- [First LLM-judged evaluation](first-llm-judged-evaluation.md)
- [Choose your storage backend](../start-here/choose-your-storage-backend.md)

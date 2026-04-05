---
title: First LLM-judged evaluation
diataxis: tutorial
audience: users learning workflow-backed metrics
goal: Teach judge-backed evaluation and artifact inspection without external providers.
---

# First LLM-judged evaluation

## What you will build

You will run a workflow-backed metric with builtin demo judges and inspect the stored evaluation execution.

## Prerequisites

- familiarity with persisted runs
- base Themis install

## Steps

1. Configure a workflow-backed metric and demo judge models.
2. Run the experiment into an in-memory store.
3. Inspect the stored evaluation execution for judge calls and scores.

```python
--8<-- "examples/docs/llm_judged_evaluation.py"
```

## Expected results

--8<-- "docs/_snippets/tutorials/llm-judged-outcome.md"

## Common failure points

- expecting workflow-backed metrics to behave like pure metrics
- forgetting to provide judge models for workflow-backed scoring

## Next steps

- [First advanced run](first-advanced-run.md)
- [Use workflow-backed metrics](../how-to/use-workflow-backed-metrics.md)

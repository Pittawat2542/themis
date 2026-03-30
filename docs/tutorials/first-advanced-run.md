---
title: First advanced run
diataxis: tutorial
audience: users combining multiple execution stages
goal: Teach multi-candidate generation, reduction, and mixed metrics in one run.
---

# First advanced run

## What you will build

You will run multiple candidates per case, reduce them to one reduced candidate, and score the result with both pure and workflow-backed metrics.

## Prerequisites

- familiarity with workflow-backed metrics
- understanding of candidate fan-out

## Steps

1. Configure `num_samples` greater than one.
2. Keep a reducer in the generation stage.
3. Add both pure and workflow-backed metrics to evaluation.

```python
--8<-- "examples/docs/advanced_run.py"
```

## Expected results

--8<-- "docs/_snippets/tutorials/advanced-run-outcome.md"

## Common failure points

- expecting multiple candidates to appear when `num_samples` is left at `1`
- mixing up reducer responsibilities with metric responsibilities

## Next steps

- [First custom component](first-custom-component.md)
- [Fan-out, reduction, parsing, and scoring](../explanation/fanout-reduction-parsing-scoring.md)

---
title: Reproduce and rejudge runs
diataxis: how-to
audience: users working with stored artifacts and workflow-backed metrics
goal: Show how to move persisted artifacts and re-run scoring without regenerating candidates.
---

# Reproduce and rejudge runs

Goal: export/import run artifacts and re-run workflow-backed metrics from stored upstream data.

When to use this:

Use this guide when generation should stay fixed but evaluation needs to move stores or be rerun.

## Procedure

```python
--8<-- "examples/docs/rejudge_bundle.py"
```

--8<-- "docs/_snippets/how-to/reproduce-note.md"

## Variants

- portable generation artifacts only: generation bundle export/import
- portable evaluation artifacts too: evaluation bundle export/import
- rerun workflow-backed metrics in place: `Experiment.rejudge()`

## Expected result

You should be able to move artifacts between stores and rerun workflow-backed metrics without regenerating candidates.

## Troubleshooting

- [Reproducibility and rejudge](../explanation/reproducibility-and-rejudge.md)
- [Stores and inspection reference](../reference/stores-and-inspection.md)

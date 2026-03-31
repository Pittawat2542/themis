---
title: Reproduce and rejudge runs
diataxis: how-to
audience: users working with stored artifacts and workflow-backed metrics
goal: Show how to move persisted artifacts and re-run scoring without regenerating candidates.
---

# Reproduce and rejudge runs

Goal: export/import run artifacts and replay downstream evaluation stages from stored upstream data.

When to use this:

Use this guide when generation should stay fixed but evaluation needs to move stores or be rerun from a downstream stage.

## Procedure

Use this sequence when you need to move evidence or rerun workflow-backed evaluation without regenerating candidates.

```mermaid
sequenceDiagram
    participant S as Source store
    participant B as Bundle files
    participant T as Target store
    participant E as "Experiment.replay(stage='judge')"
    S->>B: export generation/evaluation bundle
    B->>T: import bundle
    T->>E: reopen stored upstream artifacts
    E-->>T: write new evaluation executions
```

The crucial boundary is that generation artifacts stay fixed while evaluation moves or reruns.

```python
--8<-- "examples/docs/rejudge_bundle.py"
```

--8<-- "docs/_snippets/how-to/reproduce-note.md"

## Variants

- portable generation artifacts only: generation bundle export/import
- portable evaluation artifacts too: evaluation bundle export/import
- rerun workflow-backed metrics in place: `Experiment.replay(stage="judge")`
- rerun pure scoring from parsed outputs: `Experiment.replay(stage="score")`

## Expected result

You should be able to move artifacts between stores and replay downstream stages without regenerating candidates.

## Troubleshooting

- [Reproducibility and rejudge](../explanation/reproducibility-and-rejudge.md)
- [Stores and inspection reference](../reference/stores-and-inspection.md)

---
title: Artifacts and inspection
diataxis: explanation
audience: users debugging runs and evaluating stored outputs
goal: Explain what artifacts Themis stores and how they support later inspection.
---

# Artifacts and inspection

What it is: the persisted payload model for generation results, reduction outputs, parsed outputs, scores, evaluation executions, traces, conversations, and bundle records.

When it matters: whenever a score needs to be explained without rerunning the experiment.

What you provide: generators and workflows that return structured outputs and artifacts worth preserving.

What Themis provides: default persistence, blob refs, stage bundle helpers, and inspection APIs.

Stage artifacts are first-class persisted evidence. Generation, reduction, parse, score, and evaluation artifacts can all be exported, imported, and inspected later without inventing a parallel storage model.

Imported artifacts are normalized into the same event history used by locally executed runs. That means imported data participates in `resume`, `report`, `compare`, and replay flows exactly like artifacts produced inside one Themis process.

What to inspect when it goes wrong: `get_run_snapshot(...)`, `get_execution_state(...)`, `get_evaluation_execution(...)`, exported bundles, and report projections.

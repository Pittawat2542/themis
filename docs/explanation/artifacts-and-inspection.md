---
title: Artifacts and inspection
diataxis: explanation
audience: users debugging runs and evaluating stored outputs
goal: Explain what artifacts Themis stores and how they support later inspection.
---

# Artifacts and inspection

What it is: the persisted payload model for generation results, traces, conversations, evaluation executions, and bundle records.

When it matters: whenever a score needs to be explained without rerunning the experiment.

What you provide: generators and workflows that return structured outputs and artifacts worth preserving.

What Themis provides: default persistence, blob refs, bundle helpers, and inspection APIs.

What to inspect when it goes wrong: `get_run_snapshot(...)`, `get_execution_state(...)`, `get_evaluation_execution(...)`, exported bundles, and report projections.

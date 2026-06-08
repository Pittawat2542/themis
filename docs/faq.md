---
title: FAQ
diataxis: reference
audience: active Themis users
goal: Answer common questions and repeated confusion points quickly.
---

# FAQ

## Should I start with `Experiment(...)` or config/CLI?

Start with `Experiment(...)` in reviewed Python modules for canonical experiment definitions and local debugging. Use config and CLI when you need automation, environment-specific overrides, worker-pool submission, or batch execution.

## What changes a `run_id`?

Only identity-bearing inputs inside `RunSnapshot.identity` change `run_id`: dataset refs and fingerprints, component refs, candidate policy, judge config, workflow overrides, and seeds.

## Why does memory-backed replay require `store=...`?

The memory store only exists in-process. Replay needs stored upstream artifacts, so a memory-backed run must reuse the original store instance.

## When should I use SQLite instead of memory?

Use SQLite whenever the run needs to survive process boundaries or support later `resume`, `report`, `compare`, or `export` calls.

## What is stored by default?

The runtime persists generation results, evaluation executions, snapshots, execution state, and related artifacts needed for later inspection, reporting, bundle export/import, and replay flows.

## What is the difference between resume and replay?

The `themis resume` command reopens the stored run and prints status/progress. Continuing unfinished work happens when you run the same compiled `run_id` again with `existing_run_policy="auto"`. Replay re-runs downstream stages from stored upstream artifacts without regenerating candidates. `rejudge()` is the workflow-metric specialization of `replay(stage="judge")`.

## What is intentionally not supported yet?

Current non-goals and deferred areas are explicit:

- no native provider batch API orchestration
- no config diff tooling
- no first-class grid-search reuse
- no storage-efficiency redesign for very large artifacts
- cross-environment reproducibility still relies on user-managed dependency locking

## How should I add docs for a new public surface?

Update the relevant page in `docs/`, add or expand a runnable example when the change is user-facing, and make sure the coverage still appears in the docs inventory script and tests.

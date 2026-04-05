---
title: FAQ
diataxis: reference
audience: active Themis users
goal: Answer common questions and repeated confusion points quickly.
---

# FAQ

## Should I start with `evaluate(...)` or `Experiment(...)`?

Start with `evaluate(model=..., data=..., metric=..., ...)` for the smallest possible script. Move to `Experiment(...)` when you need `compile()`, `replay()`, config loading, or a reusable experiment definition.

## What changes a `run_id`?

Only identity-bearing inputs inside `RunSnapshot.identity` change `run_id`: dataset refs and fingerprints, component refs, candidate policy, judge config, workflow overrides, and seeds.

## Why does memory-backed replay require `store=...`?

The memory store only exists in-process. Replay needs stored upstream artifacts, so a memory-backed run must reuse the original store instance.

## When should I use SQLite instead of memory?

Use SQLite whenever the run needs to survive process boundaries or support later `resume`, `report`, `compare`, or `export` calls.

## What is stored by default?

The runtime persists generation results, evaluation executions, snapshots, execution state, and related artifacts needed for later inspection, reporting, bundle export/import, and replay flows.

## What is the difference between resume and replay?

Resume continues unfinished work for the same run. Replay re-runs downstream stages from stored upstream artifacts without regenerating candidates. `rejudge()` is the workflow-metric specialization of `replay(stage="judge")`.

## What is intentionally not supported yet?

Current non-goals and deferred areas are explicit:

- no native provider batch API orchestration
- no config diff tooling
- no first-class grid-search reuse
- no storage-efficiency redesign for very large artifacts
- no first-class long-term reproducibility/version-pinning workflow yet

## How should I add docs for a new public surface?

Update the relevant page in `docs/`, add or expand a runnable example when the change is user-facing, and make sure the coverage still appears in the docs inventory script and tests.

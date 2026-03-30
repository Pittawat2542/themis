---
title: FAQ
diataxis: reference
audience: active Themis users
goal: Answer common questions and repeated confusion points quickly.
---

# FAQ

## Should I start with `evaluate(...)` or `Experiment(...)`?

Start with `evaluate(...)` for the smallest possible script. Move to `Experiment(...)` when you need `compile()`, `rejudge()`, config loading, or a reusable experiment definition.

## What changes a `run_id`?

Only identity-bearing inputs inside `RunSnapshot.identity` change `run_id`: dataset refs and fingerprints, component refs, candidate policy, judge config, workflow overrides, and seeds.

## Why does memory-backed rejudge require `store=...`?

The memory store only exists in-process. Rejudge needs stored upstream artifacts, so a memory-backed run must reuse the original store instance.

## When should I use SQLite instead of memory?

Use SQLite whenever the run needs to survive process boundaries or support later `resume`, `report`, `compare`, or `export` calls.

## What is stored by default?

The runtime persists generation results, evaluation executions, and related artifacts needed for later inspection, reporting, bundle export/import, and workflow-backed rejudge flows.

## What is the difference between resume and rejudge?

Resume continues unfinished work for the same run. Rejudge re-runs workflow-backed metrics from stored upstream artifacts without regenerating candidates.

## How should I add docs for a new public surface?

Update the relevant page in `docs/`, add or expand a runnable example when the change is user-facing, and make sure the coverage still appears in the docs inventory script and tests.

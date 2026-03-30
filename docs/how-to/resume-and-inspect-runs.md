---
title: Resume and inspect runs
diataxis: how-to
audience: users debugging or continuing persisted work
goal: Show how to reopen persisted runs and inspect execution state safely.
---

# Resume and inspect runs

Goal: continue interrupted work and inspect stored execution state.

When to use this:

Use this guide when a run already exists and you want to inspect or continue it rather than starting from scratch.

## Procedure

1. Use a persistent store, typically SQLite.
2. Reopen the run by the same compiled `run_id`.
3. Inspect execution state before rerunning anything.
4. Use the CLI or Python helpers to examine progress and failures.

## Variants

- quick state summary: `themis quickcheck`
- explicit persisted state inspection: `get_execution_state(...)`
- report generation from the stored run: `Reporter` or `themis report`

## Expected result

You should know whether the run can be resumed, what already completed, and where failures occurred.

## Troubleshooting

- [Failure, retry, and resume](../explanation/failure-retry-and-resume.md)
- [CLI reference](../reference/cli.md)

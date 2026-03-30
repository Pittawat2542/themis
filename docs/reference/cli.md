---
title: CLI reference
diataxis: reference
audience: users operating Themis from the shell
goal: Document command groups, inputs, output shapes, and persistence expectations.
---

# CLI reference

## Command groups

- `quick-eval`
- `run`
- `submit`
- `resume`
- `estimate`
- `report`
- `quickcheck`
- `compare`
- `export`
- `init`
- `worker`
- `batch`

## Command behavior

- `run --config ...`: executes the experiment and prints JSON with `run_id`, `status`, and `metric_means`
- `resume --config ...`: inspects persisted state for the compiled `run_id`
- `estimate --config ...`: prints planner output for the compiled snapshot
- `quickcheck --config ...`: prints a compact status summary for a stored run
- `report --config ... --format ...`: exports JSON, Markdown, CSV, or LaTeX from persisted projections
- `compare --baseline-config ... --candidate-config ...`: compares benchmark projections from two persisted runs
- `submit --config ... --mode worker-pool|batch`: writes a manifest for deferred execution

## Subcommands

- `quick-eval inline`
- `quick-eval file`
- `quick-eval huggingface`
- `quick-eval benchmark`
- `export generation`
- `export evaluation`
- `worker run`
- `batch run`

## Output notes

JSON-producing commands generally emit compact machine-readable JSON to stdout. Commands that inspect stored runs require a persistent store unless the current process still owns the original memory store.

---
title: CLI reference
diataxis: reference
audience: users operating Themis from the shell
goal: Document command groups, inputs, output shapes, and persistence expectations.
---

# CLI reference

## Command groups

| Command | What it does | When to use it | Key inputs / constraints |
| --- | --- | --- | --- |
| `quick-eval` | Runs inline examples, files, Hugging Face datasets, or catalog benchmarks with minimal setup | You want the shortest path to an evaluation run | Trades flexibility for convenience |
| `run` | Compiles and executes a config automation surface | You want the main shell-driven runtime path | Accepts `--config` and optional `--until-stage` |
| `replay` | Re-runs downstream stages from stored upstream artifacts | You want to regenerate reduction, parse, score, or judge results without fresh generation | Requires persisted upstream artifacts |
| `rerun` | Re-runs a targeted subset of a stored run | You want to retry failed cases, specific cases, metadata slices, or selected metrics | Requires persisted state for the compiled `run_id` |
| `submit` | Writes deferred-execution manifests | You want worker-pool or batch execution instead of immediate in-process execution | Requires `--mode worker-pool` or `--mode batch` |
| `resume` | Reopens a stored run and prints status/progress JSON | You want to inspect whether persistent work is complete or pending | Depends on a persistent store; use `run` with `existing_run_policy=auto` to continue work |
| `estimate` | Prints planner and token-estimate output for a compiled snapshot | You want execution counts and token assumptions before running | Estimates are informational, not pricing |
| `report` | Exports summary-first reports in multiple formats | You want shareable output from a stored run | Requires a stored run and a format choice |
| `inspect` | Reads snapshots, state, or evaluation executions from the store | You want to diagnose or inspect persisted artifacts | Uses subcommands for each payload type |
| `quickcheck` | Prints a compact status summary for one stored run | You want a fast health check instead of a full report | Depends on persisted state |
| `compare` | Compares two persisted benchmark results | You want baseline vs candidate analysis across completed runs | Requires two config-backed runs |
| `export` | Writes stage-aware artifact bundles | You want portable generation or evaluation artifacts | CLI currently covers `generation` and `evaluation` bundles |
| `init` | Scaffolds starter files | You want help bootstrapping a new workflow from the shell | Keep using Python authoring if you need live objects |
| `worker` | Executes queued manifests in worker-pool mode | You want queue-driven deferred execution | Use with manifests produced by `submit --mode worker-pool` |
| `batch` | Executes explicit request manifests in batch mode | You want request-file driven deferred execution | Use with manifests produced by `submit --mode batch` |

## Command behavior

| Command | What it does | When to use it | Key inputs / constraints |
| --- | --- | --- | --- |
| `run --config ... [--until-stage ...]` | Executes the experiment and prints JSON with `run_id`, `status`, `completed_through_stage`, and `metric_means` | You want the main config-driven execution path | With `existing_run_policy=auto`, incomplete stored runs continue and completed stage-compatible runs are reused |
| `replay --config ... --stage reduce|parse|score|judge` | Re-runs downstream stages from stored upstream artifacts | You want fresh downstream scoring without new generation | Requires stored upstream artifacts |
| `rerun --config ... --stage generate|reduce|parse|score|judge` | Re-runs matching cases from the requested stage onward | You want targeted recovery instead of a full replay | Filter with `--failed-only`, `--case-id`, `--case-key`, `--metadata key=value`, and `--metric-id` |
| `resume --config ...` | Reopens the compiled `run_id` and prints stored status, completed stage, total cases, and completed cases | You want to inspect stored progress before deciding whether to run more work | This command does not execute stages |
| `estimate --config ...` | Prints planner output, task counts, token estimates, and estimate assumptions | You want pre-run sizing and cost-model inputs | No pricing is applied by Themis |
| `quickcheck --config ...` | Prints a compact status summary for a stored run | You want a quick operational check | Less detail than `report` or `inspect` |
| `report --config ... --format ...` | Exports JSON projections or Markdown, CSV, and LaTeX metric summaries | You want a shareable report from stored state | Requires a supported `--format` |
| `inspect snapshot --config ...` | Prints the stored `RunSnapshot` | You want identity and provenance details | Snapshot inspection is read-only |
| `inspect state --config ...` | Prints stored execution state | You want stage completion, counters, and failure visibility | Persistent storage is required outside a still-live memory store |
| `inspect evaluation --config ... --case-id ... --metric-id ...` | Prints one stored workflow execution | You want judge prompts, responses, or workflow artifacts for a specific case | Only applies to workflow-backed metrics |
| `compare --baseline-config ... --candidate-config ...` | Compares persisted benchmark projections from two runs | You want paired benchmark analysis | Both runs must already exist |
| `submit --config ... --mode worker-pool|batch` | Writes a manifest for deferred execution | You want worker-pool or batch handoff instead of direct execution | Follow with `worker run` or `batch run` |

## Subcommands

| Command | What it does | When to use it | Key inputs / constraints |
| --- | --- | --- | --- |
| `quick-eval inline` | Runs a small inline dataset or prompt set | You want the shortest shell-driven smoke test | Best for local examples |
| `quick-eval file` | Loads input data from a file for a quick evaluation run | You want a lightweight file-backed run | Less structured than a reviewed Python `Experiment(...)` |
| `quick-eval huggingface` | Loads a dataset from Hugging Face for quick evaluation | You want a short path to remote dataset-backed evaluation | Requires the `datasets` extra |
| `quick-eval benchmark` | Runs a shipped named benchmark recipe | You want catalog convenience from the shell | Requires benchmark-specific extras such as dataset access or code execution backends |
| `inspect snapshot` | Prints the stored compiled snapshot | You want identity and provenance details for a run | Requires a stored run |
| `inspect state` | Prints stored execution state | You want progress and failure details | Requires a stored run |
| `inspect evaluation` | Prints one stored workflow evaluation artifact | You want per-case judge execution details | Requires workflow-backed metrics |
| `export generation` | Exports generation artifacts to a portable bundle | You want portable candidate outputs | CLI export currently covers this stage directly |
| `export evaluation` | Exports evaluation workflow artifacts to a portable bundle | You want portable judge execution artifacts | Best for workflow-backed metrics |
| `worker run` | Pulls manifests from the worker queue and executes them | You want queue-driven deferred execution | Requires a queue root |
| `batch run` | Executes one explicit batch request manifest | You want request-file driven deferred execution | Requires a request manifest path |

## Output notes

JSON-producing commands generally emit compact machine-readable JSON to stdout. Commands that inspect stored runs require a persistent store unless the current process still owns the original memory store.

`report --format csv` and `report --format latex` emit compact metric summary tables with `metric_id`, `count`, `mean`, `min`, `max`, `ci_lower`, and `ci_upper`. `report --format json` includes the full stored projections plus `stats_summary`; raw per-case score rows remain in `benchmark_result.score_rows`.

## Current CLI boundary

| Surface | Current behavior | Use instead when | Notes |
| --- | --- | --- | --- |
| `run` | Exposes `--until-stage` directly in the CLI | You want deliberate stage-boundary stopping from the shell | Good fit for later replay or export |
| `export` | Exposes only `generation` and `evaluation` bundle export in the CLI | You need reduction, parse, or score bundles | Use the Python export helpers for those intermediate stages |
| Reduction, parse, and score bundle handoff | Supported by the runtime but not surfaced in the CLI | You need portable intermediate-stage artifacts | Use Python helpers until the CLI grows those commands |

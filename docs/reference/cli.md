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
| `run` | Compiles and executes a config-backed experiment | You want the main config-driven runtime path | Accepts `--config` and optional `--until-stage` |
| `replay` | Re-runs downstream stages from stored upstream artifacts | You want to regenerate reduction, parse, score, or judge results without fresh generation | Requires persisted upstream artifacts |
| `submit` | Writes deferred-execution manifests | You want worker-pool or batch execution instead of immediate in-process execution | Requires `--mode worker-pool` or `--mode batch` |
| `resume` | Reopens a stored run and continues according to runtime policy | You want to continue interrupted or partially completed persistent work | Depends on a persistent store |
| `estimate` | Prints planner and token-estimate output for a compiled snapshot | You want execution counts and token assumptions before running | Estimates are informational, not pricing |
| `report` | Exports score and outcome reports in multiple formats | You want shareable output from a stored run | Requires a stored run and a format choice |
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
| `run --config ... [--until-stage ...]` | Executes the experiment and prints JSON with `run_id`, `status`, `completed_through_stage`, and `metric_means` | You want the main config-driven execution path | `--until-stage` stops intentionally at a stage boundary |
| `replay --config ... --stage reduce|parse|score|judge` | Re-runs downstream stages from stored upstream artifacts | You want fresh downstream scoring without new generation | Requires stored upstream artifacts |
| `resume --config ...` | Reopens the compiled `run_id` and continues if the store shows pending work | You want to continue interrupted persistent work | Depends on a persistent store |
| `estimate --config ...` | Prints planner output, task counts, token estimates, and estimate assumptions | You want pre-run sizing and cost-model inputs | No pricing is applied by Themis |
| `quickcheck --config ...` | Prints a compact status summary for a stored run | You want a quick operational check | Less detail than `report` or `inspect` |
| `report --config ... --format ...` | Exports JSON, Markdown, CSV, or LaTeX projections | You want a shareable report from stored state | Requires a supported `--format` |
| `inspect snapshot --config ...` | Prints the stored `RunSnapshot` | You want identity and provenance details | Snapshot inspection is read-only |
| `inspect state --config ...` | Prints stored execution state | You want stage completion, counters, and failure visibility | Persistent storage is required outside a still-live memory store |
| `inspect evaluation --config ... --case-id ... --metric-id ...` | Prints one stored workflow execution | You want judge prompts, responses, or workflow artifacts for a specific case | Only applies to workflow-backed metrics |
| `compare --baseline-config ... --candidate-config ...` | Compares persisted benchmark projections from two runs | You want paired benchmark analysis | Both runs must already exist |
| `submit --config ... --mode worker-pool|batch` | Writes a manifest for deferred execution | You want worker-pool or batch handoff instead of direct execution | Follow with `worker run` or `batch run` |

## Subcommands

| Command | What it does | When to use it | Key inputs / constraints |
| --- | --- | --- | --- |
| `quick-eval inline` | Runs a small inline dataset or prompt set | You want the shortest shell-driven smoke test | Best for local examples |
| `quick-eval file` | Loads input data from a file for a quick evaluation run | You want a lightweight file-backed run | Less structured than a full config-driven experiment |
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

`report` and exported score tables include `outcome`, `error_category`, `error_message`, and `details` columns alongside metric values.

## Current CLI boundary

| Surface | Current behavior | Use instead when | Notes |
| --- | --- | --- | --- |
| `run` | Exposes `--until-stage` directly in the CLI | You want deliberate stage-boundary stopping from the shell | Good fit for later replay or export |
| `export` | Exposes only `generation` and `evaluation` bundle export in the CLI | You need reduction, parse, or score bundles | Use the Python export helpers for those intermediate stages |
| Reduction, parse, and score bundle handoff | Supported by the runtime but not surfaced in the CLI | You need portable intermediate-stage artifacts | Use Python helpers until the CLI grows those commands |

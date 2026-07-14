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
| `resume` | Reopens a stored run and prints status/progress JSON | You want to inspect whether persistent work is complete or pending | Depends on a persistent store; use `run` with `existing_run_policy=reuse` to continue work |
| `estimate` | Prints planner and token-estimate output for a compiled snapshot | You want execution counts and token assumptions before running | Estimates are informational, not pricing |
| `report` | Exports summary-first reports in multiple formats | You want shareable output from a stored run | Requires a stored run and a format choice |
| `inspect` | Reads snapshots, state, or evaluation executions from the store | You want to diagnose or inspect persisted artifacts | Uses subcommands for each payload type |
| `quickcheck` | Prints a compact status summary for one stored run | You want a fast health check instead of a full report | Depends on persisted state |
| `compare` | Compares two persisted benchmark results | You want baseline vs candidate analysis across completed runs | Requires two config-backed runs |
| `compare-runs` and `compare-latest` | Compares exact run ids or latest labeled runs | You want a Score Claim over already persisted Evidence | Uses matched case keys and reports missing rows |
| `suite` | Lists, inspects, or runs named benchmark suites | You want suite automation without changing run identity | Each expanded suite item creates a normal run |
| `export` | Writes stage-aware artifact bundles | You want portable generation or evaluation artifacts | CLI currently covers `generation` and `evaluation` bundles |
| `init` | Scaffolds starter files | You want help bootstrapping a new workflow from the shell | Keep using Python authoring if you need live objects |
| `worker` | Executes queued manifests in worker-pool mode | You want queue-driven deferred execution | Use with manifests produced by `submit --mode worker-pool` |
| `batch` | Executes explicit request manifests in batch mode | You want request-file driven deferred execution | Use with manifests produced by `submit --mode batch` |

## Command behavior

| Command | What it does | When to use it | Key inputs / constraints |
| --- | --- | --- | --- |
| `run --config ... [--until-stage ...]` | Executes the experiment and prints JSON with `run_id`, `status`, `completed_through_stage`, and `metric_means` | You want the main config-driven execution path | With `existing_run_policy=reuse`, incomplete stored runs continue and completed stage-compatible runs are reused |
| `replay --config ... --stage reduce|parse|score|judge` | Re-runs downstream stages from stored upstream artifacts | You want fresh downstream scoring without new generation | Requires stored upstream artifacts |
| `rerun --config ... --stage generate|reduce|parse|score|judge` | Re-runs matching cases from the requested stage onward | You want targeted recovery instead of a full replay | Filter with `--failed-only`, `--case-id`, `--case-key`, `--metadata key=value`, and `--metric-id` |
| `resume --config ...` | Reopens the compiled `run_id` and prints stored status, completed stage, total cases, and completed cases | You want to inspect stored progress before deciding whether to run more work | This command does not execute stages |
| `estimate --config ...` | Prints planner output, task counts, token estimates, resource plan, and estimate assumptions | You want pre-run sizing, backend requirements, and cost-model inputs | No pricing is applied by Themis |
| `quickcheck --config ...` | Prints a compact status summary for a stored run | You want a quick operational check | Less detail than `report` or `inspect` |
| `report --config ... --format ...` | Exports JSON projections or Markdown, CSV, and LaTeX metric summaries | You want a shareable report from stored state | Requires a supported `--format` |
| `inspect snapshot --config ...` | Prints the stored `RunSnapshot` | You want identity and provenance details | Snapshot inspection is read-only |
| `inspect state --config ...` | Prints stored execution state | You want stage completion, counters, and failure visibility | Persistent storage is required outside a still-live memory store |
| `inspect evaluation --config ... --case-id ... --metric-id ...` | Prints one stored workflow execution | You want judge prompts, responses, or workflow artifacts for a specific case | Only applies to workflow-backed metrics |
| `compare --baseline-config ... --candidate-config ...` | Compares persisted benchmark projections from two runs | You want paired benchmark analysis | Both runs must already exist |
| `compare-runs --config ... --baseline-run-id ... --candidate-run-id ...` | Emits a pairwise Score Claim for two exact run ids | You already know the evidence runs | Missing rows are reported and excluded from paired deltas |
| `compare-latest --config ... --baseline-label ...` | Compares latest labeled runs | You use registry baseline labels | Requires matching persisted run records |
| `submit --config ... --mode worker-pool|batch` | Writes a manifest for deferred execution | You want worker-pool or batch handoff instead of direct execution | Follow with `worker run` or `batch run` |

## Subcommands

| Command | What it does | When to use it | Key inputs / constraints |
| --- | --- | --- | --- |
| `quick-eval inline --input-json ... [--expected-output-json ...]` | Runs one inline JSON case | You want the shortest shell-driven smoke test | Uses the builtin demo generator, JSON parser, and exact-match metric |
| `quick-eval file --path ...` | Loads JSONL input from a file for a quick evaluation run | You want a lightweight file-backed run | Less structured than a reviewed Python `Experiment(...)` |
| `quick-eval huggingface --dataset ... --split ... --input-field ...` | Loads a dataset from Hugging Face for quick evaluation | You want a short path to remote dataset-backed evaluation | Requires the `datasets` extra; output and case-id fields are optional |
| `quick-eval benchmark --name ...` | Runs a shipped named benchmark recipe | You want catalog convenience from the shell | Requires benchmark-specific extras such as dataset access or code execution backends |
| `suite list` | Lists suite ids | You want to discover named suites | Process-local registrations are included |
| `suite inspect SUITE-ID` | Prints a suite and its expanded experiment items | You want to review suite membership before execution | Expansion preserves one experiment per item |
| `suite run SUITE-ID` | Runs every expanded suite item | You want suite convenience from the shell | Uses the default in-memory store |
| `inspect snapshot` | Prints the stored compiled snapshot | You want identity and provenance details for a run | Requires a stored run |
| `inspect state` | Prints stored execution state | You want progress and failure details | Requires a stored run |
| `inspect evaluation` | Prints one stored workflow evaluation artifact | You want per-case judge execution details | Requires workflow-backed metrics |
| `inspect case` | Prints one case audit | You want case-level evidence across stages | Requires `--run-id`, `--case-id`, and optional `--dataset-id` |
| `inspect runs` | Queries stored run records | You want registry filtering by ids, labels, tags, status, lineage, or timestamps | All filters are optional except `--config` |
| `inspect run-record` | Prints one exact run record | You want registry metadata for a known run | Requires `--run-id` |
| `inspect lineage` | Prints lineage for a run id or baseline label | You want parent-attempt history | Supply `--run-id` or `--baseline-label` |
| `inspect telemetry` | Prints the provider telemetry summary for a run | You want request, token, latency, retry, or failure aggregates | Requires `--run-id` |
| `export generation` | Exports generation artifacts to a portable bundle | You want portable candidate outputs | CLI export currently covers this stage directly |
| `export evaluation` | Exports evaluation workflow artifacts to a portable bundle | You want portable judge execution artifacts | Best for workflow-backed metrics |
| `worker run --definition-root ... [--queue-root ...]` | Pulls manifests from the worker queue and executes them | You want queue-driven deferred execution | `--definition-root` is required; the queue defaults to `runs/queue` |
| `batch run --request ... --definition-root ...` | Executes one explicit batch request manifest | You want request-file driven deferred execution | Both arguments are required |

## Output notes

Ordinary JSON-producing commands emit a versioned envelope:

```json
{
  "schema_version": "1",
  "command": "run",
  "data": {"run_id": "...", "status": "completed"}
}
```

Text, CSV, Markdown, and LaTeX output is unchanged. Artifact export commands
emit raw bundle documents because each bundle already carries its own schema
version. Commands that inspect stored runs require a persistent store unless the
current process still owns the original memory store.

Workers verify that executable definitions remain inside configured roots and
match the submitted SHA-256 digest before import. Claims use renewable leases
and move to `failed/` after retry exhaustion. Cross-trust-boundary queues should
require HMAC signatures and obtain the signing key from an environment variable.

`report --format csv` and `report --format latex` emit compact metric summary tables with `metric_id`, `count`, `mean`, `min`, `max`, `ci_lower`, and `ci_upper`. `report --format json` includes the full stored projections plus `stats_summary`; raw per-case score rows remain in `benchmark_result.score_rows`.

## Current CLI boundary

| Surface | Current behavior | Use instead when | Notes |
| --- | --- | --- | --- |
| `run` | Exposes `--until-stage` directly in the CLI | You want deliberate stage-boundary stopping from the shell | Good fit for later replay or export |
| `export` | Exposes only `generation` and `evaluation` bundle export in the CLI | You need reduction, parse, or score bundles | Use the Python export helpers for those intermediate stages |
| Reduction, parse, and score bundle handoff | Supported by the runtime but not surfaced in the CLI | You need portable intermediate-stage artifacts | Use Python helpers until the CLI grows those commands |

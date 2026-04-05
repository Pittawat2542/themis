---
title: Compare, export, and report
diataxis: how-to
audience: users analyzing finished runs
goal: Show how to produce reports, bundle exports, and paired comparisons from stored runs.
---

# Compare, export, and report

Goal: generate portable output from stored runs and compare two completed experiments.

When to use this:

Use this guide when execution is already done and the next task is inspection, reporting, export, or comparison.

## Procedure

Use:

- `Reporter.export_json(...)`, `export_markdown(...)`, `export_csv(...)`, and `export_latex(...)`
- `themis report --config ... --format ...`
- `themis compare --baseline-config ... --candidate-config ...`
- `themis export generation|evaluation --config ...`

Portable artifact handoff is stage-aware:

- generation artifacts: `export_generation_bundle(...)`
- reduction artifacts: `export_reduction_bundle(...)`
- parse artifacts: `export_parse_bundle(...)`
- pure-score artifacts: `export_score_bundle(...)`
- workflow execution artifacts: `export_evaluation_bundle(...)`

Reporting output is now outcome-aware. `benchmark_result.score_rows` and CSV exports include:

- `outcome`: `correct`, `incorrect`, or `error`
- `error_category`: for example `parse_failure`, `parse_null`, `parse_invalid`, `evaluation_failure`, `evaluation_partial_failure`, or `score_failure`
- `error_message`: the stored failure reason when the row is an error
- `details`: metric-specific structured payload for downstream qualitative analysis

Outside Themis workflows:

- external leaderboard construction: export JSON or CSV, then aggregate `benchmark_result` across runs in your notebook, warehouse, or dashboard job
- prompt sweep aggregation: run one experiment per prompt variant, then compare the exported `benchmark_result` payloads outside Themis
- external LM-judge handoff: export generation artifacts, run your own judge or provider batch API outside Themis, then convert the results back with a custom script and import the matching evaluation-stage artifacts

Artifact-interop support for `R3-R4` is intentional but scriptable. Themis owns the persistent stage artifacts and downstream replay path, while the mapping from an external job result into Themis-compatible bundle records still happens in your code.

Use this output shape when you build downstream leaderboards or prompt-sweep dashboards outside Themis. Themis owns the per-run read models; cross-run aggregation is expected to happen in your notebook, warehouse, or reporting job.

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| One-run reporting | You want human-readable or machine-readable output for one completed run | Does not compare multiple runs by itself | `Reporter`, `themis report --config ... --format ...` |
| Portable artifact handoff | Another system should consume stored stage artifacts | You need to manage exported bundles explicitly | `themis export generation|evaluation`, `export_generation_bundle(...)`, `export_evaluation_bundle(...)` |
| Side-by-side benchmark comparison | You want baseline vs candidate analysis inside Themis | Requires two completed persisted runs | `themis compare --baseline-config ... --candidate-config ...` |
| External leaderboard or dashboard | Aggregation belongs in notebooks, warehouses, or dashboards outside Themis | You own cross-run aggregation logic | JSON or CSV exports, `BenchmarkResult` payloads |
| Prompt sweep analysis | You are comparing multiple prompt variants over repeated runs | Sweep aggregation still happens outside Themis | Exported `benchmark_result` payloads |
| External judge pipeline | Judging should happen outside Themis, then come back as imported artifacts | Requires custom mapping code at the handoff boundary | Export/import bundle helpers |

## Expected result

You should have machine-readable or human-readable output that can be shared without rerunning the experiment.

## Troubleshooting

- [Reporting and read models](../explanation/reporting-and-read-models.md)
- [CLI reference](../reference/cli.md)

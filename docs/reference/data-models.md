---
title: Data models reference
diataxis: reference
audience: users inspecting run payloads and contributors extending the runtime
goal: Document runtime models, snapshot models, and projection/read-model types.
---

# Data models reference

## Important payloads to inspect directly

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `RunEstimate` | Planning model | You want task counts plus token estimate fields such as `estimated_generation_input_tokens`, `estimated_generation_output_tokens`, `estimated_judge_prompt_tokens`, `estimated_judge_output_tokens`, `estimated_total_tokens`, and `assumptions` | Informational only; pair with your own pricing model |
| `BenchmarkScoreRow` | Per-case score model | You want one scored row with `outcome`, `value`, `error_category`, `error_message`, and `details` | Includes additive `dataset_id` and `case_key` fields so duplicate `case_id`s across datasets stay distinguishable |
| `BenchmarkResult` | Aggregated benchmark model | You want combined `score_rows`, `metric_means`, `outcome_counts`, and `error_counts` | Best for reporting and comparison |

## Dataset-scoped case identity

Case-level runtime and read-model payloads now carry three related fields:

- `case_id`: the original case identifier from the dataset
- `dataset_id`: the dataset that contributed the case
- `case_key`: the dataset-scoped internal identity used by execution state, resume, and projections

`CaseResult`, `BenchmarkScoreRow`, `TimelineEntry`, and trace records all expose `dataset_id` and `case_key` additively. New runs persist these fields directly. Older stored runs remain readable via `case_key` or `case_id` fallback, but only new runs provide full duplicate-`case_id` safety across datasets.

Core runtime and output models:

::: themis.core.models

Prompt-oriented models:

::: themis.core.prompts

Run state, results, and bundle models:

::: themis.core.results

Snapshot and identity models:

::: themis.core.snapshot

Projection/read-model types:

::: themis.core.read_models

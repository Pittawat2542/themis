---
title: Data models reference
diataxis: reference
audience: users inspecting run payloads and contributors extending the runtime
goal: Document runtime models, snapshot models, and projection/read-model types.
---

# Data models reference

Important payloads to inspect directly:

- `RunEstimate`: task counts plus token estimate fields such as `estimated_generation_input_tokens`, `estimated_generation_output_tokens`, `estimated_judge_prompt_tokens`, `estimated_judge_output_tokens`, `estimated_total_tokens`, and `assumptions`
- `BenchmarkScoreRow`: one row with `outcome`, `value`, `error_category`, `error_message`, and `details`
- `BenchmarkResult`: aggregated `score_rows`, `metric_means`, `outcome_counts`, and `error_counts`

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

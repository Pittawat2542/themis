---
title: Tune runtime controls
diataxis: how-to
audience: users optimizing execution behavior
goal: Show how to adjust concurrency, rate limits, and retry behavior without changing run identity.
---

# Tune runtime controls

Goal: adjust concurrency, provider limits, retry behavior, and duplicate-run policy safely.

When to use this:

Use this guide when the experiment definition is correct but execution behavior needs operational tuning.

## Procedure

Configure `RuntimeConfig` to change execution-time behavior:

- `max_concurrent_tasks`
- `stage_concurrency`
- `provider_concurrency`
- `provider_rate_limits`
- `generation_retry_attempts`, `generation_retry_delay`, `generation_retry_backoff`
- `judge_retry_attempts`, `judge_retry_delay`, `judge_retry_backoff`
- `store_retry_attempts`
- `store_retry_delay`
- `existing_run_policy`

Provider-backed models are treated as endpoints. Use `provider_concurrency` and `provider_rate_limits` to keep one process fair across multiple endpoint-backed models or benchmarks without changing the experiment identity.

Retry behavior:

- generation retries classify explicit retryable errors, timeouts, connection failures, `429` rate limits, and `5xx` provider failures
- judge retries use the same classification and preserve retry history, including `retry_after_s` hints when available
- retry metadata is persisted on generation artifacts and workflow failures so later inspection can distinguish a hard failure from a transient recovery

Estimate behavior:

- `themis estimate --config ...` now returns task counts and token-level estimates
- generation estimates report input and assumed output tokens
- judge estimates report estimated prompt and assumed output tokens
- Themis does not price those tokens; use the estimate JSON as input to an external cost model

## Variants

- conservative provider rollout: low provider concurrency and explicit rate limits
- throughput-oriented local runs: raise concurrency carefully while keeping storage stable

## Expected result

You should be able to alter execution behavior without changing `run_id`.

## Troubleshooting

- [Identity vs provenance](../explanation/identity-vs-provenance.md)
- [Config schema](../reference/config-schema.md)

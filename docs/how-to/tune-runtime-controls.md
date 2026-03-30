---
title: Tune runtime controls
diataxis: how-to
audience: users optimizing execution behavior
goal: Show how to adjust concurrency, rate limits, and retry behavior without changing run identity.
---

# Tune runtime controls

Goal: adjust concurrency, provider limits, and store retry behavior safely.

When to use this:

Use this guide when the experiment definition is correct but execution behavior needs operational tuning.

## Procedure

Configure `RuntimeConfig` to change execution-time behavior:

- `max_concurrent_tasks`
- `stage_concurrency`
- `provider_concurrency`
- `provider_rate_limits`
- `store_retry_attempts`
- `store_retry_delay`

## Variants

- conservative provider rollout: low provider concurrency and explicit rate limits
- throughput-oriented local runs: raise concurrency carefully while keeping storage stable

## Expected result

You should be able to alter execution behavior without changing `run_id`.

## Troubleshooting

- [Identity vs provenance](../explanation/identity-vs-provenance.md)
- [Config schema](../reference/config-schema.md)

---
title: Config schema reference
diataxis: reference
audience: config-driven Themis users
goal: Document config model fields, defaults, and identity/persistence implications.
---

# Config schema reference

## `GenerationConfig`

- `generator`: required; builtin component id or custom generator object/module path
- `candidate_policy`: defaults to `{}`; affects `run_id`
- `reducer`: optional; affects `run_id`

## `EvaluationConfig`

- `metrics`: list of pure or workflow-backed metrics; affects `run_id`
- `parsers`: list of parsers; affects `run_id`
- `judge_models`: workflow judge models; affects `run_id`
- `judge_config`: workflow evaluation config; affects `run_id`
- `workflow_overrides`: additional workflow config; affects `run_id`

## `StorageConfig`

- `store`: backend name such as `memory`, `sqlite`, `jsonl`, `mongodb`, or `postgres`
- `parameters`: backend-specific settings; stored as provenance, not identity

## `RuntimeConfig`

- `max_concurrent_tasks`: global execution cap; does not affect `run_id`
- `stage_concurrency`: per-stage caps; does not affect `run_id`
- `provider_concurrency`: per-provider concurrency caps; does not affect `run_id`
- `provider_rate_limits`: per-provider rate limits; does not affect `run_id`
- `store_retry_attempts` and `store_retry_delay`: persistence retry behavior; do not affect `run_id`
- `queue_root` and `batch_root`: submission paths; do not affect `run_id`

Use [Identity vs provenance](../explanation/identity-vs-provenance.md) when deciding whether a config change should create a new logical run.

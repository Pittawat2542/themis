---
title: Config schema reference
diataxis: reference
audience: config-driven Themis users
goal: Document config model fields, defaults, and identity/persistence implications.
---

# Config schema reference

Config file support:

- `Experiment.from_config(...)` loads `YAML` (`.yaml` / `.yml`) and `TOML` (`.toml`)
- config files carry strings and JSON-like values; live Python objects belong only in direct Python authoring

Component target syntax:

- builtin ids such as `builtin/exact_match`
- importable module paths such as `package.module:factory` or `package.module:Class`
- when a config target resolves to a class, Themis instantiates it without constructor arguments

## `GenerationConfig`

- `generator`: required; builtin component id or importable module path in config files, or a live generator object in Python authoring
- `candidate_policy`: defaults to `{}`; affects `run_id`
- `prompt_spec`: optional prompt instructions, prefixes, suffixes, and few-shot examples; affects `run_id`
- `reducer`: optional; affects `run_id`

## `EvaluationConfig`

- `metrics`: list of pure or workflow-backed metrics; affects `run_id`
- `parsers`: list of parsers; affects `run_id`
- `judge_models`: workflow judge models; affects `run_id`
- `prompt_spec`: optional prompt instructions, prefixes, suffixes, and few-shot examples for builtin judge workflows; affects `run_id`
- `judge_config`: workflow evaluation config; affects `run_id`
- `workflow_overrides`: additional workflow config; affects `run_id`

## `StorageConfig`

- `store`: backend name such as `memory`, `sqlite`, `jsonl`, `mongodb`, or `postgres`
- `parameters`: backend-specific settings; stored as provenance, not identity
- relative `parameters.path`, `parameters.root`, and `parameters.blob_root` values resolve relative to the config file directory

## `RuntimeConfig`

- `max_concurrent_tasks`: global execution cap; does not affect `run_id`
- `stage_concurrency`: per-stage caps; does not affect `run_id`
- `provider_concurrency`: per-provider concurrency caps; does not affect `run_id`
- `provider_rate_limits`: per-provider rate limits; does not affect `run_id`
- `generation_retry_attempts`, `generation_retry_delay`, and `generation_retry_backoff`: generation retry behavior; do not affect `run_id`
- `judge_retry_attempts`, `judge_retry_delay`, and `judge_retry_backoff`: judge retry behavior; do not affect `run_id`
- `store_retry_attempts` and `store_retry_delay`: persistence retry behavior; do not affect `run_id`
- `existing_run_policy`: duplicate-run handling (`auto`, `error`, `rerun`); does not affect `run_id`
- `queue_root` and `batch_root`: submission paths; do not affect `run_id`
- relative `queue_root` and `batch_root` values resolve relative to the config file directory

Overrides:

- `Experiment.from_config(path, overrides=[...])` accepts OmegaConf dotlist overrides before normalization and component loading
- use overrides for environment-specific paths or small execution changes without forking the whole config file

Use [Identity vs provenance](../explanation/identity-vs-provenance.md) when deciding whether a config change should create a new logical run.

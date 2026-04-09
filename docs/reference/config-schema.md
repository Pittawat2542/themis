---
title: Config schema reference
diataxis: reference
audience: config-driven Themis users
goal: Document config model fields, defaults, and identity/persistence implications.
---

# Config schema reference

## Config file support

| Field | Required | Purpose | Affects run_id | Notes |
| --- | --- | --- | --- | --- |
| Config file format | Yes | `Experiment.from_config(...)` loads `YAML` (`.yaml` / `.yml`) and `TOML` (`.toml`) | Yes, after normalization into the compiled snapshot | Choose the format that best fits your repo conventions |
| Config values | Yes | Carry strings and JSON-like values for components, prompts, storage, and runtime settings | Yes for identity-bearing fields; no for pure runtime tuning fields | Live Python objects belong only in direct Python authoring |

## Component target syntax

| Field | Required | Purpose | Affects run_id | Notes |
| --- | --- | --- | --- | --- |
| Builtin ids such as `builtin/exact_match` | No | Reference shipped catalog components from config files | Yes | Best when you want stable builtins without writing import paths |
| Importable factory path such as `package.module:factory` | No | Reference your own component factory from config | Yes | Best when constructor logic belongs in Python |
| Importable class path such as `package.module:Class` | No | Reference a component type directly from config | Yes | Themis instantiates the class without constructor arguments |

## `GenerationConfig`

| Field | Required | Purpose | Affects run_id | Notes |
| --- | --- | --- | --- | --- |
| `generator` | Yes | Chooses the candidate producer | Yes | In config, use a builtin id or import path; in Python, you may pass a live object |
| `candidate_policy` | No | Controls generation fan-out such as `num_samples` | Yes | Defaults to `{}` and is part of logical experiment identity |
| `prompt_spec` | No | Carries prompt instructions, prefixes, suffixes, and generic prompt blocks | Yes | Prompt changes invalidate generation-stage cache reuse as expected |
| `PromptSpec.blocks` | No | Stores arbitrary structured prompt material | Yes | Themis does not assign example-specific semantics to block contents |
| `reducer` | No | Chooses how multiple candidates collapse after fan-out | Yes | Pair with selectors or reducers when `num_samples` is greater than one |

## `EvaluationConfig`

| Field | Required | Purpose | Affects run_id | Notes |
| --- | --- | --- | --- | --- |
| `metrics` | Yes | Lists the pure or workflow-backed metrics to run | Yes | Metric choice defines evaluation semantics |
| `parsers` | No | Normalizes reduced output into metric-ready subjects | Yes | Choose parsers that match the expected output shape |
| `judge_models` | No | Provides judge models for workflow-backed metrics | Yes | Omit when using only deterministic pure metrics |
| `prompt_spec` | No | Adds prompt instructions or blocks for builtin judge workflows | Yes | Judge prompt changes are identity-bearing |
| `judge_config` | No | Carries generic runtime configuration for workflow implementations | Yes | Exposed to workflows as `EvalScoreContext.judge_config`; use for custom workflow config that should affect runtime behavior and identity |
| `workflow_overrides` | No | Carries builtin-oriented prompt and rubric overrides | Yes | Exposed as `EvalScoreContext.eval_workflow_config`; useful for rubric text and benchmark-specific builtin judge settings |

## `StorageConfig`

| Field | Required | Purpose | Affects run_id | Notes |
| --- | --- | --- | --- | --- |
| `store` | Yes | Selects the backend such as `memory`, `sqlite`, `jsonl`, `mongodb`, or `postgres` | Yes | Choose based on persistence and operational needs |
| `parameters` | No | Supplies backend-specific settings | No | Stored as provenance rather than logical run identity |
| Relative `parameters.path`, `parameters.root`, and `parameters.blob_root` | No | Resolves storage paths from the config file directory | No | Keeps checked-in configs portable across environments |

## `RuntimeConfig`

| Field | Required | Purpose | Affects run_id | Notes |
| --- | --- | --- | --- | --- |
| `max_concurrent_tasks` | No | Sets the global execution cap | No | Use for coarse operational throttling |
| `stage_concurrency` | No | Sets per-stage concurrency caps | No | Useful when generation and judging need different limits |
| `provider_concurrency` | No | Limits concurrency per provider endpoint | No | Helps share one process fairly across models or services |
| `provider_rate_limits` | No | Sets explicit per-provider request or token limits | No | Use when the endpoint enforces quotas or rate contracts |
| `generation_retry_attempts`, `generation_retry_delay`, `generation_retry_backoff` | No | Controls generation retry behavior | No | Retries transient provider failures without changing identity |
| `judge_retry_attempts`, `judge_retry_delay`, `judge_retry_backoff` | No | Controls judge retry behavior | No | Applies only to workflow-backed metrics |
| `store_retry_attempts`, `store_retry_delay` | No | Controls persistence retry behavior | No | Use when the store can fail transiently |
| `existing_run_policy` | No | Chooses duplicate-run handling with `auto`, `error`, or `rerun` | No | Affects execution behavior, not logical identity |
| `queue_root` and `batch_root` | No | Select manifest output roots for deferred execution | No | Used by `submit`, `worker`, and `batch` flows |
| Relative `queue_root` and `batch_root` | No | Resolves runtime paths from the config file directory | No | Keeps checked-in configs portable across machines |

## Overrides

| Field | Required | Purpose | Affects run_id | Notes |
| --- | --- | --- | --- | --- |
| `Experiment.from_config(path, overrides=[...])` | No | Applies OmegaConf dotlist overrides before normalization and component loading | Depends on the fields you override | Useful for environment-specific paths or small execution changes |
| Override usage | No | Lets one checked-in config serve multiple environments or execution shapes | Depends on the fields you override | Prefer this over forking a config file for small changes |

Use [Identity vs provenance](../explanation/identity-vs-provenance.md) when deciding whether a config change should create a new logical run.

# Configuration Guide

Themis uses Hydra/OmegaConf so you can describe experiments declaratively. This
guide explains the schema and shows common overrides.

## Schema overview

```yaml
name: math500_zero_shot          # experiment type (currently math500 helper)
dataset:
  source: huggingface            # huggingface | local | inline
  data_dir: null                 # required when source=local
  limit: null                    # sample cap
  subjects: []                   # subject filter for math500
  inline_samples: []             # list of dicts when source=inline
generation:
  model_identifier: fake-math-llm
  provider:
    name: fake                   # registry name from themis.providers
    options: {}                  # kwargs passed to provider factory
  sampling:
    temperature: 0.0
    top_p: 0.95
    max_tokens: 512
  runner:
    max_parallel: 1
    max_retries: 3
    retry_initial_delay: 0.5
    retry_backoff_multiplier: 2.0
    retry_max_delay: 2.0
storage:
  path: null                     # e.g., .cache/runs
max_samples: null                # cap tasks after plan expansion
run_id: null                     # resume/cache key
resume: true
```

Every field is optional; unspecified values fall back to sensible defaults.

## Inline datasets

Use inline data for smoke tests or synthetic tasks:

```yaml
dataset:
  source: inline
  inline_samples:
    - unique_id: inline-1
      problem: "What is 4 + 4?"
      answer: "8"
```

Fields become part of the prompt context and metadata automatically.

## Local or HF datasets

```yaml
dataset:
  source: local
  data_dir: /path/to/MATH-500
  limit: 100
  subjects: ["algebra", "number theory"]
```

Switch to `source: huggingface` and drop `data_dir` for remote loads.

## Provider options

Registered providers accept arbitrary keyword arguments:

```yaml
generation:
  provider:
    name: fake
    options:
      seed: 1337
```

Swap `name` for custom providers you registered via
`themis.providers.register_provider`.

## Sampling and retries

Tune sampling and the retry/backoff policy:

```yaml
generation:
  sampling:
    temperature: 0.2
    top_p: 0.9
    max_tokens: 256
  runner:
    max_parallel: 4
    max_retries: 5
    retry_initial_delay: 0.2
    retry_backoff_multiplier: 1.5
```

The runner records attempt metadata and the CLI logs each retry.

## Storage & caching

```yaml
storage:
  path: .cache/themis      # Specific storage path (takes precedence)
  default_path: .cache/runs # Default storage path when path is not specified
run_id: math500-smoke
resume: true
```

Generations and evaluation scores are cached per `run_id`. Reruns reuse any task
already in storage, so you can interrupt long runs safely.

If `path` is specified, it will be used for storage. If `path` is null/empty but 
`default_path` is specified, then `default_path` will be used as the storage location.
This allows you to set a default storage location for all experiments while still
being able to override it for specific runs.

## Hydra overrides

All CLI commands accept `--overrides` arguments interpreted by Hydra:

```bash
# First generate a config file
uv run python -m themis.cli init --output my_config.yaml

# Then run with overrides
uv run python -m themis.cli run-config \
  --config my_config.yaml \
  --overrides generation.provider.options.seed=99 max_samples=5
```

Use dotted paths to tweak nested fields; multiple overrides can be specified as
separate arguments.

## Programmatic usage

```python
from pathlib import Path
from themis.config import load_experiment_config, run_experiment_from_config

# Load your config file (create with: uv run python -m themis.cli init)
cfg = load_experiment_config(Path("my_config.yaml"), overrides=["max_samples=3"])
report = run_experiment_from_config(cfg)
print(report.metadata["successful_generations"])
```

Pass `dataset=` and `on_result=` kwargs to `run_experiment_from_config` when you
need tighter control (custom progress bar, streaming logs, etc.).

---

See `docs/EXAMPLES.md` for concrete end-to-end scenarios and
`docs/ADDING_COMPONENTS.md` to extend the schema with new experiment types.

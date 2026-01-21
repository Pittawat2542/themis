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
  config:                        # StorageConfig options
    save_raw_responses: false    # Save full API responses
    save_dataset: true           # Save dataset copy
    compression: gzip            # Compression: gzip | none
    deduplicate_templates: true  # Deduplicate prompt templates
max_samples: null                # cap tasks after plan expansion
run_id: null                     # resume/cache key
resume: true
integrations:
  wandb:
    enable: false
    project: null
    entity: null
    tags: []
  huggingface_hub:
    enable: false
    repository: null
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

## Integrations

Configure external integrations like Weights & Biases (W&B) and Hugging Face Hub.

```yaml
integrations:
  wandb:
    enable: true             # Enable W&B logging
    project: my-themis-project # W&B project name
    entity: my-wandb-entity  # W&B entity (username or team)
    tags: ["llm-eval", "math"]
  huggingface_hub:
    enable: true             # Enable Hugging Face Hub uploads
    repository: my-username/my-themis-results # Hugging Face Hub repository ID
```

- `wandb.enable`: Set to `true` to enable Weights & Biases logging for experiment metrics and results.
- `wandb.project`: The name of the W&B project to log to.
- `wandb.entity`: Your W&B username or team name.
- `wandb.tags`: A list of tags to associate with the W&B run.
- `huggingface_hub.enable`: Set to `true` to enable uploading experiment results to the Hugging Face Hub.
- `huggingface_hub.repository`: The ID of the Hugging Face Hub repository (e.g., `your-username/your-repo-name`) where results will be uploaded as a dataset.

## Storage Configuration

Themis provides configurable storage options to optimize disk space and performance.

### StorageConfig Options

```python
from themis.experiment.storage import StorageConfig, ExperimentStorage, RetentionPolicy

config = StorageConfig(
    save_raw_responses=False,    # Save full API responses (default: False)
    save_dataset=True,           # Save dataset copy (default: True)
    compression="gzip",          # Compression: "gzip" | "none" (default: "gzip")
    deduplicate_templates=True,  # Store templates once (default: True)
    enable_checksums=True,       # Data integrity validation (default: True)
    use_sqlite_metadata=True,    # Use SQLite for metadata (default: True)
    checkpoint_interval=100,     # Save checkpoint every N records (default: 100)
    retention_policy=RetentionPolicy(  # Automatic cleanup (default: None)
        max_runs_per_experiment=10,
        max_age_days=30,
        keep_latest_n=5,
    ),
)

storage = ExperimentStorage("outputs/experiments", config=config)
```

### Storage Optimizations

**1. Compression (50-60% savings)**
- `compression="gzip"`: Enable gzip compression for all JSONL files
- `compression="none"`: Disable compression (easier to inspect, but larger files)
- Files are automatically decompressed when loaded
- Default: `"gzip"` (recommended)

**2. Raw API Responses (~5MB savings per 1,500 samples)**
- `save_raw_responses=False`: Don't save full API responses (recommended)
- `save_raw_responses=True`: Keep raw responses for debugging
- Raw responses include full API metadata, token IDs, etc.
- Usually not needed since extracted output is saved
- Default: `False` (recommended)

**3. Template Deduplication (~627KB savings per 1,500 samples)**
- `deduplicate_templates=True`: Store each unique template once
- `deduplicate_templates=False`: Store template in every task
- Significant savings for large experiments with repeated templates
- Default: `True` (recommended)

**4. Dataset Saving**
- `save_dataset=True`: Save a copy of the dataset
- `save_dataset=False`: Don't save dataset (if loading from file)
- Set to `False` when loading from existing files to avoid duplication
- Default: `True`

**5. Format Versioning**
- All files include version headers for safe format evolution
- Example header: `{"_type": "header", "_format_version": "1.0.0", "_file_type": "records"}`
- Automatic versioning prevents incompatibility issues

### Storage Profiles

**Production (Maximum Optimization):**
```python
config = StorageConfig(
    save_raw_responses=False,
    compression="gzip",
    deduplicate_templates=True,
    save_dataset=False,  # If loading from file
)
# Typical savings: 60-75% reduction
# 18.5MB → 3-5MB for 1,500 samples
```

**Development (Balanced):**
```python
config = StorageConfig(
    save_raw_responses=False,
    compression="gzip",
    deduplicate_templates=True,
)
# Good balance of space savings and usability
```

**Debug (Keep Everything):**
```python
config = StorageConfig(
    save_raw_responses=True,
    compression="none",
    deduplicate_templates=False,
)
# Easier to inspect files manually
# No space savings
```

### Storage Structure

With default configuration, storage structure is:

```
outputs/run-id/
├── templates.jsonl.gz     # Unique templates (deduplication)
├── tasks.jsonl.gz         # Task definitions (reference templates)
├── records.jsonl.gz       # Generation outputs (no raw responses)
├── evaluation.jsonl.gz    # Evaluation results
├── summary.json           # Quick summary (1KB, uncompressed)
└── report.json            # Full report (optional)
```

### Quick Summary Export

Export lightweight summaries for fast result viewing:

```python
from themis.experiment.export import export_summary_json

export_summary_json(
    report,
    "outputs/run-123/summary.json",
    run_id="run-123"
)
```

View summaries via CLI:

```bash
# View summary for a run (~1KB file vs ~1.6MB report)
uv run python -m themis.cli results-summary --run-id run-123

# List all runs with metrics
uv run python -m themis.cli results-list

# List 10 most recent runs
uv run python -m themis.cli results-list --limit 10
```

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

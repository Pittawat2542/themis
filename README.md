# Themis

Lightweight experimentation harness for text-generation systems. Themis stitches
prompt templates, LLM providers, generation strategies, evaluation metrics, and
storage/resume behaviors into reproducible pipelines.

## At a glance

- Generation domain: prompt templates, sampling plans, provider registry, retrying
  runners, and routing helpers.
- Evaluation domain: extraction utilities (JSON, math-verify), metrics, and
  strategies for multi-attempt scoring.
- Experiment domain: orchestration, storage, and CLI wiring with Hydra configs,
  progress reporting, and structured logging.

## Table of contents

1. [Architecture](#architecture)
2. [Quick start](#quick-start)
3. [CLI workflows](#cli-workflows)
4. [Configuration](#configuration)
5. [Logging & progress](#logging--progress)
6. [Examples](#examples)
7. [Extending Themis](#extending-themis)
8. [Development](#development)

## Architecture

Experiments flow from CLI/config ➜ builders ➜ generation plan ➜ runner/providers ➜
evaluation ➜ report. See `docs/DIAGRAM.md` for a mermaid overview and
`docs/ADDING_COMPONENTS.md` for extension points.

## Quick start

```bash
uv run python main.py               # print CLI help
uv run python -m themis.cli demo    # smoke-test two inline math prompts
uv run python -m themis.cli math500 --help
uv run pytest                       # execute the unit tests
```

The demo uses the bundled fake math model so you can verify the full pipeline
without credentials.

## CLI workflows

### MATH-500 helper

Run the zero-shot benchmark with resumability and caching:

```bash
uv run python -m themis.cli math500 \
  --source local \
  --data-dir /path/to/MATH-500 \
  --limit 50 \
  --storage .cache/themis \
  --run-id math500-local \
  --temperature 0.1 \
  --log-level info
```

Use `--source huggingface` (default) to fetch directly from the HF hub. Cached
datasets and generation results are keyed by `--run-id`, so subsequent executions
reuse prior work when `--resume` is true.

### Config-driven runs

Every CLI feature is configurable via Hydra/OmegaConf. Point the runner at a YAML
file (the repo ships with `configs/math500_demo.yaml`):

```bash
uv run python -m themis.cli run-config \
  --config configs/math500_demo.yaml \
  --overrides generation.sampling.temperature=0.2 max_samples=1 \
  --log-level trace
```

Overrides let you tweak scenarios without editing files. Configs support inline
datasets, local/HF sources, provider options, retry/backoff knobs, and storage
paths. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full schema.

## Configuration

- `themis.config.schema` defines the structured config (dataset/generation/storage).
- `themis.config.loader` merges YAML with defaults and applies Hydra-style overrides.
- `themis.config.runtime` exposes `run_experiment_from_config` plus `load_dataset_from_config`
  so notebooks and services can reuse the exact same flow as the CLI.

Common recipes—inline datasets, retry overrides, and programmatic usage—are
covered in [docs/EXAMPLES.md](docs/EXAMPLES.md).

## Logging & progress

All CLI commands accept `--log-level` (`critical`, `error`, `warning`, `info`,
`debug`, `trace`). Under the hood `themis.utils.logging_utils.configure_logging`
sets up structured timestamps, while runners and evaluation pipelines emit
per-attempt traceability.

While an experiment runs, `tqdm` progress bars show how many samples have been
processed (respecting `--max-samples`). You can attach your own callback via the
`on_result` hook provided to `ExperimentOrchestrator.run`.

## Examples

`docs/EXAMPLES.md` walks through:

1. Demo run
2. Cached math500 evaluation
3. Config-driven executions
4. Inline datasets
5. Retry/backoff overrides
6. Programmatic embeddings

Use it as a cookbook when onboarding teammates.

## Extending Themis

- `docs/ADDING_COMPONENTS.md` – add providers, datasets, prompts, strategies, metrics.
- `experiments/example`, `experiments/advanced_example`, `experiments/agentic_example`
  – runnable references (each exposes `python -m experiments.<name>.cli`).
- `docs/DIAGRAM.md` – architecture diagram for presentations/reviews.

The reusable `themis.experiment.builder` module assembles plans, runners,
pipelines, and storage from declarative definitions so new experiments mostly
specify templates and metrics.

## Development

```bash
uv run python -m themis.cli demo              # smoke test
uv run python -m themis.cli math500 --limit 5 # targeted run
uv run pytest                                 # full test suite
```

Optional extras:

- `uv pip install '.[dev]'` – testing.
- `uv pip install '.[math]'` – math-verify integration for numeric datasets.

Linting/formatting is intentionally lightweight; rely on `pytest` plus type
checkers in your editor. Use storage paths under `.cache/` to keep generated data
local and gitignored.

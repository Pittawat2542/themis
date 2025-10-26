# Practical Examples

This page highlights a few end-to-end flows so you can turn knobs without
digging through the source.

## 1. Built-in Demo (quick sanity check)

Runs two inline math problems with the fake model, shows progress, and prints a
summary. Perfect for verifying your environment.

```bash
uv run python -m themis.cli demo --max-samples 2 --log-level info
```

Output includes a progress bar plus logging (if `--log-level` is `info` or
higher).

## 2. Math500 evaluation with caching

Run the bundled MATH-500 helper against a Hugging Face download or local mirror.
This command caches results so reruns skip completed generations.

```bash
uv run python -m themis.cli math500 \
  --source huggingface \
  --limit 50 \
  --storage .cache/themis \
  --run-id math500-hf-50 \
  --temperature 0.1 \
  --log-level debug
```

Use `--source local --data-dir /path/to/mirror` for offline runs.

## 3. Config-driven experiment
## Declarative configuration

Every setting (dataset source, provider options, retry/backoff) can be defined in
a YAML file. Generate a config file with the `init` command:

```bash
# Generate a config file
uv run python -m themis.cli init --output my_config.yaml

# Run with optional overrides
uv run python -m themis.cli run-config \
  --config my_config.yaml \
  --overrides generation.sampling.temperature=0.2 \
  --log-level trace
```

Hydra/OmegaConf overrides let you tweak scenarios without editing the file.

## 4. Custom inline dataset via config

Create `my_inline.yaml`:

```
name: math500_zero_shot
dataset:
  source: inline
  inline_samples:
    - unique_id: inline-1
      problem: "What is 4 + 4?"
      answer: "8"
generation:
  provider:
    name: fake
    options:
      seed: 42
storage:
  path: null              # No specific path, will use default_path
  default_path: .cache/themis  # Default storage location for all experiments
max_samples: 1
```

Run:

```bash
uv run python -m themis.cli run-config --config my_inline.yaml --log-level info
```

This configuration will store generation and evaluation results in the `.cache/themis` directory
by default. You can still override this by specifying a specific `path` value or using the
`--overrides storage.path=/custom/path` CLI option.

## 5. Adjust retry/backoff behavior

The config schema exposes generation runner knobs. Example override:

```bash
# First create a config file if you haven't
uv run python -m themis.cli init --output my_config.yaml

# Then run with retry overrides
uv run python -m themis.cli run-config \
  --config my_config.yaml \
  --overrides generation.runner.max_retries=5 generation.runner.retry_initial_delay=0.2
```

The CLI logs every retry attempt so you can inspect flaky providers.

## 6. Programmatic usage

Use the config runtime helpers to embed Themis inside notebooks or scripts:

```python
from pathlib import Path
from themis.config import load_experiment_config, run_experiment_from_config

# Load your config (create with: uv run python -m themis.cli init)
cfg = load_experiment_config(Path("my_config.yaml"))
report = run_experiment_from_config(cfg)
print(report.metadata["successful_generations"])
```

Pass `on_result=` callbacks to stream progress or integrate custom progress bars.

---

See `README.md` for feature overviews and `docs/ADDING_COMPONENTS.md` for how to
extend Themis with new prompts, models, or metrics.

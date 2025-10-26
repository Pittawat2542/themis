# Configuration-Driven Experiments

This example teaches you how to run experiments purely through configuration files, without writing any code. This is ideal for:

- Quickly iterating on experiment parameters
- Sharing reproducible experiment setups with teammates
- Running systematic parameter sweeps
- Documenting experiment configurations in version control

## What You'll Learn

1. How to structure configuration files
2. How to override config values from the CLI
3. How to run multiple models and sampling strategies
4. How to use different datasets
5. How to organize configs for different scenarios

## Prerequisites

```bash
# Install Themis
uv pip install -e .
```

## Basic Configuration

The simplest config file (`config.sample.json`):

```json
{
  "run_id": "math500-demo",
  "storage_dir": ".cache/math500-demo",
  "resume": true,
  "models": [
    {
      "name": "fake-math-llm",
      "provider": "fake",
      "description": "Heuristic math model"
    }
  ],
  "samplings": [
    {
      "name": "zero-shot",
      "temperature": 0.0,
      "top_p": 0.95,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "demo",
      "kind": "demo",
      "limit": 2
    }
  ]
}
```

Run it:

```bash
uv run python -m experiments.02_config_file.cli run --config-path experiments/02_config_file/config.sample.json
```

## Configuration Fields Explained

### Top-Level Fields

- **`run_id`**: Unique identifier for this experiment run. Used for caching and resumability.
- **`storage_dir`**: Directory where results are cached. Enables resume functionality.
- **`resume`**: Boolean. If true, skips already-completed samples when rerunning.

### Models Array

Each model needs:
- **`name`**: Your identifier for this model
- **`provider`**: The provider type (`fake`, `openai-compatible`, etc.)
- **`description`**: Human-readable description
- **`provider_options`**: (Optional) Provider-specific configuration

Example with multiple models:

```json
{
  "models": [
    {
      "name": "model-a",
      "provider": "fake",
      "description": "First model to compare"
    },
    {
      "name": "model-b",
      "provider": "fake",
      "description": "Second model to compare",
      "provider_options": {
        "seed": 42
      }
    }
  ]
}
```

### Samplings Array

Sampling parameters control generation behavior:

```json
{
  "samplings": [
    {
      "name": "greedy",
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 512
    },
    {
      "name": "creative",
      "temperature": 0.8,
      "top_p": 0.95,
      "max_tokens": 512
    }
  ]
}
```

**Common parameters:**
- **`temperature`**: Controls randomness (0.0 = deterministic, higher = more random)
- **`top_p`**: Nucleus sampling threshold (0.0-1.0)
- **`max_tokens`**: Maximum tokens to generate
- **`name`**: Identifier for this sampling configuration

### Datasets Array

Specify which datasets to evaluate:

```json
{
  "datasets": [
    {
      "name": "demo",
      "kind": "demo",
      "limit": 10
    },
    {
      "name": "math500-sample",
      "kind": "math500_hf",
      "limit": 50
    }
  ]
}
```

**Dataset options:**
- **`name`**: Your identifier for this dataset
- **`kind`**: Dataset type (`demo`, `math500_hf`, `math500_local`)
- **`limit`**: (Optional) Maximum number of samples to use
- **`data_dir`**: (Optional) For local datasets, path to data directory

## CLI Overrides

You can override any config value from the command line:

```bash
# Override run_id
uv run python -m experiments.02_config_file.cli run \
  --config-path config.sample.json \
  --run-id my-custom-run

# Override storage directory
uv run python -m experiments.02_config_file.cli run \
  --config-path config.sample.json \
  --storage-dir .cache/custom-storage

# Override resume behavior
uv run python -m experiments.02_config_file.cli run \
  --config-path config.sample.json \
  --resume false

# Combine multiple overrides
uv run python -m experiments.02_config_file.cli run \
  --config-path config.sample.json \
  --run-id experiment-1 \
  --storage-dir .cache/exp1 \
  --resume true
```

## Example Configurations

### Configuration 1: Single Model, Multiple Sampling Strategies

Create `compare_sampling.json`:

```json
{
  "run_id": "sampling-comparison",
  "storage_dir": ".cache/sampling-comparison",
  "resume": true,
  "models": [
    {
      "name": "test-model",
      "provider": "fake"
    }
  ],
  "samplings": [
    {
      "name": "greedy",
      "temperature": 0.0,
      "max_tokens": 512
    },
    {
      "name": "low-temp",
      "temperature": 0.3,
      "max_tokens": 512
    },
    {
      "name": "medium-temp",
      "temperature": 0.7,
      "max_tokens": 512
    },
    {
      "name": "high-temp",
      "temperature": 1.0,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "demo",
      "kind": "demo",
      "limit": 10
    }
  ]
}
```

This runs the same model with 4 different temperature settings, allowing you to compare their impact on accuracy.

### Configuration 2: Multiple Models, Single Sampling

Create `compare_models.json`:

```json
{
  "run_id": "model-comparison",
  "storage_dir": ".cache/model-comparison",
  "resume": true,
  "models": [
    {
      "name": "model-variant-a",
      "provider": "fake",
      "description": "Baseline model"
    },
    {
      "name": "model-variant-b",
      "provider": "fake",
      "description": "Experimental model",
      "provider_options": {
        "seed": 123
      }
    },
    {
      "name": "model-variant-c",
      "provider": "fake",
      "description": "Another variant",
      "provider_options": {
        "seed": 456
      }
    }
  ],
  "samplings": [
    {
      "name": "standard",
      "temperature": 0.7,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "demo",
      "kind": "demo",
      "limit": 20
    }
  ]
}
```

This compares 3 different models using the same sampling parameters.

### Configuration 3: Full Grid Search

Create `grid_search.json`:

```json
{
  "run_id": "grid-search",
  "storage_dir": ".cache/grid-search",
  "resume": true,
  "models": [
    {
      "name": "model-a",
      "provider": "fake"
    },
    {
      "name": "model-b",
      "provider": "fake",
      "provider_options": {"seed": 42}
    }
  ],
  "samplings": [
    {
      "name": "greedy",
      "temperature": 0.0,
      "max_tokens": 512
    },
    {
      "name": "balanced",
      "temperature": 0.7,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "demo",
      "kind": "demo",
      "limit": 10
    }
  ]
}
```

This creates a 2×2 grid: 2 models × 2 sampling strategies = 4 total configurations evaluated.

## Running Experiments

```bash
# Run sampling comparison
uv run python -m experiments.02_config_file.cli run \
  --config-path compare_sampling.json

# Run model comparison
uv run python -m experiments.02_config_file.cli run \
  --config-path compare_models.json

# Run grid search
uv run python -m experiments.02_config_file.cli run \
  --config-path grid_search.json

# Export results
uv run python -m experiments.02_config_file.cli run \
  --config-path grid_search.json \
  --csv-output results.csv \
  --html-output results.html
```

## Resumability and Caching

One of the most powerful features is automatic resumability:

```bash
# First run: processes all samples
uv run python -m experiments.02_config_file.cli run \
  --config-path grid_search.json

# Second run: skips already-completed samples (because resume=true)
uv run python -m experiments.02_config_file.cli run \
  --config-path grid_search.json

# Force re-run everything (disable resume)
uv run python -m experiments.02_config_file.cli run \
  --config-path grid_search.json \
  --resume false
```

**How it works:**
- Results are cached in `storage_dir` with keys based on model, sampling, and sample ID
- When `resume=true`, completed samples are skipped
- This allows you to interrupt experiments and continue later
- Add new models/samplings to an existing config and only new combinations run

## Organizing Configurations

For larger projects, organize configs by purpose:

```
configs/
├── dev/
│   ├── quick_test.json          # 2 samples for fast iteration
│   └── model_debug.json          # Single model, verbose settings
├── benchmarks/
│   ├── math500_full.json         # Full MATH-500 evaluation
│   └── math500_sample.json       # 50-sample quick benchmark
└── experiments/
    ├── temperature_sweep.json    # Systematic temperature comparison
    └── model_comparison.json     # Compare multiple model variants
```

## Dry Run Mode

Preview what will run without actually executing:

```bash
uv run python -m experiments.02_config_file.cli run \
  --config-path grid_search.json \
  --dry-run
```

Output:
```
Configured models: model-a, model-b
Datasets: demo
Samples per dataset: limit=10
Storage: .cache/grid-search
```

## Best Practices

1. **Use descriptive run_ids**: Include date, purpose, or version
   - Good: `2024-01-15-temperature-sweep`
   - Bad: `test1`

2. **Enable resume by default**: Prevents wasted computation
   - Set `"resume": true` in most configs

3. **Use separate storage_dirs**: Avoid conflicts between experiments
   - Pattern: `.cache/{run_id}`

4. **Start small**: Use `limit` to test with a few samples first
   - Example: `"limit": 5` for quick testing, then remove for full run

5. **Version control your configs**: Track experiment parameters
   - Commit config files to git
   - Use meaningful filenames

6. **Document provider_options**: Add comments in separate docs
   - Config format doesn't support comments (JSON limitation)
   - Keep a `CONFIG_NOTES.md` for complex settings

## Troubleshooting

**Q: My config file won't load**
- Ensure valid JSON (no trailing commas, proper quotes)
- Use a JSON validator: `python -m json.tool your_config.json`

**Q: Changes aren't taking effect**
- Check if `resume=true` is skipping samples
- Try `--resume false` or use a new `storage_dir`

**Q: How do I know what fields are valid?**
- See `config.py` for the Pydantic models
- Check the sample configs in each example

**Q: Can I use environment variables in configs?**
- Not directly in JSON, but you can use CLI overrides
- Or load configs programmatically with variable substitution

## Next Steps

- **03_openai_compatible**: Configure real LLM endpoints
- **04_projects**: Organize multiple config-driven experiments
- **05_advanced**: Override behavior with custom code

## File Structure

```
02_config_file/
├── README.md              # This file
├── cli.py                 # CLI entry point
├── config.py              # Configuration models
├── config.sample.json     # Basic sample
├── compare_sampling.json  # (Create this) Sampling comparison
├── compare_models.json    # (Create this) Model comparison
├── grid_search.json       # (Create this) Full grid
├── datasets.py            # Dataset loaders
└── experiment.py          # Experiment implementation
```

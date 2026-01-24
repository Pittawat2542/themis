# Getting Started Example

This example corresponds to `examples/getting_started` and demonstrates the core CLI commands and configuration.

## What it does
Runs a basic evaluation on a custom "demo" dataset using a fake model. This is the "Hello World" of Themis.

## How to Run

### Standard Run
Uses the default configuration bundled in the example.

```bash
uv run python -m examples.getting_started.cli run \
  --config-path examples/getting_started/config.sample.json
```

### With Custom Config
You can copy the sample config and modify it:

```bash
# 1. Copy config
cp examples/getting_started/config.sample.json my_config.json

# 2. Edit my_config.json (e.g. change limit to 20)

# 3. Run with custom config
uv run python -m examples.getting_started.cli run --config-path my_config.json
```

## Key Concepts
- **Config File**: Defines the experiment parameters (model, metrics, dataset).
- **CLI**: The interface to run and control experiments.

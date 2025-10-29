# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Themis is a lightweight experimentation harness for text-generation systems. It orchestrates prompt templates, LLM providers, generation strategies, evaluation metrics, and storage into reproducible, resumable pipelines for systematic LLM experimentation.

## Common Development Commands

### Installation & Setup
```bash
# Install with uv (recommended)
uv sync

# Full installation with dev tools
uv sync --extra dev

# With math evaluation support
uv sync --extra math
```

### Running Tests
```bash
# Full test suite
uv run pytest

# Specific test file
uv run pytest tests/experiment/test_builder.py

# With coverage
uv run pytest --cov=themis --cov-report=html

# Verbose output
uv run pytest -v
```

### Core CLI Commands
```bash
# Smoke test core functionality
uv run python -m themis.cli demo

# System information
uv run python -m themis.cli info

# List available providers
uv run python -m themis.cli list-providers --verbose

# List available benchmarks
uv run python -m themis.cli list-benchmarks --verbose

# Run built-in math500 benchmark
uv run python -m themis.cli math500 --limit 50

# Generate config file
uv run python -m themis.cli init --template math500 --output my_config.yaml

# Validate configuration
uv run python -m themis.cli validate-config --config my_config.yaml

# Run from config file
uv run python -m themis.cli run-config --config my_config.yaml

# Create new project
uv run python -m themis.cli new-project --project-name my-project
```

### Example Experiments
```bash
# Preview what will run
uv run python -m examples.getting_started.cli run --dry-run

# Run basic example
uv run python -m examples.getting_started.cli run

# Run config-based example with grid search
uv run python -m examples.config_file.cli run --config-path grid_search.json

# Prompt engineering experiment
uv run python -m examples.prompt_engineering.cli run --analyze

# Advanced examples
uv run python -m examples.advanced.cli run --enable-subject-breakdown
```

## High-Level Architecture

### Three-Layer Architecture

```
Configuration Layer (JSON/YAML/CLI)
    ↓
Experiment Layer (Orchestration)
    ↓
┌──────────────────────┐  ┌────────────────────────┐
│  Generation Domain   │  │  Evaluation Domain     │
│  • Prompts           │  │  • Extractors          │
│  • Providers         │  │  • Metrics             │
│  • Sampling plans    │  │  • Aggregation         │
│  • Retry logic       │  │                        │
└──────────────────────┘  └────────────────────────┘
```

### Module Organization

```
themis/
├── cli/                 # Command-line interface (Cyclopts)
├── config/              # Configuration schema & loader (Pydantic, Hydra)
├── core/                # Core entities (prompts, sampling specs, results)
├── datasets/            # Dataset loaders (inline, HuggingFace, local)
├── evaluation/          # Extractors, metrics, evaluation strategies
├── experiment/          # Orchestration, builder patterns, storage
├── generation/          # Generation strategies, runners, retry logic
├── interfaces/          # Abstract base classes
├── project/             # Multi-experiment project management
├── providers/           # LLM provider implementations
└── utils/               # Logging, progress tracking, helpers
```

### Key Components

1. **Generation Pipeline**: Prompt templates → Provider routing → Sampling strategies → Retry/backoff
2. **Evaluation Pipeline**: Response extraction → Metric computation → Multi-attempt scoring → Aggregation
3. **Experiment Orchestration**: Dataset loading → Generation plans → Runner execution → Storage → Reporting

See `docs/DIAGRAM.md` for detailed data flow diagrams.

## Configuration System

Themis uses Hydra/OmegaConf for configuration. Configuration files are YAML/JSON with these key sections:

```yaml
dataset:
  source: huggingface  # huggingface | local | inline
  limit: null          # sample cap
  subjects: []         # subject filter
generation:
  model_identifier: fake-math-llm
  provider:
    name: fake         # provider name from registry
    options: {}        # provider kwargs
  sampling:
    temperature: 0.0
    max_tokens: 512
storage:
  path: .cache/runs    # cache directory
run_id: null           # resume/cache key
resume: true
integrations:
  wandb:
    enable: false
  huggingface_hub:
    enable: false
```

See `docs/CONFIGURATION.md` for complete schema and `COOKBOOK.md` for common patterns.

## Examples & Tutorials

Start with the examples cookbook (`examples/README.md`) - a comprehensive tutorial series:

| Example | Focus | Time | What You'll Learn |
|---------|-------|------|-------------------|
| **getting_started** | Basics | 15 min | Prompts, models, sampling, evaluation |
| **config_file** | Configuration | 20 min | JSON configs, grid searches, resumability |
| **prompt_engineering** | Prompt Strategies | 25 min | Zero-shot, few-shot, chain-of-thought, systematic comparison |
| **projects** | Organization | 45 min | Multi-experiment projects, workflows |
| **advanced** | Customization | 60 min | Custom runners, pipelines, metrics |

Quick reference: `COOKBOOK.md` - contains common patterns and troubleshooting.

## Provider Ecosystem

- **LiteLLM provider** supports 100+ LLM providers (OpenAI, Anthropic, Azure, AWS Bedrock, Google AI, local LLMs, etc.)
- **Fake provider** for testing without API calls
- **Custom providers** can be registered via `themis.providers.register_provider`

See `docs/LITELLM.md` for provider configuration details and configuration files for local LLM setup.

## Built-in Benchmarks

- **math500**: MATH-500 dataset evaluation
- **Competition math**: aime24, aime25, amc23, olympiadbench, beyondaime
- **supergpqa**: Multiple-choice evaluation
- **mmlu-pro**: Professional-level multiple-choice evaluation
- **demo**: Synthetic inline dataset for testing

## Extension Points

Themis is designed for extensibility:

- **Custom providers**: Implement `Provider` interface
- **Custom datasets**: Implement `DatasetLoader`
- **Custom metrics**: Implement `Metric` interface
- **Custom runners**: Override generation loops
- **Custom pipelines**: Build evaluation pipelines with custom extractors

See `docs/ADDING_COMPONENTS.md` for detailed extension guides and `examples/advanced/` for working examples.

## Project Structure Patterns

### Standard Experiment Layout
```
examples/getting_started/
├── README.md              # Tutorial
├── config.sample.json     # Configuration template
├── experiment.py          # Programmatic example
└── cli.py                 # CLI wrapper
```

### Multi-Experiment Project
```
examples/projects/
├── README.md
├── project.toml           # Project metadata
├── configs/               # Shared configurations
└── experiments/
    ├── zero_shot.toml
    ├── few_shot.toml
    └── chain_of_thought.toml
```

## Caching & Resumability

- Experiments are automatically cached by `run_id`
- Use `--resume false` to force re-run
- Store cache in `.cache/` directory to keep data local
- Different experiments should use different `storage_dir` values

```bash
# Use custom storage for isolation
uv run python -m themis.cli math500 \
  --storage .cache/experiment-1 \
  --run-id run-2024-10-29
```

## Integrations

### Weights & Biases
Enable in configuration:
```yaml
integrations:
  wandb:
    enable: true
    project: themis-experiments
    entity: your-entity
    tags: ["experiment", "math500"]
```

### Hugging Face Hub
```yaml
integrations:
  huggingface_hub:
    enable: true
    repository: username/themis-results
```

## Coding Standards

- **Python 3.12+**, PEP 8 (4-space indent)
- **Type hints** throughout (mypy-compatible)
- **Dataclasses and Pydantic models** for configs/entities
- **File names**: `snake_case`
- **Classes**: `PascalCase`
- **CLI commands**: `dashed-names` (handled by Cyclopts)
- **Provider names, run IDs, dataset IDs**: lowercase (e.g., `fake-math-llm`, `run-2024Q1`)

## Testing Guidelines

- **Framework**: pytest (configured via `pyproject.toml`)
- **Test location**: `tests/` mirroring module paths
- **Naming**: Descriptive names (`test_pipeline_returns_metric_aggregates`)
- **Coverage**: Aim to cover new abstractions (strategies, builder hooks, storage)
- **Always run**: `uv run pytest` before submitting PRs

## Key Documentation

- **examples/README.md** - Comprehensive tutorial cookbook (START HERE!)
- **COOKBOOK.md** - Quick reference and common patterns
- **docs/CONFIGURATION.md** - Complete configuration schema
- **docs/DIAGRAM.md** - Architecture diagrams
- **docs/ADDING_COMPONENTS.md** - Extension guide
- **AGENTS.md** - Repository guidelines for AI agents

## Best Practices

1. **Always use resumability**: Set `"resume": true` in configs
2. **Start small**: Use `"limit": 5` for quick testing
3. **Use descriptive IDs**: `run_id: "2024-10-29-temperature-sweep"`
4. **Separate storage**: Different `storage_dir` per experiment
5. **Export results**: Always use `--csv-output` for analysis
6. **Version control configs**: Commit configurations to git
7. **Dry run first**: Use `--dry-run` to preview
8. **Document experiments**: Add descriptions to Project metadata

## Troubleshooting

### Common Issues

**"Module not found" errors**
- Always run from project root: `cd /path/to/themis`
- Use `uv run` prefix for commands

**Results not updating**
- Check if resume is enabled: use `--resume false` or new `--run-id`
- Clear cache: `rm -rf .cache/your-storage-dir`

**Slow performance**
- Reduce `n_parallel` in provider_options
- Use `"limit": 5` for quick testing
- Use fake provider for testing without API calls

**Connection refused (OpenAI-compatible)**
- Verify server is running: `curl http://localhost:1234/v1/models`
- Check port in config matches server
- Try `127.0.0.1` instead of `localhost`

## Learning Path

1. **Start with `getting_started`** (15 min)
   - Run the example, read README, modify config

2. **Try `config_file`** (20 min)
   - Use configuration files, run grid searches, learn resumability

3. **Master prompt engineering with `prompt_engineering`** (25 min)
   - Compare zero-shot vs few-shot vs chain-of-thought strategies
   - Analyze prompt effectiveness across models
   - Export results for analysis

4. **Organize with `projects`** (45 min)
   - Create project structure, define multiple experiments

5. **Customize with `advanced`** (60 min)
   - Override generation loops, create custom metrics, build agentic workflows

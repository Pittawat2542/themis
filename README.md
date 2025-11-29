# Themis

> **Lightweight evaluation platform for LLM experiments**

Themis orchestrates prompt templates, LLM providers, generation strategies, evaluation metrics, and storage into reproducible, resumable pipelines for systematic LLM experimentation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why Themis?

- **🎯 Config-driven**: Define experiments in JSON/YAML, run them with a single command
- **🔄 Resumable**: Automatic caching and resume—never lose your expensive LLM runs
- **📊 Systematic**: Grid search over models × prompts × sampling strategies
- **🔌 Provider-agnostic**: Works with 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Azure, AWS Bedrock, Google AI, local LLMs, and more)
- **📈 Built-in evaluation**: Exact match, math verification, custom metrics
- **🎓 Production-ready**: Type-safe configs, structured logging, progress tracking

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Examples & Tutorials](#examples--tutorials)
4. [Core Concepts](#core-concepts)
5. [CLI Reference](#cli-reference)
6. [Configuration](#configuration)
7. [Architecture](#architecture)
8. [Development](#development)
9. [Documentation](#documentation)

---

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/themis.git
cd themis

# Install with uv
uv sync

# Verify installation
uv run python -m themis.cli --version
```

### Using pip

```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With math evaluation support
pip install -e ".[math]"

# Full installation
pip install -e ".[dev,math]"
```

### Requirements

- Python 3.12+
- Dependencies: `pydantic`, `cyclopts`, `hydra-core`, `tqdm`, `httpx`

---

## Quick Start

### 1. Explore Available Components

See what's available in your installation:

```bash
# Show system info and quick start
uv run python -m themis.cli info

# List available LLM providers
uv run python -m themis.cli list-providers --verbose

# List available benchmarks
uv run python -m themis.cli list-benchmarks --verbose
```

### 2. Run the Built-in Demo

Test your installation with the fake model provider:

```bash
# Run a quick smoke test
uv run python -m themis.cli demo

# See what's happening
uv run python -m themis.cli demo --log-level info
```

This runs two inline math prompts through a fake LLM provider to verify the pipeline works end-to-end.

### 3. Try Your First Real Experiment

Start with the comprehensive examples cookbook:

```bash
# Your first experiment (15 minutes)
uv run python -m examples.getting_started.cli run

# Preview what will happen
uv run python -m examples.getting_started.cli run --dry-run

# Export results
uv run python -m examples.getting_started.cli run --csv-output results.csv
```

### 4. Connect to a Real LLM

Use any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, OpenAI):

```bash
# Generate a config file
uv run python -m themis.cli init --template math500 --output my_config.yaml

# Edit my_config.yaml: set base_url, api_key, model name
# Then validate it
uv run python -m themis.cli validate-config --config my_config.yaml

# Run the experiment
uv run python -m themis.cli run-config --config my_config.yaml
```

---

## Examples & Tutorials

**👉 Start here: [`examples/README.md`](examples/README.md)**

A comprehensive, hands-on cookbook with 5 progressive tutorials:

### Learning Path

| Example | Focus | Time | What You'll Learn |
|---------|-------|------|-------------------|
| **[getting_started](examples/getting_started/)** | Basics | 15 min | Prompts, models, sampling, evaluation |
| **[config_file](examples/config_file/)** | Configuration | 20 min | JSON configs, grid searches, resumability |
| **[prompt_engineering](examples/prompt_engineering/)** | Prompt Strategies | 25 min | Zero-shot, few-shot, chain-of-thought, systematic comparison |
| **[projects](examples/projects/)** | Organization | 45 min | Multi-experiment projects, research workflows |
| **[advanced](examples/advanced/)** | Customization | 60 min | Custom runners, pipelines, metrics, agentic workflows |

### Quick Reference

**🚀 [COOKBOOK.md](COOKBOOK.md)** - Cheat sheet with common patterns and troubleshooting

**Example commands:**
```bash
# Basic experiment
uv run python -m examples.getting_started.cli run

# Grid search (2 models × 3 temperatures)
uv run python -m examples.config_file.cli run --config-path grid_search.json

# Prompt engineering experiment
uv run python -m examples.prompt_engineering.cli run

# Multi-experiment project
uv run python -m examples.projects.cli list-experiments
uv run python -m examples.projects.cli run --experiment zero-shot

# Custom behavior
uv run python -m examples.advanced.cli run --enable-subject-breakdown
```

---

## Core Concepts

### Three-Layer Architecture

```
╭─────────────────────────────────────────────────────────╮
│  Configuration Layer (JSON/YAML/CLI)                    │
│  • Dataset specs • Models • Sampling • Storage          │
╰─────────────────────────────────────────────────────────╯
                         ↓
╭─────────────────────────────────────────────────────────╮
│  Experiment Layer (Orchestration)                       │
│  • Builder patterns • Runner coordination               │
│  • Progress tracking • Caching & resume                 │
╰─────────────────────────────────────────────────────────╯
                         ↓
╭───────────────────────╮    ╭────────────────────────────╮
│  Generation Domain    │    │  Evaluation Domain         │
│  • Prompts            │───▶│  • Extractors (JSON/math) │
│  • Providers          │    │  • Metrics (exact/custom)  │
│  • Sampling plans     │    │  • Aggregation             │
│  • Retry logic        │    │                            │
╰───────────────────────╯    ╰────────────────────────────╯
```

### Key Components

- **Generation**: Prompt templates → Provider routing → Sampling strategies → Retry/backoff
- **Evaluation**: Response extraction → Metric computation → Multi-attempt scoring → Aggregation
- **Experiment**: Dataset loading → Generation plans → Runner execution → Storage → Reporting

See [`docs/DIAGRAM.md`](docs/DIAGRAM.md) for detailed architecture diagrams.

---

## CLI Reference

Themis provides experiment commands and utility commands:

### Utility Commands

Quick commands for discovering and configuring Themis:

#### `info` - System Information

Show installed components and quick start guide:

```bash
uv run python -m themis.cli info
```

Displays:
- Version and Python info
- Installed providers
- Available benchmarks
- Example locations
- Documentation links

#### `list-providers` - Available LLM Providers

List all registered LLM providers:

```bash
uv run python -m themis.cli list-providers

# Show detailed information
uv run python -m themis.cli list-providers --verbose
```

Shows built-in providers (fake, litellm, vllm) and any custom registered providers.

**Note:** The `litellm` provider supports 100+ LLM providers including OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Google AI, Cohere, and many more. See [docs/LITELLM_PROVIDER.md](docs/LITELLM_PROVIDER.md) for details.

#### `list-benchmarks` - Available Datasets

List available datasets and benchmarks:

```bash
uv run python -m themis.cli list-benchmarks

# Show details with subjects and commands
uv run python -m themis.cli list-benchmarks --verbose
```

Shows math500, competition math benchmarks (aime24/aime25/amc23/olympiadbench/beyondaime), supergpqa, mmlu-pro, demo, and inline dataset options.

#### `init` - Generate Config File

Create a sample configuration file:

```bash
# Basic template
uv run python -m themis.cli init

# MATH-500 with OpenAI-compatible endpoint
uv run python -m themis.cli init --template math500 --output my_config.yaml

# Inline dataset template
uv run python -m themis.cli init --template inline --output custom.yaml
```

Available templates:
- `basic` - Simple fake provider setup for testing
- `math500` - MATH-500 benchmark with OpenAI-compatible provider
- `inline` - Custom inline dataset with examples

#### `validate-config` - Validate Configuration

Check a config file for errors without running:

```bash
uv run python -m themis.cli validate-config --config my_config.yaml
```

Shows parsed configuration and identifies errors before running expensive experiments.

#### `new-project` - Create a New Project

Scaffold a new project directory:

```bash
uv run python -m themis.cli new-project --project-name my-new-project
```

This will create a new directory called `my-new-project` with a basic project structure, including a sample configuration file, a CLI script, and a README.

---

### Experiment Commands

Commands for running experiments:

#### `demo` - Built-in Smoke Test

Quick verification using fake providers:

```bash
uv run python -m themis.cli demo [OPTIONS]

Options:
  --log-level TEXT    Logging verbosity: critical|error|warning|info|debug|trace
  --help             Show help message
```

#### `math500` - MATH-500 Benchmark

Zero-shot evaluation on the MATH-500 dataset:

```bash
uv run python -m themis.cli math500 [OPTIONS]

Options:
  --source TEXT       Dataset source: 'huggingface' (default) or 'local'
  --data-dir PATH     Local MATH-500 directory (if --source local)
  --limit INTEGER     Limit number of samples (for testing)
  --storage PATH      Cache directory (default: .cache/themis)
  --run-id TEXT       Unique run identifier for resumability
  --temperature FLOAT Model temperature (default: 0.0)
  --log-level TEXT    Logging verbosity
  --resume / --no-resume  Resume from cache (default: true)

Examples:
  # Quick test with 50 samples
  uv run python -m themis.cli math500 --limit 50

  # Full evaluation with custom storage
  uv run python -m themis.cli math500 \
    --storage .cache/math500-eval \
    --run-id run-2024-01-15 \
    --temperature 0.0

  # Use local MATH-500 dataset
  uv run python -m themis.cli math500 \
    --source local \
    --data-dir /path/to/MATH-500 \
    --limit 100
```

#### Competition Math Benchmarks (`aime24`, `aime25`, `amc23`, `olympiadbench`, `beyondaime`)

Zero-shot evaluation across multiple competition datasets:

```bash
uv run python -m themis.cli <benchmark> [OPTIONS]

Benchmarks:
  aime24, aime25, amc23, olympiadbench, beyondaime

Options:
  --source TEXT       Dataset source: 'huggingface' (default) or 'local'
  --split TEXT        Dataset split to load (default: test)
  --data-dir PATH     Local dataset directory (if --source local)
  --limit INTEGER     Limit number of samples (for smoke tests)
  --subjects TEXT...  Optional subject filters (if provided by the dataset)
  --storage PATH      Cache directory for generations
  --run-id TEXT       Unique run identifier for resumability
  --temperature FLOAT Model temperature (default: 0.0)
  --log-level TEXT    Logging verbosity
  --resume / --no-resume  Resume from cache (default: true)
```

#### `supergpqa` - SuperGPQA Benchmark

Multiple-choice evaluation on the SuperGPQA dataset:

```bash
uv run python -m themis.cli supergpqa [OPTIONS]

Options:
  --source TEXT       Dataset source: 'huggingface' (default) or 'local'
  --split TEXT        Dataset split to load (default: test)
  --data-dir PATH     Local dataset directory (if --source local)
  --limit INTEGER     Limit number of samples (for quick smoke tests)
  --subjects TEXT...  Optional category filters
  --storage PATH      Cache directory for generations
  --run-id TEXT       Unique run identifier for resumability
  --temperature FLOAT Model temperature (default: 0.0)
  --log-level TEXT    Logging verbosity
  --resume / --no-resume  Resume from cache (default: true)
```

#### `mmlu-pro` - MMLU-Pro Benchmark

Professional-level multiple-choice evaluation with refined distractors:

```bash
uv run python -m themis.cli mmlu-pro [OPTIONS]

Options:
  --source TEXT       Dataset source: 'huggingface' (default) or 'local'
  --split TEXT        Dataset split to load (default: test)
  --data-dir PATH     Local dataset directory (if --source local)
  --limit INTEGER     Limit number of samples
  --subjects TEXT...  Optional subject filters
  --storage PATH      Cache directory for generations
  --run-id TEXT       Unique run identifier for resumability
  --temperature FLOAT Model temperature (default: 0.0)
  --log-level TEXT    Logging verbosity
  --resume / --no-resume  Resume from cache (default: true)
```

#### `run-config` - Config-Driven Experiments

Execute experiments defined in YAML config files:

```bash
uv run python -m themis.cli run-config [OPTIONS]

Options:
  --config PATH       Path to YAML configuration file
  --overrides TEXT    Hydra-style overrides (space-separated)
  --log-level TEXT    Logging verbosity

Examples:
  # Generate a config file first
  uv run python -m themis.cli init --output my_config.yaml

  # Run from config
  uv run python -m themis.cli run-config --config my_config.yaml

  # Override specific parameters
  uv run python -m themis.cli run-config \
    --config my_config.yaml \
    --overrides "generation.sampling.temperature=0.2 max_samples=100"

  # Multiple overrides
  uv run python -m themis.cli run-config \
    --config my_config.yaml \
    --overrides "storage.run_id=new-run dataset.limit=50" \
    --log-level debug
```

---

## Configuration

### Config File Structure

Themis uses JSON or YAML for configuration. Here's a complete example:

```json
{
  "run_id": "my-experiment",
  "storage_dir": ".cache/my-experiment",
  "resume": true,
  "models": [
    {
      "name": "gpt-4",
      "provider": "litellm",
      "provider_options": {
        "api_key": "sk-...",
        "timeout": 60,
        "n_parallel": 10
      }
    },
    {
      "name": "claude-3-opus-20240229",
      "provider": "anthropic",
      "provider_options": {
        "timeout": 120
      }
    }
  ],
  "samplings": [
    {"name": "greedy", "temperature": 0.0, "max_tokens": 512},
    {"name": "creative", "temperature": 0.8, "max_tokens": 512}
  ],
  "datasets": [
    {
      "name": "math500",
      "kind": "math500_hf",
      "limit": 50
    }
  ]
}
```

### Configuration Options

**Core settings:**
- `run_id`: Unique identifier for caching and resumability
- `storage_dir`: Where to cache generations and results
- `resume`: Continue from previous runs (default: true)

**Models:**
- `name`: Model identifier
- `provider`: `litellm`, `openai`, `anthropic`, `azure`, `bedrock`, `fake`, or custom
- `provider_options`: Provider-specific configuration (API keys, endpoints, timeouts)

**Samplings:**
- `name`: Sampling strategy name
- `temperature`: Randomness (0.0 = deterministic, 1.0+ = creative)
- `max_tokens`: Maximum response length
- `top_p`, `top_k`: Nucleus/top-k sampling (optional)

**Datasets:**
- `name`: Dataset identifier
- `kind`: `demo`, `math500_hf`, `math500_local`, `inline`, or custom
- `limit`: Maximum samples (for testing)
- `source_path`: Path for local datasets (optional)

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the complete schema and [`docs/EXAMPLES.md`](docs/EXAMPLES.md) for common recipes.

---

## Architecture

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

### Extension Points

Themis is designed for extensibility:

- **Custom providers**: Implement `Provider` interface for new LLM APIs
- **Custom datasets**: Implement `DatasetLoader` for new data sources
- **Custom metrics**: Implement `Metric` interface for domain-specific evaluation
- **Custom runners**: Override generation loops for specialized workflows
- **Custom pipelines**: Build evaluation pipelines with custom extractors

See [`docs/ADDING_COMPONENTS.md`](docs/ADDING_COMPONENTS.md) for detailed extension guides and [`examples/advanced/`](examples/advanced/) for working examples.

---

## Development

### Running Tests

```bash
# Full test suite
uv run pytest

# Specific test file
uv run pytest tests/generation/test_strategies.py

# With coverage
uv run pytest --cov=themis --cov-report=html

# Verbose output
uv run pytest -v
```

### Project Commands

```bash
# Smoke test core CLI
uv run python -m themis.cli demo

# Test example pipelines
uv run python -m examples.getting_started.cli run --dry-run

# Run with specific storage (keeps cache local)
uv run python -m themis.cli math500 \
  --storage .cache/dev-test \
  --limit 5
```

### Code Style

- Python 3.12+, PEP 8 (4-space indent)
- Type hints throughout (mypy-compatible)
- Dataclasses and Pydantic models for configs/entities
- File names: `snake_case`
- Classes: `PascalCase`
- CLI commands: `dashed-names` (handled by Cyclopts)

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `uv run pytest` to verify
5. Submit a pull request with:
   - Summary of changes
   - Test evidence
   - Documentation updates (if applicable)
   - Links to related issues

---

## Documentation

### Core Documentation

- **[examples/README.md](examples/README.md)** - Comprehensive tutorial cookbook (START HERE!)
- **[COOKBOOK.md](COOKBOOK.md)** - Quick reference and cheat sheet
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** - Complete configuration schema
- **[docs/ADDING_COMPONENTS.md](docs/ADDING_COMPONENTS.md)** - Extension guide
- **[docs/DIAGRAM.md](docs/DIAGRAM.md)** - Architecture diagrams
- **[docs/EXAMPLES.md](docs/EXAMPLES.md)** - Additional recipes and patterns
- **[AGENTS.md](AGENTS.md)** - Repository guidelines for AI agents

### Key Features

✅ **Resumability**: Automatic caching by `run_id`—interrupted runs pick up where they left off  
✅ **Grid Search**: Cartesian product over models × samplings × prompts  
✅ **Progress Tracking**: tqdm progress bars and structured logging  
✅ **Type Safety**: Pydantic validation for configs and runtime entities  
✅ **Provider Agnostic**: Unified interface for 100+ providers via LiteLLM (OpenAI, Anthropic, Azure, Bedrock, local LLMs, etc.)
✅ **Math Evaluation**: Built-in math-verify integration for numeric correctness  
✅ **Export Options**: CSV, JSON, HTML output formats  
✅ **Multi-Experiment Projects**: Organize related experiments with shared configs  
✅ **Integrations**: Log results to Weights & Biases and upload artifacts to Hugging Face Hub

---

## Use Cases

### Academic Research
- Systematic prompt engineering experiments
- Model comparison studies
- Benchmark evaluations (MATH, GSM8K, etc.)
- Reproducible experiment pipelines

### LLM Development
- Testing model variants during training
- Evaluating fine-tuned models
- Comparing sampling strategies
- A/B testing prompts and templates

### Production Monitoring
- Regression testing for model updates
- Quality assurance on real-world examples
- Performance benchmarking
- Cost/latency analysis

---


---

## Citation

If you use Themis in your research, please cite it using the following metadata:

```yaml
@software{themis2025,
  author = {Pittawat Taveekitworachai},
  title = {Themis: Lightweight evaluation platform for LLM experiments},
  year = {2025},
  url = {https://github.com/Pittawat2542/themis}
}
```

See [`CITATION.cff`](CITATION.cff) for more details.

---

## Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on how to get started, run tests, and submit pull requests.

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Getting Help

- **Start with examples**: [`examples/README.md`](examples/README.md) has comprehensive tutorials
- **Quick reference**: [`COOKBOOK.md`](COOKBOOK.md) for common patterns
- **Check docs**: [`docs/`](docs/) directory has detailed guides
- **Search issues**: Look for similar problems in GitHub issues
- **Create issue**: Open a new issue with your question or bug report

---

## Roadmap

- [x] Universal provider support via LiteLLM (100+ providers)
- [ ] Web UI for experiment visualization
- [ ] Distributed generation support
- [ ] Advanced metrics (BLEU, ROUGE, semantic similarity)
- [ ] Experiment comparison dashboard
- [ ] Cost tracking and optimization

---

**Happy experimenting!** 🚀

For a guided introduction, start with the [examples](examples/README.md). For quick lookups, check the [cookbook](COOKBOOK.md).

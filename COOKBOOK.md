# Themis Cookbook - Quick Reference

This is a quick reference guide to the Themis examples cookbook. For detailed tutorials, see [`examples/README.md`](examples/README.md).

## 🎯 Which Example Should I Start With?

| Your Goal | Start Here | Time |
|-----------|------------|------|
| I'm completely new to Themis | [getting_started](examples/getting_started/) | 15 min |
| I want to run systematic experiments | [config_file](examples/config_file/) | 20 min |
| I want to master prompt engineering | [prompt_engineering](examples/prompt_engineering/) | 25 min |
| I'm managing multiple experiments | [projects](examples/projects/) | 45 min |
| I need custom behavior | [advanced](examples/advanced/) | 60 min |

## 🚀 Quick Start Commands

### Run Your First Experiment
```bash
# Preview what will run
uv run python -m examples.getting_started.cli run --dry-run

# Run it
uv run python -m examples.getting_started.cli run

# Export results
uv run python -m examples.getting_started.cli run --csv-output results.csv
```

### Use Configuration Files
```bash
# Run with a config
uv run python -m examples.config_file.cli run --config-path examples/config_file/config.sample.json

# Override settings from CLI
uv run python -m examples.config_file.cli run \
  --config-path examples/config_file/config.sample.json \
  --run-id my-experiment \
  --resume false
```

### Master Prompt Engineering
```bash
# Run prompt engineering experiment
uv run python -m examples.prompt_engineering.cli run

# Run with analysis
uv run python -m examples.prompt_engineering.cli run --analyze

# Export results for analysis
uv run python -m examples.prompt_engineering.cli run \
  --csv-output results.csv \
  --html-output results.html
```

## 📚 Example Overview

### 1. Getting Started
**Location:** `examples/getting_started/`

**What you'll learn:**
- How to define prompt templates
- How to configure models and sampling parameters
- How to run experiments programmatically or via CLI
- How to view and export results

**Key files:**
- `README.md` - Comprehensive tutorial
- `config.sample.json` - Basic configuration
- `experiment.py` - Programmatic example

### 2. Config File
**Location:** `examples/config_file/`

**What you'll learn:**
- How to structure configuration files
- How to run grid searches (multiple models × samplings)
- How to use CLI overrides
- How to leverage resumability and caching

**Key files:**
- `README.md` - Configuration guide
- `compare_sampling.json` - Temperature comparison
- `compare_models.json` - Model comparison
- `grid_search.json` - Full grid search example

### 3. Prompt Engineering
**Location:** `examples/prompt_engineering/`

**What you'll learn:**
- How to systematically compare different prompting strategies
- Zero-shot vs few-shot vs chain-of-thought prompting
- How to analyze prompt effectiveness across models
- How to export and analyze results

**Key files:**
- `README.md` - Comprehensive prompt engineering guide
- `USAGE.md` - Detailed usage examples
- `config.sample.json` - Example configuration
- `prompts.py` - Prompt template definitions
- `results_analysis.py` - Analysis utilities

**Key concepts:**
- Prompt variations and strategy comparison
- Built-in metrics for evaluation
- Export to CSV, JSON, HTML formats
- Systematic experimentation workflows

### 4. Projects
**Location:** `examples/projects/`

**What you'll learn:**
- How to organize multiple experiments in a project
- How to share configurations across experiments
- How to run experiments selectively
- How to maintain reproducible research workflows

**Key concepts:**
- Project structure and metadata
- Experiment definitions and tags
- Systematic evaluation campaigns

### 5. Advanced
**Location:** `examples/advanced/`

**What you'll learn:**
- How to override generation loops with custom runners
- How to create custom evaluation pipelines
- How to implement domain-specific metrics
- How to build agentic workflows with multi-step generation
- How to add instrumentation and debugging

**Advanced topics:**
- Custom generation runners (prioritized, batched)
- Subject-aware evaluation pipelines
- Instrumented provider routers
- Agentic workflows (plan → execute → verify)
- Custom metrics with partial credit

## 🔧 Common Configuration Patterns

### Single Model, Single Sampling
```json
{
  "run_id": "simple-test",
  "storage_dir": ".cache/simple-test",
  "resume": true,
  "models": [
    {"name": "my-model", "provider": "fake"}
  ],
  "samplings": [
    {"name": "greedy", "temperature": 0.0, "max_tokens": 512}
  ],
  "datasets": [
    {"name": "demo", "kind": "demo", "limit": 10}
  ]
}
```

### Grid Search (2 Models × 3 Temperatures)
```json
{
  "run_id": "grid-search",
  "storage_dir": ".cache/grid-search",
  "resume": true,
  "models": [
    {"name": "model-a", "provider": "fake"},
    {"name": "model-b", "provider": "fake"}
  ],
  "samplings": [
    {"name": "greedy", "temperature": 0.0, "max_tokens": 512},
    {"name": "balanced", "temperature": 0.7, "max_tokens": 512},
    {"name": "creative", "temperature": 1.0, "max_tokens": 512}
  ],
  "datasets": [
    {"name": "demo", "kind": "demo", "limit": 20}
  ]
}
```

### OpenAI-Compatible Endpoint
```json
{
  "run_id": "openai-test",
  "storage_dir": ".cache/openai-test",
  "resume": true,
  "models": [
    {
      "name": "local-llm",
      "provider": "openai-compatible",
      "provider_options": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "model_mapping": {
          "local-llm": "qwen2.5-7b-instruct"
        },
        "timeout": 60,
        "n_parallel": 2
      }
    }
  ],
  "samplings": [
    {"name": "standard", "temperature": 0.7, "max_tokens": 512}
  ],
  "datasets": [
    {"name": "math500", "kind": "math500_hf", "limit": 50}
  ]
}
```

### Integrations

Enable Weights & Biases and Hugging Face Hub integrations:

```json
{
  "run_id": "integrated-experiment",
  "storage_dir": ".cache/integrated-experiment",
  "resume": true,
  "models": [
    {"name": "my-model", "provider": "fake"}
  ],
  "samplings": [
    {"name": "greedy", "temperature": 0.0, "max_tokens": 512}
  ],
  "datasets": [
    {"name": "demo", "kind": "demo", "limit": 10}
  ],
  "integrations": {
    "wandb": {
      "enable": true,
      "project": "themis-experiments",
      "entity": "your-wandb-entity",
      "tags": ["smoke-test", "demo"]
    },
    "huggingface_hub": {
      "enable": true,
      "repository": "your-hf-username/themis-results"
    }
  }
}
```

## 🛠️ Common Tasks

### Creating a New Project

Scaffold a new Themis project with a basic structure:

```bash
uv run python -m themis.cli new-project --project-name my-themis-project
```

This command creates a directory `my-themis-project` containing:
- `config.sample.json`: A basic experiment configuration.
- `cli.py`: A simple CLI to run experiments defined in `config.sample.json`.
- `README.md`: Project-specific instructions.

After creation, navigate into the new project directory and customize `config.sample.json`.

### Preview Configuration (Dry Run)
```bash
uv run python -m examples.getting_started.cli run --dry-run
```

### Run with Custom Storage
```bash
uv run python -m examples.getting_started.cli run \
  --storage-dir .cache/my-experiment \
  --run-id experiment-2024-01-15
```

### Export Results
```bash
uv run python -m examples.getting_started.cli run \
  --csv-output results.csv \
  --html-output results.html \
  --json-output results.json
```

### Limit Dataset Size (Quick Testing)
```bash
# Use "limit" in config.json datasets section
uv run python -m examples.getting_started.cli run \
  --config-path config.sample.json

# Or modify config to add:
# "datasets": [{"name": "demo", "kind": "demo", "limit": 5}]
```

### Disable Resume (Force Re-run)
```bash
uv run python -m examples.getting_started.cli run --resume false
```

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Always run from the project root
cd /path/to/themis
uv run python -m examples.getting_started.cli run
```

### "Connection refused" (OpenAI compatible)
```bash
# Verify server is running
curl http://localhost:1234/v1/models

# Check port in config matches your server
# Try 127.0.0.1 instead of localhost
```

### Results not updating
```bash
# Check if resume is enabled and using cached results
# Solution 1: Disable resume
uv run python -m examples.getting_started.cli run --resume false

# Solution 2: Use a new run ID
uv run python -m examples.getting_started.cli run --run-id new-run-1

# Solution 3: Clear cache
rm -rf .cache/your-storage-dir
```

### Slow performance
```bash
# 1. Reduce parallelism in provider_options
"n_parallel": 1  # in config.json

# 2. Use limit for quick testing
"limit": 5  # in datasets section

# 3. Use fake provider for testing
"provider": "fake"  # in config.json
```

### Invalid configuration
```bash
# Validate JSON syntax
python -m json.tool your_config.json

# Check for:
# - Trailing commas
# - Missing quotes
# - Incorrect brackets
```

## 📖 Learning Path

1. **Start with `getting_started`** (15 min)
   - Run the example
   - Read the README
   - Modify the config

2. **Try `config_file`** (20 min)
   - Use configuration files
   - Run grid searches
   - Learn about resumability

3. **Master prompt engineering with `prompt_engineering`** (25 min)
   - Compare zero-shot vs few-shot vs chain-of-thought
   - Analyze prompt effectiveness across models
   - Export results for analysis

4. **Organize with `projects`** (45 min)
   - Create a project structure
   - Define multiple experiments
   - Run experiments selectively

5. **Customize with `advanced`** (60 min)
   - Override generation loops
   - Create custom metrics
   - Build agentic workflows

## 🎓 Best Practices

1. **Always use resumability**: Set `"resume": true` in configs
2. **Start small**: Use `"limit": 5` for quick testing
3. **Use descriptive IDs**: `run_id: "2024-01-15-temperature-sweep"`
4. **Separate storage**: Different `storage_dir` per experiment
5. **Export results**: Always use `--csv-output` for analysis
6. **Version control configs**: Commit configurations to git
7. **Dry run first**: Use `--dry-run` to preview
8. **Document experiments**: Add descriptions to Project metadata

## 📚 Additional Resources

- **Full tutorials**: [`examples/README.md`](examples/README.md)
- **API documentation**: [`docs/ADDING_COMPONENTS.md`](docs/ADDING_COMPONENTS.md)
- **Architecture**: [`docs/DIAGRAM.md`](docs/DIAGRAM.md)
- **Configuration**: [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- **More examples**: [`docs/EXAMPLES.md`](docs/EXAMPLES.md)

## 🤝 Getting Help

1. Read the example README for detailed explanations
2. Check error messages - they're usually helpful
3. Search issues in the repository
4. Create a new issue with your question

Happy experimenting! 🚀
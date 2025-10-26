# Themis Cookbook - Quick Reference

This is a quick reference guide to the Themis examples cookbook. For detailed tutorials, see [`examples/README.md`](examples/README.md).

## 🎯 Which Example Should I Start With?

| Your Goal | Start Here | Time |
|-----------|------------|------|
| I'm completely new to Themis | [getting_started](examples/getting_started/) | 15 min |
| I want to run systematic experiments | [config_file](examples/config_file/) | 20 min |
| I need to use real LLMs | [openai_compatible](examples/openai_compatible/) | 30 min |
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

### Connect to Real LLMs
```bash
# Start your local LLM server (e.g., LM Studio, Ollama)
# Then edit examples/openai_compatible/config.sample.json

# Run on MATH-500 benchmark
uv run python -m examples.openai_compatible.cli run \
  --config-path examples/openai_compatible/config.sample.json \
  --n-records 10
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

### 3. OpenAI Compatible
**Location:** `examples/openai_compatible/`

**What you'll learn:**
- How to connect to LM Studio, Ollama, vLLM, OpenAI
- How to configure API endpoints and authentication
- How to handle timeouts and parallelism
- How to run on real benchmarks (MATH-500)

**Key files:**
- `README.md` - Comprehensive endpoint guide
- `config.sample.json` - Local LLM configuration
- `config.comprehensive.json` - Full-featured example

**Supported endpoints:**
- LM Studio (`http://localhost:1234/v1`)
- Ollama (`http://localhost:11434/v1`)
- vLLM (`http://localhost:8000/v1`)
- OpenAI API (`https://api.openai.com/v1`)
- Any OpenAI-compatible server

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

## 🛠️ Common Tasks

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
# Using n-records flag (OpenAI compatible example)
uv run python -m examples.openai_compatible.cli run \
  --config-path config.sample.json \
  --n-records 5

# Or use "limit" in config.json datasets section
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

# 3. Use --n-records flag
uv run python -m examples.openai_compatible.cli run --n-records 3
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

3. **Connect to real LLMs with `openai_compatible`** (30 min)
   - Set up a local LLM server
   - Configure the endpoint
   - Run on MATH-500

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
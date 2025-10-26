# Themis Examples Cookbook

This directory contains practical, hands-on examples that teach you how to use Themis effectively. Each example builds on the previous ones, forming a comprehensive learning path.

## Learning Path

### 🚀 [getting_started](getting_started/) - Your First Experiment

**Start here if you're new to Themis.**

Learn the absolute basics:
- Define a prompt template
- Configure models and sampling
- Run an experiment programmatically or via CLI
- View and export results

**Time to complete:** 15 minutes

**Key concepts:** Prompt templates, model specs, sampling configs, exact match metrics

**Example commands:**
```bash
# Run your first experiment
uv run python -m examples.getting_started.cli run

# Preview configuration
uv run python -m examples.getting_started.cli run --dry-run

# Export results
uv run python -m examples.getting_started.cli run --csv-output results.csv
```

---

### ⚙️ [config_file](config_file/) - Configuration-Driven Workflows

**Perfect for systematic experimentation.**

Learn how to drive experiments entirely through configuration files:
- Structure config files for reproducibility
- Compare multiple models and sampling strategies
- Override configs from the command line
- Use resumability and caching effectively

**Time to complete:** 20 minutes

**Key concepts:** JSON configs, CLI overrides, grid searches, resumability

**Example commands:**
```bash
# Run with a config file
uv run python -m examples.config_file.cli run --config-path compare_sampling.json

# Override from CLI
uv run python -m examples.config_file.cli run --config-path config.sample.json --run-id my-run

# Grid search: 2 models × 4 temperatures
uv run python -m examples.config_file.cli run --config-path grid_search.json
```

---

### 🌐 [openai_compatible](openai_compatible/) - Real LLM Endpoints

**Connect to actual language models.**

Learn how to use Themis with real LLM servers:
- Configure OpenAI-compatible endpoints (LM Studio, Ollama, vLLM)
- Set up local LLM servers
- Handle authentication and API keys
- Run experiments on real benchmarks (MATH-500)
- Troubleshoot connection and timeout issues

**Time to complete:** 30 minutes

**Key concepts:** Provider options, API authentication, model mapping, timeouts, parallelism

**Example commands:**
```bash
# Quick test with local LLM
uv run python -m examples.openai_compatible.cli run --config-path config.sample.json --n-records 5

# Full evaluation
uv run python -m examples.openai_compatible.cli run --config-path my_config.json

# Export for analysis
uv run python -m examples.openai_compatible.cli run \
  --config-path my_config.json \
  --csv-output results.csv \
  --html-output results.html
```

---

### 📊 [projects](projects/) - Organizing Multiple Experiments

**Scale up to research-level organization.**

Learn how to manage complex projects with multiple experiments:
- Group related experiments in a Project
- Share configurations across experiments
- Run experiments selectively
- Compare results across different approaches
- Maintain reproducible research workflows

**Time to complete:** 45 minutes

**Key concepts:** Project structure, experiment definitions, metadata, tags, systematic evaluation

**Example commands:**
```bash
# List all experiments in project
uv run python -m examples.projects.cli list-experiments

# Run specific experiment
uv run python -m examples.projects.cli run --experiment zero-shot

# Run with custom settings
uv run python -m examples.projects.cli run \
  --experiment few-shot \
  --storage-dir .cache/my-project \
  --limit 10
```

---

### 🔧 [advanced](advanced/) - Advanced Customization

**For power users who need full control.**

Learn how to customize and extend Themis:
- Override generation loops with custom runners
- Create custom evaluation pipelines
- Implement domain-specific metrics
- Add instrumentation and debugging
- Build agentic workflows with multi-step generation

**Time to complete:** 60 minutes

**Key concepts:** Custom runners, evaluation pipelines, metrics, agentic workflows, instrumentation

**Example commands:**
```bash
# Run with subject-aware evaluation
uv run python -m examples.advanced.cli run --enable-subject-breakdown

# Use chain-of-thought prompting
uv run python -m examples.advanced.cli run --prompt-style cot

# Full custom configuration
uv run python -m examples.advanced.cli run \
  --prompt-style cot \
  --enable-subject-breakdown \
  --config-path examples/advanced/config.sample.json
```

---

## Quick Reference

### Common Tasks

**Run a quick test:**
```bash
cd examples/getting_started
uv run python -m examples.getting_started.cli run --dry-run
```

**Compare models:**
```bash
cd examples/config_file
uv run python -m examples.config_file.cli run --config-path compare_models.json
```

**Evaluate on real data:**
```bash
cd examples/openai_compatible
uv run python -m examples.openai_compatible.cli run --config-path config.sample.json
```

**Organize experiments:**
```bash
cd examples/projects
uv run python -m examples.projects.cli list-experiments
uv run python -m examples.projects.cli run --experiment zero-shot
```

**Custom behavior:**
```bash
cd examples/advanced
uv run python -m examples.advanced.cli run --enable-subject-breakdown
```

### Configuration Patterns

**Single model, single sampling:**
```json
{
  "models": [{"name": "model-1", "provider": "fake"}],
  "samplings": [{"name": "greedy", "temperature": 0.0, "max_tokens": 512}]
}
```

**Grid search (M models × N samplings):**
```json
{
  "models": [
    {"name": "model-1", "provider": "fake"},
    {"name": "model-2", "provider": "fake"}
  ],
  "samplings": [
    {"name": "greedy", "temperature": 0.0, "max_tokens": 512},
    {"name": "creative", "temperature": 0.8, "max_tokens": 512}
  ]
}
```

**OpenAI-compatible endpoint:**
```json
{
  "models": [{
    "name": "my-model",
    "provider": "openai-compatible",
    "provider_options": {
      "base_url": "http://localhost:1234/v1",
      "api_key": "not-needed",
      "model_mapping": {"my-model": "actual-model-name"}
    }
  }]
}
```

## Troubleshooting

### Common Issues

**Can't find the example:**
```bash
# Always run from project root
cd /path/to/themis
uv run python -m examples.getting_started.cli run
```

**Connection refused (OpenAI compatible):**
1. Verify server is running: `curl http://localhost:1234/v1/models`
2. Check port number in config
3. Try `127.0.0.1` instead of `localhost`

**Results not updating:**
- Check `resume: true` in config - may be using cached results
- Try `--resume false` or use a new `run_id`

**Slow performance:**
- Reduce `n_parallel` if server is overloaded
- Use `limit` to test with fewer samples first
- Check CPU/GPU usage on server

## File Structure

```
examples/
├── README.md                      # This file
├── getting_started/               # Basics
│   ├── README.md
│   ├── cli.py
│   ├── config.py
│   ├── config.sample.json
│   ├── datasets.py
│   └── experiment.py
├── config_file/                   # Configuration-driven
│   ├── README.md
│   ├── cli.py
│   ├── config.sample.json
│   ├── compare_sampling.json
│   ├── compare_models.json
│   └── grid_search.json
├── openai_compatible/             # Real LLM endpoints
│   ├── README.md
│   ├── cli.py
│   ├── config.sample.json
│   └── config.comprehensive.json
├── projects/                      # Multi-experiment projects
│   ├── README.md
│   ├── cli.py
│   └── project_setup.py
└── advanced/                      # Advanced customization
    ├── README.md
    ├── cli.py
    ├── generation.py
    ├── pipeline.py
    └── config.sample.json
```

## Tips for Learning

1. **Start with getting_started** - Don't skip ahead! The basics are essential.

2. **Run every example** - Reading isn't enough. Run the code to understand it.

3. **Modify configs** - Copy `config.sample.json` and experiment with different values.

4. **Use dry-run mode** - Preview what will happen: `--dry-run`

5. **Start small** - Use `limit: 5` or `--n-records 5` for quick iteration.

6. **Check storage dirs** - Results are cached in `storage_dir` for resumability.

7. **Export results** - Use `--csv-output` and `--html-output` for analysis.

8. **Read error messages** - They usually tell you exactly what's wrong.

## Next Steps

After completing these examples:

1. **Read the docs**: Check `docs/ADDING_COMPONENTS.md` for extension points
2. **Review the source**: `themis/` contains well-documented code
3. **Check the tests**: `tests/` show usage patterns
4. **Build your own**: Create a new experiment for your use case

## Getting Help

- **Documentation**: See `docs/` directory
- **API Reference**: Check docstrings in `themis/` modules
- **Issues**: Search or create issues on the repository
- **Examples**: These examples are your best resource!

## Contributing Examples

Have a useful example to share? Contributions are welcome!

1. Follow the existing structure (README, CLI, configs)
2. Include clear explanations and multiple use cases
3. Test all commands in the README
4. Submit a pull request

Happy experimenting! 🚀
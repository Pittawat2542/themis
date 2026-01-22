# CLI Reference

Complete guide to the Themis command-line interface.

## Overview

Themis provides 5 essential commands:

```bash
themis eval        # Run evaluations
themis compare     # Compare runs statistically
themis serve       # Start API server
themis list        # List resources
themis clean       # Clean storage
```

---

## themis eval

Run an LLM evaluation on a benchmark or dataset.

### Synopsis

```bash
themis eval BENCHMARK --model MODEL [OPTIONS]
```

### Arguments

**`BENCHMARK`** (required)

Benchmark name or dataset. Built-in benchmarks:
- `demo` - Quick testing (10 samples)
- `gsm8k` - Grade school math (8.5K)
- `math500` - Advanced math (500)
- `aime24` - Math competition (30)
- `mmlu_pro` - General knowledge
- `supergpqa` - Advanced reasoning

### Options

**`--model MODEL`** (required)

Model identifier. Examples:
- `gpt-4` - OpenAI GPT-4
- `gpt-3.5-turbo` - OpenAI GPT-3.5
- `claude-3-opus-20240229` - Anthropic Claude
- `azure/gpt-4` - Azure OpenAI
- `ollama/llama3` - Local Ollama

**`--limit N`**

Evaluate first N samples (default: all)

**`--temperature FLOAT`**

Sampling temperature 0.0-2.0 (default: 0.0)

**`--max-tokens INT`**

Maximum tokens to generate (default: 512)

**`--num-samples INT`**

Samples per prompt (default: 1)

**`--workers INT`**

Parallel workers (default: 4)

**`--run-id STR`**

Unique run identifier (default: auto-generated)

**`--storage PATH`**

Storage directory (default: .cache/experiments)

**`--resume / --no-resume`**

Resume from cache (default: --resume)

**`--output FILE`**

Export results (.json, .csv, .html)

### Examples

```bash
# Basic evaluation
themis eval gsm8k --model gpt-4

# With limit
themis eval gsm8k --model gpt-4 --limit 100

# Custom configuration
themis eval gsm8k \
  --model gpt-4 \
  --temperature 0.7 \
  --max-tokens 1024 \
  --workers 16 \
  --run-id gpt4-experiment-1

# Export results
themis eval gsm8k --model gpt-4 --limit 100 --output results.json

# Test without API key
themis eval demo --model fake-math-llm --limit 5
```

---

## themis compare

Compare multiple runs with statistical tests.

### Synopsis

```bash
themis compare RUN_ID_1 RUN_ID_2 [RUN_ID_3...] [OPTIONS]
```

### Arguments

**`RUN_IDS`** (required)

At least 2 run IDs to compare

### Options

**`--storage PATH`**

Storage directory (default: .cache/experiments)

**`--test NAME`**

Statistical test: `t_test`, `bootstrap`, `permutation`, `none` (default: bootstrap)

**`--alpha FLOAT`**

Significance level (default: 0.05 for 95% confidence)

**`--output FILE`**

Export comparison (.json, .html, .md)

**`--verbose`**

Show detailed pairwise comparisons

### Examples

```bash
# Basic comparison
themis compare run-1 run-2

# With specific test
themis compare run-1 run-2 --test bootstrap --alpha 0.05

# Compare 3+ runs
themis compare run-1 run-2 run-3

# Export to HTML
themis compare run-1 run-2 --output comparison.html

# Verbose output
themis compare run-1 run-2 --verbose
```

---

## themis serve

Start the API server with web dashboard.

### Synopsis

```bash
themis serve [OPTIONS]
```

### Options

**`--port INT`**

Port to run server on (default: 8080)

**`--host STR`**

Host to bind to (default: 127.0.0.1)

**`--storage PATH`**

Storage directory (default: .cache/experiments)

**`--reload`**

Enable auto-reload for development

### Examples

```bash
# Default (localhost:8080)
themis serve

# Custom port
themis serve --port 3000

# Public access
themis serve --host 0.0.0.0 --port 8080

# Development mode
themis serve --reload
```

### Endpoints

Once running:
- **Dashboard**: `http://localhost:8080/dashboard`
- **API Docs**: `http://localhost:8080/docs`
- **Health**: `http://localhost:8080/`
- **Runs API**: `http://localhost:8080/api/runs`

---

## themis list

List available resources.

### Synopsis

```bash
themis list WHAT [OPTIONS]
```

### Arguments

**`WHAT`** (required)

What to list:
- `runs` - All experiment runs
- `benchmarks` - Available benchmarks
- `metrics` - Available metrics

### Options

**`--storage PATH`**

Storage directory (for listing runs)

**`--limit N`**

Limit number of results

### Examples

```bash
# List runs
themis list runs

# List benchmarks
themis list benchmarks

# List metrics
themis list metrics

# Limit results
themis list runs --limit 10
```

---

## themis clean

Clean storage and cached results.

### Synopsis

```bash
themis clean [OPTIONS]
```

### Options

**`--storage PATH`**

Storage directory to clean (default: .cache/experiments)

**`--run-id STR`**

Clean specific run only

**`--all`**

Clean all runs (requires confirmation)

### Examples

```bash
# Clean specific run
themis clean --run-id my-experiment

# Clean all (with confirmation)
themis clean --all

# Clean custom storage
themis clean --storage ~/experiments --all
```

---

## Global Options

These options work with all commands:

**`--help`**

Show help message

**`--version`**

Show version information

### Examples

```bash
# Show help
themis --help
themis eval --help

# Show version
themis --version
```

---

## Environment Variables

### API Keys

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://..."
export AZURE_API_VERSION="2023-05-15"
```

### Themis Settings

```bash
# Default storage
export THEMIS_STORAGE="~/.themis/experiments"

# Log level
export THEMIS_LOG_LEVEL="INFO"
```

---

## Exit Codes

- `0` - Success
- `1` - Error (invalid arguments, API failure, etc.)
- `130` - Interrupted by user (Ctrl+C)

---

## Tips and Tricks

### 1. Test Before Full Run

```bash
# Test with 10 samples first
themis eval gsm8k --model gpt-4 --limit 10

# Then run full evaluation
themis eval gsm8k --model gpt-4
```

### 2. Use Meaningful Run IDs

```bash
# Good
themis eval gsm8k --model gpt-4 --run-id gsm8k-gpt4-baseline-2024-01-15

# Bad (auto-generated, hard to track)
themis eval gsm8k --model gpt-4
```

### 3. Monitor Costs

Export results to check costs:

```bash
themis eval gsm8k --model gpt-4 --limit 100 --output results.json

# Check results.json for cost field
cat results.json | jq '.cost'
```

### 4. Pipeline Commands

Chain commands together:

```bash
# Run evaluation then compare
themis eval gsm8k --model gpt-4 --run-id run-1 && \
themis eval gsm8k --model claude-3-opus --run-id run-2 && \
themis compare run-1 run-2 --output comparison.html
```

### 5. Use Aliases

Create shell aliases for common tasks:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias te='themis eval'
alias tc='themis compare'
alias ts='themis serve'

# Then use
te gsm8k --model gpt-4 --limit 10
tc run-1 run-2
```

---

## Troubleshooting

### Command Not Found

```bash
# Make sure Themis is installed
pip install themis-eval

# Or with uv
uv pip install themis-eval

# Verify installation
which themis
themis --version
```

### Import Errors

```bash
# Install with all features
pip install themis-eval[all]
```

### Permission Denied

```bash
# Check storage directory permissions
ls -la .cache/experiments

# Use different storage location
themis eval gsm8k --model gpt-4 --storage ~/my-experiments
```

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8080

# Use different port
themis serve --port 8081
```

---

## See Also

- [Evaluation Guide](evaluation.md) - Detailed evaluation guide
- [Comparison Guide](../COMPARISON.md) - Statistical comparison
- [API Server](../API_SERVER.md) - Server documentation
- [API Reference](../api/evaluate.md) - Python API documentation

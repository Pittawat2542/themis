# CLI Reference

Complete guide to the Themis command-line interface.

## Overview

Themis provides a focused set of commands:

```bash
themis demo     # Run the demo benchmark
themis eval     # Run evaluations
themis compare  # Compare runs statistically
themis share    # Generate shareable assets
themis serve    # Start API server
themis list     # List runs, benchmarks, metrics
themis clean    # Clean old runs
```

Defaults for storage can be set with `THEMIS_STORAGE`.

---

## themis demo

Run the built-in demo benchmark.

```bash
themis demo --model fake-math-llm --limit 10
```

---

## themis eval

Run an evaluation on a benchmark.

### Synopsis

```bash
themis eval BENCHMARK --model MODEL [OPTIONS]
```

### Options

- `--model MODEL` (required)
- `--limit N`
- `--prompt TEMPLATE`
- `--temperature FLOAT`
- `--max-tokens INT`
- `--workers INT`
- `--run-id STR`
- `--storage PATH`
- `--resume / --no-resume`
- `--output FILE` (`.csv`, `.json`, `.html`)

Notes:
- Custom dataset files are not yet supported via CLI. Use the Python API for custom datasets.

### Examples

```bash
# Basic evaluation
themis eval gsm8k --model gpt-4

# Limit to 100 samples
themis eval gsm8k --model gpt-4 --limit 100

# Custom prompt
themis eval gsm8k --model gpt-4 --prompt "Q: {prompt}\nA:"

# Export results
themis eval gsm8k --model gpt-4 --output results.json
```

---

## themis compare

Compare multiple runs with statistical tests.

### Synopsis

```bash
themis compare RUN_ID_1 RUN_ID_2 [RUN_ID_3...] [OPTIONS]
```

### Options

- `--metric NAME` (limit comparison to one metric)
- `--storage PATH`
- `--output FILE` (`.json`, `.html`, `.md`)
- `--show-diff`

### Example

```bash
themis compare run-1 run-2 --output comparison.html --show-diff
```

---

## themis share

Generate a shareable SVG badge and Markdown snippet for a run.

### Synopsis

```bash
themis share RUN_ID [OPTIONS]
```

### Options

- `--metric NAME` (highlight metric)
- `--storage PATH`
- `--output-dir DIR`

### Example

```bash
themis share run-20260118-032014 --metric accuracy --output-dir share
```

---

## themis serve

Start the API server with REST and WebSocket endpoints.

### Synopsis

```bash
themis serve [OPTIONS]
```

### Options

- `--port INT` (default: 8080)
- `--host STR` (default: 127.0.0.1)
- `--storage PATH`
- `--reload` (dev mode)

Requires `themis[server]`.

---

## themis list

List runs, benchmarks, or metrics.

### Synopsis

```bash
themis list WHAT [OPTIONS]
```

### Options

- `runs` | `benchmarks` | `metrics`
- `--storage PATH`
- `--limit N`
- `--verbose`

### Examples

```bash
themis list benchmarks
themis list metrics
themis list runs --verbose
```

---

## themis clean

Clean runs older than a threshold.

### Synopsis

```bash
themis clean --older-than DAYS [OPTIONS]
```

### Options

- `--storage PATH`
- `--older-than DAYS`
- `--dry-run`

### Example

```bash
# Preview what will be deleted
themis clean --older-than 30 --dry-run

# Delete runs older than 30 days
themis clean --older-than 30
```

---

## Global Options

- `--help`
- `--version`

---

## Environment Variables

### API Keys

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Themis Settings

```bash
# Default storage
export THEMIS_STORAGE="~/.themis/experiments"

# Log level
export THEMIS_LOG_LEVEL="INFO"
```

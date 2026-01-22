# Themis

> **Modern LLM evaluation framework for researchers and practitioners**

Themis makes it easy to evaluate language models systematically with one-liner Python APIs, built-in benchmarks, statistical comparisons, and a web dashboard.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why Themis?

- **🚀 Simple**: One-line Python API or CLI commands—no configuration files needed
- **📊 Comprehensive**: 100+ LLM providers, built-in benchmarks, NLP & code metrics
- **🔬 Statistical**: Compare runs with t-tests, bootstrap, and permutation tests
- **💾 Reliable**: Automatic caching, resume failed runs, smart cache invalidation
- **🌐 Visual**: Web dashboard for exploring results and comparisons
- **🔌 Extensible**: Pluggable backends for custom storage and execution

---

## Quick Start

### Installation

```bash
# Using pip
pip install themis-eval

# Or with uv (recommended)
uv pip install themis-eval

# With optional features
pip install themis-eval[math,nlp,code,server]
```

### One-Liner Evaluation

```python
from themis import evaluate

# Evaluate any model on any benchmark
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    limit=100
)

print(f"Accuracy: {result.metrics['exact_match']:.2%}")
```

### CLI Usage

```bash
# Evaluate a model
themis eval gsm8k --model gpt-4 --limit 100

# Compare two models
themis eval gsm8k --model gpt-4 --limit 100 --run-id gpt4-run
themis eval gsm8k --model claude-3-opus --limit 100 --run-id claude-run
themis compare gpt4-run claude-run

# Start web dashboard
themis serve
```

---

## Features

### 🎯 Built-in Benchmarks

Themis includes 6 popular benchmarks out-of-the-box:

```python
# Math reasoning
evaluate(benchmark="gsm8k", model="gpt-4", limit=100)
evaluate(benchmark="math500", model="gpt-4", limit=50)
evaluate(benchmark="aime24", model="gpt-4")

# General knowledge
evaluate(benchmark="mmlu_pro", model="gpt-4", limit=1000)
evaluate(benchmark="supergpqa", model="gpt-4")

# Quick testing
evaluate(benchmark="demo", model="fake-math-llm", limit=10)
```

**See all available benchmarks:**
```bash
themis list benchmarks
```

### 📈 Rich Metrics

**Math Metrics:**
- Exact Match
- Math Verification (symbolic & numeric)

**NLP Metrics:**
- BLEU, ROUGE, BERTScore, METEOR

**Code Metrics:**
- Pass@k, CodeBLEU, Execution Accuracy

```python
# Use specific metrics
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["exact_match", "bleu", "rouge1"],
)
```

### 🔬 Statistical Comparison

Compare multiple runs with statistical significance testing:

```python
from themis.comparison import compare_runs

report = compare_runs(
    run_ids=["gpt4-run", "claude-run"],
    storage_path=".cache/experiments",
    statistical_test="bootstrap",
    alpha=0.05
)

print(report.summary())
# Shows: win/loss matrices, p-values, effect sizes
```

**CLI:**
```bash
themis compare run-1 run-2 --test bootstrap --output comparison.html
```

### 🌐 Web Dashboard

Start the API server and view results in your browser:

```bash
themis serve

# Open http://localhost:8080/dashboard
# API docs at http://localhost:8080/docs
```

**Features:**
- List all experiment runs
- View detailed results
- Compare multiple runs
- REST API + WebSocket support

### 🔌 100+ LLM Providers

Themis uses [LiteLLM](https://github.com/BerriAI/litellm) for broad provider support:

```python
# OpenAI
evaluate(benchmark="gsm8k", model="gpt-4")

# Anthropic
evaluate(benchmark="gsm8k", model="claude-3-opus-20240229")

# Azure OpenAI
evaluate(benchmark="gsm8k", model="azure/gpt-4")

# Local models (vLLM, Ollama, etc.)
evaluate(benchmark="gsm8k", model="ollama/llama3")

# AWS Bedrock
evaluate(benchmark="gsm8k", model="bedrock/anthropic.claude-3")
```

### 💾 Smart Caching

Themis automatically caches results and resumes failed runs:

```python
# Run with caching
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    limit=1000,
    run_id="my-experiment",
    resume=True  # Skip already-evaluated samples
)
```

Cache invalidation is automatic when you change:
- Model parameters (temperature, max_tokens, etc.)
- Prompt template
- Evaluation metrics

---

## Examples

### Custom Dataset

```python
from themis import evaluate

# Your own data
dataset = [
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "What is 3+3?", "answer": "6"},
]

result = evaluate(
    dataset,
    model="gpt-4",
    prompt="Answer this math question: {prompt}",
    metrics=["exact_match"],
)

print(result.report)
```

### Advanced Configuration

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.7,
    max_tokens=512,
    num_samples=3,  # Sample 3 responses per prompt
    workers=8,  # Parallel execution
    storage=".cache/my-experiments",
    run_id="experiment-2024-01",
)
```

### Programmatic Comparison

```python
from themis.comparison.statistics import t_test, bootstrap_confidence_interval

# Model A scores
scores_a = [0.85, 0.87, 0.83, 0.90, 0.82]
# Model B scores  
scores_b = [0.78, 0.80, 0.79, 0.82, 0.77]

# Statistical test
result = bootstrap_confidence_interval(
    scores_a, scores_b,
    n_bootstrap=10000,
    confidence_level=0.95
)

print(f"Significant: {result.significant}")
print(f"CI: {result.confidence_interval}")
```

---

## Architecture

Themis is built on a clean, modular architecture:

```
┌─────────────────────────────────────────┐
│         themis.evaluate()                │  ← Simple API
│    (One-line evaluation interface)       │
└─────────────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
    ┌────▼─────┐     ┌────▼─────┐
    │ Presets  │     │Generation│
    │ System   │     │ Pipeline │
    └────┬─────┘     └────┬─────┘
         │                 │
    ┌────▼─────┐     ┌────▼─────┐
    │Benchmarks│     │Evaluation│
    │(6 built- │     │ Pipeline │
    │   in)    │     └────┬─────┘
    └──────────┘          │
                     ┌────▼─────┐
                     │ Storage  │
                     │  (V2)    │
                     └──────────┘
```

**Key Components:**

- **Presets**: Pre-configured benchmarks with prompts, metrics, and datasets
- **Generation**: Model inference with caching and resume
- **Evaluation**: Metric computation with smart cache invalidation
- **Storage**: Atomic writes, file locking, SQLite metadata
- **Comparison**: Statistical tests, win/loss matrices
- **Server**: REST API and WebSocket for web dashboard

---

## Documentation

- **[API Reference](docs/index.md)** - Detailed API documentation
- **[Examples](examples-simple/)** - Runnable code examples
- **[Extending Backends](docs/EXTENDING_BACKENDS.md)** - Custom storage and execution
- **[API Server](docs/API_SERVER.md)** - Web dashboard and REST API
- **[Comparison Engine](docs/COMPARISON.md)** - Statistical testing guide

---

## Advanced Usage

### Custom Backends

Implement custom storage or execution strategies:

```python
from themis.backends import StorageBackend, ExecutionBackend

class S3StorageBackend(StorageBackend):
    """Store results in AWS S3"""
    def save_generation_record(self, run_id, record):
        # Upload to S3
        pass
    # ... implement other methods

# Use custom backend
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage_backend=S3StorageBackend(bucket="my-bucket")
)
```

See [EXTENDING_BACKENDS.md](docs/EXTENDING_BACKENDS.md) for details.

### Distributed Execution

```python
from themis.backends import ExecutionBackend
import ray

class RayExecutionBackend(ExecutionBackend):
    """Distributed execution with Ray"""
    # ... implementation

result = evaluate(
    benchmark="math500",
    model="gpt-4",
    execution_backend=RayExecutionBackend(num_cpus=32)
)
```

### Monitoring & Observability

Connect to the WebSocket endpoint for real-time updates:

```python
import asyncio
import websockets
import json

async def monitor():
    async with websockets.connect("ws://localhost:8080/ws") as ws:
        await ws.send(json.dumps({"type": "subscribe", "run_id": "my-run"}))
        async for message in ws:
            print(json.loads(message))

asyncio.run(monitor())
```

---

## CLI Reference

### Evaluation

```bash
# Basic evaluation
themis eval <benchmark> --model <model> [options]

# Options:
#   --limit N              Evaluate first N samples
#   --temperature FLOAT    Sampling temperature (default: 0.0)
#   --max-tokens INT       Maximum tokens (default: 512)
#   --workers INT          Parallel workers (default: 4)
#   --run-id STR           Run identifier
#   --storage PATH         Storage directory
#   --resume               Resume from cache
#   --output FILE          Export results (.json, .csv, .html)
```

### Comparison

```bash
# Compare two or more runs
themis compare <run-id-1> <run-id-2> [run-id-3...] [options]

# Options:
#   --storage PATH         Storage directory
#   --test STR             Statistical test: t_test, bootstrap, permutation
#   --alpha FLOAT          Significance level (default: 0.05)
#   --output FILE          Export report (.json, .html, .md)
```

### Server

```bash
# Start API server
themis serve [options]

# Options:
#   --port INT             Port (default: 8080)
#   --host STR             Host (default: 127.0.0.1)
#   --storage PATH         Storage directory
#   --reload               Auto-reload (dev mode)
```

### List

```bash
# List available resources
themis list <what>

# Options:
#   runs         List all experiment runs
#   benchmarks   List available benchmarks
#   metrics      List available metrics
```

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/themis.git
cd themis

# Install with dev dependencies
uv pip install -e ".[dev,math,nlp,code,server]"

# Run tests
uv run pytest

# Run specific test
uv run pytest tests/comparison/test_statistics.py -v
```

### Project Structure

```
themis/
├── themis/
│   ├── api.py                  # Main evaluate() function
│   ├── presets/                # Benchmark presets
│   ├── generation/             # Model inference
│   ├── evaluation/             # Metrics & evaluation
│   ├── comparison/             # Statistical comparison
│   ├── backends/               # Pluggable backends
│   ├── server/                 # FastAPI server
│   └── cli/                    # CLI commands
├── tests/                      # Test suite
├── examples-simple/            # Minimal examples
├── docs/                       # Documentation
└── pyproject.toml             # Package configuration
```

### Running Examples

```bash
# Simple quickstart
uv run python examples-simple/01_quickstart.py

# Custom dataset
uv run python examples-simple/02_custom_dataset.py

# Comparison example
uv run python examples-simple/04_comparison.py

# API server example
uv run python examples-simple/05_api_server.py
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we'd love help:
- Additional benchmark presets
- New evaluation metrics
- Backend implementations (Ray, S3, etc.)
- Documentation improvements
- Bug reports and feature requests

---

## Citation

If you use Themis in your research, please cite:

```bibtex
@software{themis2024,
  title = {Themis: Modern LLM Evaluation Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/themis}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built on [LiteLLM](https://github.com/BerriAI/litellm) for provider support
- Inspired by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Statistical methods from established research practices

---

## Support

- **Documentation**: [docs/index.md](docs/index.md)
- **Examples**: [examples-simple/](examples-simple/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/themis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/themis/discussions)

---

**Made with ❤️ for the LLM research community**

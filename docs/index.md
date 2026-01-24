# Themis Documentation

Welcome to the Themis documentation! This guide will help you get the most out of Themis, from quick start to advanced usage.

## 📚 Documentation Structure

### Getting Started

- **[Quick Start](#quick-start)** - Get up and running in 5 minutes
- **[Installation](#installation)** - Install Themis with optional features
- **[Core Concepts](#core-concepts)** - Understand how Themis works

### User Guides

- **[Evaluation Guide](guides/evaluation.md)** - Run evaluations with built-in metrics
- **[API Server Guide](reference/api-server.md)** - Use the REST API and web dashboard
- **[Comparison Guide](#comparison-guide)** - Statistical comparison of runs
- **[LiteLLM / Providers](guides/providers.md)** - Use 100+ LLM providers

### Advanced Topics

- **[Extending Backends](customization/backends.md)** - Custom storage and execution
- **[Storage Architecture](guides/storage.md)** - Understanding Storage V2
- **[Cache Invalidation](guides/manual_verification_cache.md)** - Smart caching behavior
- **[Cost Tracking](guides/cost-tracking.md)** - Monitor API costs

### Reference

- **[Configuration Reference](guides/configuration.md)** - All configuration options
- **[Code Examples](examples/index.md)** - Detailed walkthroughs

---

## Quick Start

### Installation

```bash
# Basic installation
pip install themis-eval

# With all features
pip install themis-eval[math,nlp,code,server]

# Using uv (recommended)
uv pip install themis-eval[math,nlp,code,server]
```

### Your First Evaluation

```python
from themis import evaluate

# Evaluate GPT-4 on GSM8K
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    limit=100
)

print(f"Accuracy: {result.metrics['ExactMatch']:.2%}")
```

### Using the CLI

```bash
# Run evaluation
themis eval gsm8k --model gpt-4 --limit 100 --run-id my-experiment

# Compare two models
themis compare my-experiment-1 my-experiment-2

# Start web dashboard
themis serve
```

---

## Core Concepts

### 1. Evaluation API

Themis provides a simple `evaluate()` function that handles everything:

```python
from themis import evaluate

result = evaluate(
    benchmark="gsm8k",        # Built-in benchmark
    model="gpt-4",            # Any LiteLLM model
    temperature=0.0,          # Sampling parameters
    max_tokens=512,
    num_samples=1,            # Samples per prompt
    workers=8,                # Parallel execution
    storage=".cache",         # Results storage
    run_id="my-run",          # Identifier
    resume=True,              # Resume from cache
)
```

### 2. Benchmarks & Presets

Themis includes 6 built-in benchmarks:

- `demo` - Quick testing (10 samples)
- `gsm8k` - Grade school math (8.5K problems)
- `math500` - Advanced math (500 problems)
- `aime24` - AIME 2024 math competition (30 problems)
- `mmlu_pro` - MMLU-Pro knowledge benchmark
- `supergpqa` - SuperGPQA reasoning benchmark

Each preset includes:
- Pre-configured prompt template
- Appropriate evaluation metrics
- Dataset loader
- Reference answers

```python
# List available benchmarks
from themis.presets import list_benchmarks
print(list_benchmarks())  # ['demo', 'gsm8k', 'math500', ...]
```

### 3. Metrics

Themis supports multiple metric types:

**Math Metrics:**
- `ExactMatch` - String matching
- `MathVerify` - Symbolic & numeric verification

**NLP Metrics:**
- `BLEU` - N-gram overlap
- `ROUGE` - Recall-oriented matching
- `BERTScore` - Semantic similarity
- `METEOR` - Alignment-based scoring

**Code Metrics:**
- `PassAtK` - Pass rate at K samples
- `CodeBLEU` - Code similarity
- `ExecutionAccuracy` - Functional correctness

```python
# Use specific metrics
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "BLEU"],  # Specify metrics
)
```

### 4. Storage & Caching

Themis automatically caches results for:
- ✅ Resuming failed runs
- ✅ Avoiding duplicate API calls
- ✅ Reproducibility

Cache invalidation happens when you change:
- Model parameters (temperature, max_tokens, etc.)
- Prompt template
- Evaluation metrics

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="experiment-1",
    resume=True,  # Use cache if available
)
```

**Storage V2 features:**
- Atomic writes with file locking
- SQLite metadata
- Hierarchical organization
- Smart cache invalidation

See [Storage Architecture](STORAGE.md) for details.

### 5. Comparison Engine

Compare multiple runs with statistical tests:

```python
from themis.comparison import compare_runs

report = compare_runs(
    run_ids=["run-gpt4", "run-claude"],
    storage_path=".cache/experiments",
    statistical_test="bootstrap",
    alpha=0.05,
)

print(report.summary())
```

**Statistical tests:**
- `t_test` - Student's t-test (paired or independent)
- `bootstrap` - Bootstrap confidence intervals
- `permutation` - Permutation test
- `none` - No statistical testing

**Output includes:**
- Win/loss/tie matrices
- P-values and significance
- Effect sizes (Cohen's d)
- Confidence intervals

See [Comparison Guide](#comparison-guide) below.

### 6. API Server & Dashboard

Start a web server to view and compare results:

```bash
themis serve --port 8080
```

Then open:
- Dashboard: `http://localhost:8080/dashboard`
- API docs: `http://localhost:8080/docs`

**API endpoints:**
- `GET /api/runs` - List all runs
- `GET /api/runs/{run_id}` - Get run details
- `POST /api/compare` - Compare runs
- `WS /ws` - WebSocket for real-time updates

See [API Server Guide](API_SERVER.md) for details.

---

## Comparison Guide

### Basic Comparison

Compare two runs:

```python
from themis.comparison import compare_runs

report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
)

# Print summary
print(report.summary(include_details=True))

# Get winner
print(f"Best run: {report.overall_best_run}")
```

### Statistical Tests

Choose from multiple statistical tests:

```python
from themis.comparison import compare_runs
from themis.comparison.statistics import StatisticalTest

# Bootstrap confidence intervals (default)
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.BOOTSTRAP,
    alpha=0.05,  # 95% confidence
)

# T-test
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.T_TEST,
)

# Permutation test
report = compare_runs(
    run_ids=["run-1", "run-2"],
    storage_path=".cache/experiments",
    statistical_test=StatisticalTest.PERMUTATION,
    alpha=0.01,  # 99% confidence
)
```

### Win/Loss Matrices

Compare 3+ runs:

```python
report = compare_runs(
    run_ids=["run-1", "run-2", "run-3"],
    storage_path=".cache/experiments",
)

# Get win/loss matrix for a metric
matrix = report.win_loss_matrices["ExactMatch"]
print(matrix.to_table())

# Rankings
for run_id, wins, losses, ties in matrix.rank_runs():
    print(f"{run_id}: {wins}W-{losses}L-{ties}T")
```

### Export Results

```python
import json
from pathlib import Path

# Export to JSON
output = Path("comparison.json")
output.write_text(json.dumps(report.to_dict(), indent=2))

# CLI export
# themis compare run-1 run-2 --output comparison.html
# themis compare run-1 run-2 --output comparison.md
```

### CLI Comparison

```bash
# Basic comparison
themis compare run-1 run-2

# With specific test
themis compare run-1 run-2 --test bootstrap --alpha 0.01

# Compare specific metrics
themis compare run-1 run-2 --metrics ExactMatch BLEU

# Export to HTML
themis compare run-1 run-2 --output comparison.html
```

---

## Custom Datasets

Use your own data:

```python
from themis import evaluate

# Your dataset
dataset = [
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "What is the capital of France?", "answer": "Paris"},
]

result = evaluate(
    dataset,
    model="gpt-4",
    prompt="Question: {prompt}\nAnswer:",
    metrics=["ExactMatch"],
)
```

**Dataset format:**
- List of dictionaries
- Each dict has at minimum: `prompt` (or `question`), `answer` (or `reference`)
- Optional: `id`, custom fields for prompt template

---

## Model Configuration

### Sampling Parameters

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.7,       # Sampling temperature
    max_tokens=1024,       # Max response length
    top_p=0.95,            # Nucleus sampling
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
```

### Multiple Samples

Generate multiple responses per prompt:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    num_samples=5,  # Sample 5 responses per prompt
)

# Use with Pass@K metric for code generation
result = evaluate(
    benchmark="my-code-dataset",
    model="gpt-4",
    num_samples=10,
    metrics=["PassAtK"],
)
```

### Parallel Execution

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    workers=16,  # Use 16 parallel workers
)
```

---

## Provider Support

Themis works with 100+ providers via [LiteLLM](LITELLM.md):

### OpenAI

```python
evaluate(benchmark="gsm8k", model="gpt-4")
evaluate(benchmark="gsm8k", model="gpt-3.5-turbo")
```

### Anthropic

```python
evaluate(benchmark="gsm8k", model="claude-3-opus-20240229")
evaluate(benchmark="gsm8k", model="claude-3-sonnet-20240229")
```

### Azure OpenAI

```python
evaluate(benchmark="gsm8k", model="azure/gpt-4")
```

### AWS Bedrock

```python
evaluate(benchmark="gsm8k", model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
```

### Local Models

```python
# Ollama
evaluate(benchmark="gsm8k", model="ollama/llama3")

# vLLM
evaluate(benchmark="gsm8k", model="openai/meta-llama/Llama-3-8b")
```

See [LiteLLM Integration](LITELLM.md) for full provider list and configuration.

---

## Extending Themis

### Custom Metrics

```python
from themis.evaluation.metrics import Metric, MetricScore

class MyMetric(Metric):
    """Custom metric implementation."""
    
    @property
    def name(self) -> str:
        return "MyMetric"
    
    def evaluate(self, response: str, reference: str) -> MetricScore:
        # Your evaluation logic
        score = my_evaluation_function(response, reference)
        return MetricScore(value=score, metadata={})

# Use in evaluation
result = evaluate(
    dataset=my_dataset,
    model="gpt-4",
    metrics=[MyMetric()],
)
```

### Custom Storage Backend

```python
from themis.backends import StorageBackend

class S3StorageBackend(StorageBackend):
    """Store results in AWS S3."""
    
    def save_generation_record(self, run_id, record):
        # Upload to S3
        pass
    
    # Implement other methods...

# Use custom backend
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage_backend=S3StorageBackend(bucket="my-bucket"),
)
```

See [Extending Backends](EXTENDING_BACKENDS.md) for complete guide.

### Custom Execution Backend

```python
from themis.backends import ExecutionBackend

class RayExecutionBackend(ExecutionBackend):
    """Distributed execution with Ray."""
    
    def map(self, func, items, **kwargs):
        # Distribute with Ray
        pass
    
    # Implement other methods...

result = evaluate(
    benchmark="math500",
    model="gpt-4",
    execution_backend=RayExecutionBackend(num_cpus=32),
)
```

---

## Best Practices

### 1. Use Run IDs

Always specify run IDs for reproducibility:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="experiment-2024-01-15-v1",
)
```

### 2. Enable Caching

Let Themis resume failed runs:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=True,  # Resume from cache
)
```

### 3. Use Appropriate Limits

Start with small limits for testing:

```python
# Test with 10 samples
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)

# Full evaluation once working
result = evaluate(benchmark="gsm8k", model="gpt-4")  # All samples
```

### 4. Monitor Costs

Track API costs:

```python
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

print(f"Total cost: ${result.cost:.2f}")
```

See [Cost Tracking](COST_TRACKING.md) for details.

### 5. Compare Systematically

Run multiple experiments and compare:

```python
# Experiment 1: GPT-4 with temp=0
evaluate(benchmark="gsm8k", model="gpt-4", temperature=0.0, run_id="gpt4-temp0")

# Experiment 2: GPT-4 with temp=0.7
evaluate(benchmark="gsm8k", model="gpt-4", temperature=0.7, run_id="gpt4-temp07")

# Compare
from themis.comparison import compare_runs
report = compare_runs(["gpt4-temp0", "gpt4-temp07"], storage_path=".cache/experiments")
print(report.summary())
```

---

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Install missing optional dependencies
pip install themis-eval[math,nlp,code,server]
```

**API key errors:**
```bash
# Set environment variables
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
```

**Storage errors:**
```python
# Clear cache and restart
import shutil
shutil.rmtree(".cache/experiments")
```

**Server won't start:**
```bash
# Check if port is in use
lsof -i :8080

# Use different port
themis serve --port 8081
```

---

## Need Help?

- **Examples**: Check [examples-simple/](../examples-simple/) for working code
- **API Reference**: See individual doc files above
- **Issues**: [GitHub Issues](https://github.com/yourusername/themis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/themis/discussions)

---

**Happy evaluating! 🎯**

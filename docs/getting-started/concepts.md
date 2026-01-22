# Core Concepts

Understanding the key concepts in Themis will help you use it effectively.

## Architecture Overview

Themis is built on a layered architecture:

```
┌─────────────────────────────────────┐
│      themis.evaluate()               │  Simple API
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌────▼─────┐
│Presets │          │Generation│
│System  │          │ Pipeline │
└───┬────┘          └────┬─────┘
    │                    │
┌───▼────┐          ┌────▼─────┐
│Benchmrk│          │Evaluation│
│Dataset │          │ Pipeline │
└────────┘          └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ Storage  │
                    │   V2     │
                    └──────────┘
```

## Key Concepts

### 1. Evaluation

**Evaluation** is the process of testing an LLM on a dataset and computing metrics.

```python
result = evaluate(
    benchmark="gsm8k",    # What to evaluate on
    model="gpt-4",        # What model to test
    metrics=["ExactMatch"],  # How to measure
)
```

**Three steps:**
1. **Generation**: LLM produces responses
2. **Evaluation**: Metrics compare responses to references
3. **Reporting**: Results are aggregated and stored

### 2. Benchmarks & Presets

**Benchmarks** are standardized evaluation datasets with:
- Prompts/questions
- Reference answers
- Evaluation metrics
- Prompt templates

**Presets** package these into ready-to-use configurations:

```python
# Built-in benchmarks
evaluate(benchmark="gsm8k", model="gpt-4")      # Math reasoning
evaluate(benchmark="mmlu_pro", model="gpt-4")   # Knowledge
evaluate(benchmark="aime24", model="gpt-4")     # Competition math
```

Each preset includes:
- Dataset loader
- Default prompt template
- Appropriate metrics
- Reference field mapping

### 3. Metrics

**Metrics** quantify how well an LLM performs:

**Math Metrics:**
- `ExactMatch` - Exact string matching
- `MathVerify` - Symbolic & numeric equivalence

**NLP Metrics:**
- `BLEU` - N-gram precision
- `ROUGE` - N-gram recall
- `BERTScore` - Semantic similarity
- `METEOR` - Alignment-based

**Code Metrics:**
- `PassAtK` - Pass rate for K samples
- `CodeBLEU` - Code-specific BLEU
- `ExecutionAccuracy` - Functional correctness

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "BLEU", "MathVerify"],
)

print(result.metrics)
# {'ExactMatch': 0.85, 'BLEU': 0.72, 'MathVerify': 0.87}
```

### 4. Storage & Caching

Themis automatically caches results to enable:
- **Resuming failed runs**
- **Avoiding duplicate API calls**
- **Reproducibility**

**Storage V2 features:**
- Atomic writes with file locking
- SQLite metadata
- Hierarchical organization
- Smart cache invalidation

```python
# First run - generates responses
result1 = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
)

# Second run - uses cache (instant!)
result2 = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=True,
)
```

**Cache invalidation** happens when you change:
- Model parameters (temperature, max_tokens, etc.)
- Prompt template
- Evaluation metrics

### 5. Comparison

**Comparison** analyzes differences between runs with statistical rigor:

```python
from themis.comparison import compare_runs

# Compare two models
report = compare_runs(
    run_ids=["gpt4-run", "claude-run"],
    storage_path=".cache/experiments",
)

# Statistical significance
for result in report.pairwise_results:
    if result.is_significant():
        print(f"✓ {result.winner} wins")
```

**Statistical tests:**
- T-test (parametric)
- Bootstrap (non-parametric)
- Permutation test (exact)

**Outputs:**
- P-values (significance)
- Effect sizes (magnitude)
- Confidence intervals
- Win/loss matrices

### 6. Providers

Themis uses [LiteLLM](https://docs.litellm.ai/) for provider support:

```python
# 100+ providers supported
evaluate(benchmark="gsm8k", model="gpt-4")                    # OpenAI
evaluate(benchmark="gsm8k", model="claude-3-opus")            # Anthropic
evaluate(benchmark="gsm8k", model="azure/gpt-4")              # Azure
evaluate(benchmark="gsm8k", model="bedrock/claude-3")         # AWS
evaluate(benchmark="gsm8k", model="ollama/llama3")            # Local
```

API keys are read from environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### 7. Backends

**Backends** are pluggable interfaces for:

**Storage Backends:**
- Where results are stored
- Default: Local file system
- Custom: S3, PostgreSQL, Redis, etc.

**Execution Backends:**
- How tasks are executed
- Default: Multi-threaded (`ThreadPoolExecutor`)
- Custom: Ray, Dask, async, etc.

```python
from themis.backends import LocalExecutionBackend

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    execution_backend=LocalExecutionBackend(max_workers=16),
)
```

See [Extending Backends](../EXTENDING_BACKENDS.md) for custom implementations.

## Data Flow

Understanding how data flows through Themis:

```
1. Load Dataset
   ↓
2. Apply Prompt Template
   ↓
3. Generate Responses (with caching)
   ↓
4. Extract Answers
   ↓
5. Compute Metrics
   ↓
6. Aggregate & Report
   ↓
7. Store Results
```

**With caching:**
```
1. Check cache for existing generation
2. If exists → Skip to step 4
3. If not exists → Generate (step 3)
4. Check cache for existing evaluation
5. If exists → Skip to step 7
6. If not exists → Evaluate (steps 4-6)
7. Return results
```

## Key Design Principles

### 1. Simplicity First

The API should be simple for common cases:

```python
# Simple
result = evaluate(benchmark="gsm8k", model="gpt-4")

# Advanced (when needed)
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.7,
    num_samples=5,
    workers=16,
    storage=custom_backend,
)
```

### 2. Batteries Included

Built-in benchmarks and metrics for common tasks:
- No configuration files needed
- Works out of the box
- Sensible defaults

### 3. Extensible

Clean interfaces for custom implementations:
- Custom metrics
- Custom storage backends
- Custom execution backends

### 4. Reproducible

Everything is cached and resumable:
- Deterministic by default (temperature=0)
- Run IDs for tracking
- Smart cache invalidation

### 5. Type Safe

Full type annotations:
```python
from themis import evaluate
from themis.core.entities import ExperimentReport

result: ExperimentReport = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
)
```

## Common Workflows

### Workflow 1: Single Model Evaluation

```python
# 1. Run evaluation
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

# 2. Check results
print(result.metrics)

# 3. Export if needed
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.json",
)
```

### Workflow 2: Model Comparison

```python
# 1. Run multiple models
for model in ["gpt-4", "claude-3", "gemini-pro"]:
    evaluate(
        benchmark="gsm8k",
        model=model,
        limit=100,
        run_id=f"gsm8k-{model}",
    )

# 2. Compare statistically
from themis.comparison import compare_runs

report = compare_runs(
    run_ids=["gsm8k-gpt-4", "gsm8k-claude-3", "gsm8k-gemini-pro"],
    storage_path=".cache/experiments",
)

# 3. View results
print(report.summary())
```

### Workflow 3: Prompt Engineering

```python
prompts = {
    "zero-shot": "Solve: {prompt}",
    "cot": "Let's solve step by step: {prompt}",
    "few-shot": "Examples:\nQ: 1+1\nA: 2\n\nQ: {prompt}\nA:",
}

for name, template in prompts.items():
    evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        prompt=template,
        run_id=f"prompt-{name}",
    )

# Compare prompts
report = compare_runs(
    [f"prompt-{name}" for name in prompts.keys()],
    storage_path=".cache/experiments",
)
```

### Workflow 4: Hyperparameter Search

```python
for temp in [0.0, 0.3, 0.5, 0.7, 1.0]:
    evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        temperature=temp,
        limit=100,
        run_id=f"temp-{temp}",
    )

# Find best temperature
report = compare_runs(
    [f"temp-{t}" for t in [0.0, 0.3, 0.5, 0.7, 1.0]],
    storage_path=".cache/experiments",
)

print(f"Best: {report.overall_best_run}")
```

## Next Steps

- [Evaluation Guide](../guides/evaluation.md) - Deep dive into evaluation
- [Comparison Guide](../COMPARISON.md) - Statistical comparison
- [API Reference](../api/overview.md) - Complete API docs
- [Examples](../tutorials/examples.md) - Working code examples

## Questions?

If anything is unclear:
- Check the [documentation](../index.md)
- Ask on [GitHub Discussions](https://github.com/pittawat2542/themis/discussions)
- Open an [issue](https://github.com/pittawat2542/themis/issues)

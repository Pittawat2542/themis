# Evaluation Guide

Complete guide to running evaluations with Themis.

## Table of Contents

- [Basic Evaluation](#basic-evaluation)
- [Built-in Benchmarks](#built-in-benchmarks)
- [Custom Datasets](#custom-datasets)
- [Model Configuration](#model-configuration)
- [Metrics Selection](#metrics-selection)
- [Caching and Resume](#caching-and-resume)
- [Parallel Execution](#parallel-execution)
- [Export Results](#export-results)

---

## Basic Evaluation

The simplest evaluation:

```python
from themis import evaluate

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
)

print(f"Accuracy: {result.metrics['ExactMatch']:.2%}")
```

This evaluates the entire GSM8K benchmark using GPT-4 with default settings.

---

## Built-in Benchmarks

Themis includes 6 benchmarks:

### Math Benchmarks

**GSM8K** - Grade School Math (8.5K problems)
```python
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

**MATH500** - Advanced Math (500 problems)
```python
result = evaluate(benchmark="math500", model="gpt-4")
```

**AIME24** - Math Competition (30 problems)
```python
result = evaluate(benchmark="aime24", model="gpt-4")
```

### Knowledge Benchmarks

**MMLU-Pro** - General Knowledge
```python
result = evaluate(benchmark="mmlu_pro", model="gpt-4", limit=1000)
```

**SuperGPQA** - Advanced Reasoning
```python
result = evaluate(benchmark="supergpqa", model="gpt-4")
```

### Testing

**Demo** - Quick Testing (10 samples, fake model)
```python
result = evaluate(benchmark="demo", model="fake-math-llm")
```

### List All Benchmarks

```python
from themis.presets import list_benchmarks

benchmarks = list_benchmarks()
for name in benchmarks:
    print(name)
```

Or via CLI:
```bash
themis list benchmarks
```

---

## Custom Datasets

### Basic Custom Dataset

```python
dataset = [
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "What is 5-3?", "answer": "2"},
]

result = evaluate(
    dataset,
    model="gpt-4",
    prompt="Question: {prompt}\nAnswer:",
)
```

### Required Fields

Each dataset item must have:
- **Input**: `prompt` or `question` - The input to the model
- **Output**: `answer` or `reference` - The expected output
- **Optional**: `id` - Unique identifier (auto-generated if missing)

### Custom Prompt Templates

Use format strings to customize prompts:

```python
dataset = [
    {"question": "What is the capital of France?", "answer": "Paris"},
]

result = evaluate(
    dataset,
    model="gpt-4",
    prompt="Answer this question:\n\nQ: {question}\nA:",
)
```

### With Additional Fields

Include extra fields for complex prompts:

```python
dataset = [
    {
        "question": "What is 2+2?",
        "context": "Basic arithmetic",
        "answer": "4",
    }
]

result = evaluate(
    dataset,
    model="gpt-4",
    prompt="Context: {context}\nQuestion: {question}\nAnswer:",
)
```

---

## Model Configuration

### Sampling Parameters

Control how the model generates responses:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    
    temperature=0.7,         # Randomness (0=deterministic, 1+=creative)
    max_tokens=1024,         # Maximum response length
    top_p=0.95,              # Nucleus sampling
    frequency_penalty=0.2,   # Reduce repetition
    presence_penalty=0.0,    # Encourage new topics
)
```

### Multiple Samples

Generate several responses per prompt:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    num_samples=5,  # Generate 5 responses per question
)
```

Useful for:
- Pass@K metrics (code generation)
- Measuring consistency
- Majority voting

### Provider-Specific Options

Pass provider-specific parameters:

```python
# OpenAI-specific
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    response_format={"type": "json_object"},
    seed=42,
)

# Anthropic-specific
result = evaluate(
    benchmark="gsm8k",
    model="claude-3-opus",
    max_tokens_to_sample=2048,
)
```

---

## Metrics Selection

### Using Built-in Metrics

Specify metrics explicitly:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "MathVerify", "BLEU"],
)
```

### Available Metrics

List available metrics:

```bash
themis list metrics
```

Output:
```
Math:
  - ExactMatch
  - MathVerify
NLP:
  - BLEU
  - ROUGE
  - BERTScore
  - METEOR
Code:
  - PassAtK
  - CodeBLEU
  - ExecutionAccuracy
```

### Custom Metrics

Implement your own metric:

```python
from themis.evaluation.metrics import Metric, MetricScore

class MyMetric(Metric):
    @property
    def name(self) -> str:
        return "MyMetric"
    
    def evaluate(self, response: str, reference: str) -> MetricScore:
        score = len(response) / len(reference)  # Example: length ratio
        return MetricScore(value=score, metadata={})

# Use in evaluation
result = evaluate(
    dataset=my_dataset,
    model="gpt-4",
    metrics=[MyMetric()],
)
```

---

## Caching and Resume

### Enable Caching

Caching is enabled by default:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=True,  # Default: True
)
```

### Disable Caching

Force re-evaluation:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=False,  # Ignore cache
)
```

### Cache Behavior

Cache is invalidated when you change:
- Model name or version
- Sampling parameters (temperature, max_tokens, etc.)
- Prompt template
- Evaluation metrics

### Storage Location

Specify where results are stored:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage="~/my-experiments",  # Custom location
)
```

Default storage: `.cache/experiments`

---

## Parallel Execution

### Multi-threaded Execution

Use multiple workers for faster evaluation:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    workers=16,  # Use 16 parallel threads
)
```

**Recommendations:**
- CPU-bound: `workers = num_cpus`
- I/O-bound (API calls): `workers = 8-32`
- Rate limits: Reduce workers to avoid throttling

### Sequential Execution

For debugging, use single-threaded execution:

```python
from themis.backends import SequentialExecutionBackend

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    execution_backend=SequentialExecutionBackend(),
)
```

---

## Export Results

### Export Formats

Export results to various formats:

```python
# JSON
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.json",
)

# CSV
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.csv",
)

# HTML
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.html",
)
```

### Programmatic Export

```python
from themis.experiment.export import export_json, export_csv, export_html
from pathlib import Path

result = evaluate(benchmark="gsm8k", model="gpt-4")

# Export manually
export_json(result, Path("results.json"))
export_csv(result, Path("results.csv"))
export_html(result, Path("results.html"))
```

---

## Best Practices

### 1. Start Small

Always test with small limits first:

```python
# Test run
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)

# Full run once working
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### 2. Use Run IDs

Always specify meaningful run IDs:

```python
# Good
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="gsm8k-gpt4-baseline-2024-01-15",
)

# Bad (auto-generated, hard to track)
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### 3. Monitor Costs

Check costs after evaluation:

```python
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)
print(f"Cost: ${result.cost:.2f}")
```

### 4. Use Appropriate Workers

Balance speed vs API limits:

```python
# Fast but may hit rate limits
result = evaluate(benchmark="gsm8k", model="gpt-4", workers=32)

# Safer for rate limits
result = evaluate(benchmark="gsm8k", model="gpt-4", workers=8)
```

### 5. Test with Fake Model

Use `fake-math-llm` for testing (no API key needed):

```python
result = evaluate(
    benchmark="demo",
    model="fake-math-llm",
    limit=10,
)
```

---

## Troubleshooting

### Evaluation Hangs

If evaluation hangs:
1. Reduce `workers` parameter
2. Check API key is set
3. Verify model name is correct
4. Try with `fake-math-llm` to isolate issue

### Cache Issues

If cache seems stale:
```python
# Force re-evaluation
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    resume=False,
)
```

Or clear cache:
```bash
rm -rf .cache/experiments
```

### Memory Issues

For large datasets:
1. Use `limit` parameter
2. Reduce `workers`
3. Increase system memory
4. Use streaming (custom backend)

---

## Next Steps

- [Comparison Guide](../COMPARISON.md) - Compare multiple runs
- [API Reference](../api/evaluate.md) - Complete API documentation
- [Examples](../tutorials/examples.md) - Working code examples
- [Custom Metrics](../EVALUATION.md) - Implement custom metrics

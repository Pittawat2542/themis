# Configuration Guide

Complete guide to configuring Themis evaluations.

## Overview

Themis configuration is done through function parameters—no config files required for simple use cases!

```python
from themis import evaluate

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.7,    # ← Configuration via parameters
    workers=8,
)
```

---

## Configuration Parameters

### Dataset Configuration

**`benchmark_or_dataset`**

Either a benchmark name or custom dataset:

```python
# Built-in benchmark
result = evaluate(benchmark="gsm8k", model="gpt-4")

# Custom dataset
dataset = [{"prompt": "...", "answer": "..."}]
result = evaluate(dataset, model="gpt-4")
```

**`limit`** - Number of samples to evaluate:

```python
# Evaluate first 100 samples
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

# Evaluate all samples
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

**`prompt`** - Custom prompt template:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    prompt="Question: {prompt}\nThink step by step.\nAnswer:",
)
```

---

### Model Configuration

**`model`** - Model identifier:

```python
# OpenAI
result = evaluate(benchmark="gsm8k", model="gpt-4")
result = evaluate(benchmark="gsm8k", model="gpt-3.5-turbo")

# Anthropic
result = evaluate(benchmark="gsm8k", model="claude-3-opus-20240229")

# Azure
result = evaluate(benchmark="gsm8k", model="azure/gpt-4")

# Local
result = evaluate(benchmark="gsm8k", model="ollama/llama3")
```

**`temperature`** - Sampling temperature (0.0-2.0):

```python
# Deterministic (recommended for evaluation)
result = evaluate(benchmark="gsm8k", model="gpt-4", temperature=0.0)

# Creative
result = evaluate(benchmark="gsm8k", model="gpt-4", temperature=1.0)
```

**`max_tokens`** - Maximum response length:

```python
# Short responses
result = evaluate(benchmark="gsm8k", model="gpt-4", max_tokens=256)

# Long responses
result = evaluate(benchmark="gsm8k", model="gpt-4", max_tokens=2048)
```

**`top_p`** - Nucleus sampling:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    top_p=0.95,  # Consider top 95% probability mass
)
```

**Provider-specific parameters:**

```python
# OpenAI-specific
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    frequency_penalty=0.2,
    presence_penalty=0.0,
    seed=42,
    response_format={"type": "json_object"},
)

# Anthropic-specific
result = evaluate(
    benchmark="gsm8k",
    model="claude-3-opus",
    max_tokens_to_sample=2048,
)
```

---

### Execution Configuration

**`num_samples`** - Responses per prompt:

```python
# Single response (default)
result = evaluate(benchmark="gsm8k", model="gpt-4", num_samples=1)

# Multiple samples (for Pass@K, ensembling)
result = evaluate(benchmark="gsm8k", model="gpt-4", num_samples=10)
```

**`workers`** - Parallel workers:

```python
# Few workers (safer for rate limits)
result = evaluate(benchmark="gsm8k", model="gpt-4", workers=4)

# Many workers (faster)
result = evaluate(benchmark="gsm8k", model="gpt-4", workers=16)
```

**`distributed`** - Distributed execution (future):

```python
# Single machine (default)
result = evaluate(benchmark="gsm8k", model="gpt-4", distributed=False)

# Distributed (when implemented)
result = evaluate(benchmark="gsm8k", model="gpt-4", distributed=True)
```

---

### Storage Configuration

**`storage`** - Storage directory path:

```python
# Default location
result = evaluate(benchmark="gsm8k", model="gpt-4")  # Uses .cache/experiments

# Custom location
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage="~/my-experiments",
)
```

**`run_id`** - Unique run identifier:

```python
# Auto-generated (timestamp)
result = evaluate(benchmark="gsm8k", model="gpt-4")

# Custom run ID
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="gpt4-baseline-2024-01-15",
)
```

**`resume`** - Resume from cache:

```python
# Resume from cache (default)
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=True,
)

# Force re-evaluation
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=False,
)
```

---

### Metrics Configuration

**`metrics`** - List of metrics to compute:

```python
# Use benchmark defaults
result = evaluate(benchmark="gsm8k", model="gpt-4")

# Custom metrics
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "BLEU", "MathVerify"],
)
```

---

### Output Configuration

**`output`** - Export results:

```python
# JSON export
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.json",
)

# CSV export
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.csv",
)

# HTML export
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.html",
)
```

**`on_result`** - Result callback:

```python
def log_result(record):
    print(f"✓ {record.id}: {record.response[:50]}...")

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    on_result=log_result,
)
```

---

## Environment Variables

### API Keys

Set provider API keys:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://..."
export AZURE_API_VERSION="2023-05-15"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION_NAME="us-east-1"
```

### Themis Settings

```bash
# Default storage location
export THEMIS_STORAGE="~/.themis/experiments"

# Log level
export THEMIS_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Disable caching
export THEMIS_DISABLE_CACHE="false"
```

---

## Configuration Files (Optional)

For complex experiments, you can use configuration files:

### JSON Configuration

```json
{
  "benchmark": "gsm8k",
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 512,
  "workers": 8,
  "metrics": ["ExactMatch", "MathVerify"],
  "storage": ".cache/experiments",
  "run_id": "my-experiment"
}
```

Load and use:

```python
import json
from themis import evaluate

with open("config.json") as f:
    config = json.load(f)

result = evaluate(**config)
```

### YAML Configuration

```yaml
benchmark: gsm8k
model: gpt-4
temperature: 0.7
max_tokens: 512
workers: 8
metrics:
  - ExactMatch
  - MathVerify
```

Load with PyYAML:

```python
import yaml
from themis import evaluate

with open("config.yaml") as f:
    config = yaml.safe_load(f)

result = evaluate(**config)
```

---

## Best Practices

### 1. Deterministic by Default

Use `temperature=0` for reproducible results:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    temperature=0.0,  # Deterministic
    seed=42,          # Additional determinism (provider-specific)
)
```

### 2. Meaningful Run IDs

Use descriptive run IDs:

```python
# Good
run_id = "gsm8k-gpt4-temp07-cot-2024-01-15"

# Bad
run_id = "run123"
```

Pattern: `{benchmark}-{model}-{variant}-{date}`

### 3. Start Small, Scale Up

```python
# 1. Test with 10 samples
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)

# 2. Verify it works, then scale
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

# 3. Full evaluation
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### 4. Monitor Costs

```python
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)

# Check cost before scaling up
print(f"Cost for 100 samples: ${result.cost:.2f}")
print(f"Estimated full cost: ${result.cost * 85:.2f}")  # GSM8K has 8500 samples
```

### 5. Use Appropriate Workers

```python
# Conservative (respects rate limits)
result = evaluate(benchmark="gsm8k", model="gpt-4", workers=4)

# Aggressive (faster but may hit limits)
result = evaluate(benchmark="gsm8k", model="gpt-4", workers=32)
```

---

## Presets vs Custom

### When to Use Presets

Use built-in benchmarks when:
- ✅ Evaluating on standard benchmarks
- ✅ Comparing to existing work
- ✅ You want sensible defaults

```python
result = evaluate(benchmark="gsm8k", model="gpt-4")
```

### When to Use Custom

Use custom datasets when:
- ✅ Evaluating on proprietary data
- ✅ Testing domain-specific tasks
- ✅ Experimenting with new formats

```python
result = evaluate(my_dataset, model="gpt-4", prompt="...")
```

---

## Advanced Configuration

### Custom Backends

```python
from themis.backends import StorageBackend, ExecutionBackend

# Custom storage
class S3Storage(StorageBackend):
    # Implementation
    pass

# Custom execution
class RayExecution(ExecutionBackend):
    # Implementation
    pass

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage_backend=S3Storage(bucket="my-bucket"),
    execution_backend=RayExecution(num_cpus=32),
)
```

See [Extending Backends](../EXTENDING_BACKENDS.md) for details.

---

## Next Steps

- [Evaluation Guide](evaluation.md) - Detailed evaluation guide
- [API Reference](../api/evaluate.md) - Complete parameter reference
- [Examples](../tutorials/examples.md) - Working code examples

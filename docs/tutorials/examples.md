# Code Examples

Working code examples demonstrating Themis features.

## Quick Examples (`examples-simple/`)

Minimal, focused examples for learning Themis:

### 01. Quick Start (`01_quickstart.py`)

Basic evaluation with built-in benchmark:

```python
from themis import evaluate

result = evaluate(
    benchmark="demo",
    model="fake-math-llm",
    limit=5,
)

print(f"Accuracy: {result.metrics['ExactMatch']:.2%}")
```

**What it demonstrates:**
- Using `evaluate()` function
- Built-in benchmarks
- Basic result access

**Run it:**
```bash
python examples-simple/01_quickstart.py
```

---

### 02. Custom Dataset (`02_custom_dataset.py`)

Evaluate on your own data:

```python
from themis import evaluate

dataset = [
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "What is 5-3?", "answer": "2"},
]

result = evaluate(
    dataset,
    model="fake-math-llm",
    prompt="Question: {prompt}\nAnswer:",
)
```

**What it demonstrates:**
- Custom dataset format
- Prompt templates
- Metric selection

**Run it:**
```bash
python examples-simple/02_custom_dataset.py
```

---

### 03. Distributed Execution (`03_distributed.py`)

Placeholder for distributed execution (when implemented):

```python
# Future feature
result = evaluate(
    benchmark="math500",
    model="gpt-4",
    distributed=True,
)
```

---

### 04. Comparison (`04_comparison.py`)

Compare multiple runs statistically:

```python
from themis.comparison import compare_runs
from themis.comparison.statistics import t_test, bootstrap_confidence_interval

# Simulated scores
model_a_scores = [0.8, 0.85, 0.82, 0.88, 0.79]
model_b_scores = [0.75, 0.78, 0.77, 0.80, 0.74]

# T-test
result = t_test(model_a_scores, model_b_scores, paired=True)
print(result)

# Bootstrap
result = bootstrap_confidence_interval(
    model_a_scores, model_b_scores, n_bootstrap=1000, seed=42
)
print(result)
```

**What it demonstrates:**
- Statistical comparison
- Multiple test types
- Result interpretation

**Run it:**
```bash
python examples-simple/04_comparison.py
```

---

### 05. API Server (`05_api_server.py`)

Using the REST API and WebSocket:

```python
import requests

# List all runs
response = requests.get("http://localhost:8080/api/runs")
runs = response.json()

# Compare runs
response = requests.post(
    "http://localhost:8080/api/compare",
    json={"run_ids": ["run-1", "run-2"]}
)
comparison = response.json()
```

**What it demonstrates:**
- REST API usage
- WebSocket connections
- Programmatic access

**Run it:**
```bash
# Start server first
themis serve

# Then run example
python examples-simple/05_api_server.py
```

---

## Advanced Examples (`examples/`)

More complex examples showing advanced features:

### Getting Started (`examples/getting_started/`)

Basic pipeline with explicit builder pattern:

```python
from themis.experiment import builder

experiment = (
    builder.ExperimentBuilder()
    .with_dataset(...)
    .with_model(...)
    .with_metrics(...)
    .build()
)

report = experiment.run()
```

**Note**: This uses the old API. New code should use `themis.evaluate()`.

---

### Config File (`examples/config_file/`)

Configuration-driven experiments:

```json
{
  "dataset": {
    "name": "gsm8k",
    "limit": 100
  },
  "model": {
    "name": "gpt-4",
    "temperature": 0.7
  },
  "metrics": ["ExactMatch", "MathVerify"]
}
```

**Note**: Config files are optional in v2. Direct API is simpler.

---

### Prompt Engineering (`examples/prompt_engineering/`)

Systematic prompt exploration:

```python
prompts = [
    "Solve: {prompt}",
    "Let's think step by step: {prompt}",
    "Write Python to solve: {prompt}",
]

for i, prompt in enumerate(prompts):
    result = evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        prompt=prompt,
        run_id=f"prompt-{i}",
    )
    print(f"Prompt {i}: {result.metrics['ExactMatch']:.2%}")
```

---

## Running Examples

### From Command Line

```bash
# Simple examples
python examples-simple/01_quickstart.py
python examples-simple/02_custom_dataset.py
python examples-simple/04_comparison.py

# Advanced examples (old API)
python -m examples.getting_started.cli run
python -m examples.prompt_engineering.cli run
```

### In Jupyter

```python
# You can copy example code into Jupyter cells
from themis import evaluate

result = evaluate(
    benchmark="demo",
    model="fake-math-llm",
)

print(result.metrics)
```

---

## Example Patterns

### Pattern 1: Hyperparameter Sweep

```python
from themis import evaluate

temperatures = [0.0, 0.3, 0.5, 0.7, 1.0]

for temp in temperatures:
    result = evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        temperature=temp,
        limit=100,
        run_id=f"gsm8k-temp-{temp}",
    )
    print(f"Temp {temp}: {result.metrics['ExactMatch']:.2%}")
```

### Pattern 2: Model Comparison

```python
models = {
    "gpt-4": "gpt-4",
    "gpt-3.5": "gpt-3.5-turbo",
    "claude-3": "claude-3-opus-20240229",
}

for name, model in models.items():
    result = evaluate(
        benchmark="gsm8k",
        model=model,
        limit=100,
        run_id=f"gsm8k-{name}",
    )
    print(f"{name}: {result.metrics['ExactMatch']:.2%}")

# Compare
from themis.comparison import compare_runs

report = compare_runs(
    [f"gsm8k-{name}" for name in models.keys()],
    storage_path=".cache/experiments",
)

print(report.summary())
```

### Pattern 3: Progressive Evaluation

```python
# Start small, scale up
for limit in [10, 50, 100, 500, 1000]:
    result = evaluate(
        benchmark="gsm8k",
        model="gpt-4",
        limit=limit,
        run_id=f"gsm8k-n{limit}",
    )
    
    print(f"n={limit}: {result.metrics['ExactMatch']:.2%} "
          f"(cost: ${result.cost:.2f})")
    
    # Stop if accuracy plateaus
    if limit >= 100 and result.metrics['ExactMatch'] < 0.5:
        print("Accuracy too low, stopping")
        break
```

### Pattern 4: Ensemble Evaluation

```python
# Generate multiple samples and aggregate
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    num_samples=5,  # 5 responses per question
    temperature=0.7,
    limit=100,
)

# Use majority voting (implement custom metric)
# Or use Pass@K for code generation
```

---

## Need Help?

- Check the [documentation](../index.md)
- View [API reference](../api/overview.md)
- Ask on [GitHub Discussions](https://github.com/pittawat2542/themis/discussions)

---

## Contributing Examples

Have a useful example? Contribute it!

1. Create a new example in `examples-simple/`
2. Add documentation here
3. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

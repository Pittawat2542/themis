# Frequently Asked Questions

Common questions and answers about Themis.

## General

### What is Themis?

Themis is a modern LLM evaluation framework for researchers and practitioners. It provides:
- Simple one-line Python API
- Built-in benchmarks (GSM8K, MATH500, etc.)
- Statistical comparison tools
- Web dashboard for results
- 100+ LLM provider support via LiteLLM

### Who should use Themis?

- **Researchers**: Evaluate models on benchmarks, compare approaches
- **ML Engineers**: Test prompts, compare models, track experiments
- **Students**: Learn about LLM evaluation and benchmarking
- **Teams**: Share and compare results via web dashboard

### How is Themis different from other evaluation frameworks?

- **Simpler API**: One-line `evaluate()` vs complex config files
- **Batteries included**: 6 benchmarks, 10+ metrics out of the box
- **Statistical rigor**: Built-in statistical tests for comparisons
- **Modern stack**: FastAPI, WebSocket, clean architecture

---

## Installation

### How do I install Themis?

```bash
pip install themis-eval
```

With optional features:
```bash
pip install themis-eval[all]
```

### What Python version do I need?

Python 3.12 or higher.

### Do I need to install dependencies manually?

No, pip/uv handles all dependencies automatically.

### Can I use it with Poetry/pipenv?

Yes:
```bash
poetry add themis-eval
pipenv install themis-eval
```

---

## Usage

### How do I run my first evaluation?

```python
from themis import evaluate

result = evaluate(benchmark="demo", model="fake-math-llm")
print(result.evaluation_report.metrics)
```

### Do I need API keys?

Only for real models (GPT-4, Claude, etc.). Use `fake-math-llm` for testing without API keys.

### How do I set API keys?

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

Or in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

### Can I use local models?

Yes! Via Ollama, vLLM, or other providers:

```python
# Ollama
result = evaluate(benchmark="gsm8k", model="ollama/llama3")

# vLLM
result = evaluate(benchmark="gsm8k", model="openai/meta-llama/Llama-3-8b")
```

### How do I evaluate my own dataset?

```python
dataset = [
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "What is 5-3?", "answer": "2"},
]

result = evaluate(dataset, model="gpt-4", prompt="{prompt}")
```

---

## Performance

### How fast is Themis?

Depends on:
- **Model**: GPT-4 ~1-2s/sample, local models ~0.1-0.5s/sample
- **Workers**: More workers = faster (up to rate limits)
- **Caching**: Cached results are instant

Typical: 100 samples with GPT-4 ≈ 2-5 minutes (with 8 workers).

### How do I make it faster?

1. **Increase workers**: `workers=16`
2. **Use caching**: `resume=True` (default)
3. **Use local models**: Faster than API calls
4. **Reduce samples**: `limit=100` for testing

### Does it support batching?

Not yet, but planned for future releases.

### Can I use GPUs?

Yes, if using local models (Ollama, vLLM) that support GPU inference.

---

## Caching

### How does caching work?

Themis caches:
- **Generation results**: Responses from the model
- **Evaluation results**: Computed metrics

Cache key includes model config, prompt, and sample.

### When is cache invalidated?

When you change:
- Model parameters (temperature, max_tokens, etc.)
- Prompt template
- Evaluation metrics

### How do I disable caching?

```python
result = evaluate(benchmark="gsm8k", model="gpt-4", resume=False)
```

### How do I clear cache?

```bash
# Clear all cache
rm -rf .cache/experiments

# Clear specific run
themis clean --run-id my-experiment
```

### Where is cache stored?

Default: `.cache/experiments`

Custom:
```python
result = evaluate(benchmark="gsm8k", model="gpt-4", storage="~/my-cache")
```

---

## Comparison

### How do I compare two models?

```python
from themis import evaluate
from themis.comparison import compare_runs

# Run evaluations
evaluate(benchmark="gsm8k", model="gpt-4", run_id="run-gpt4")
evaluate(benchmark="gsm8k", model="claude-3", run_id="run-claude")

# Compare
report = compare_runs(["run-gpt4", "run-claude"], storage_path=".cache/experiments")
print(report.summary())
```

### What statistical tests are available?

- **Bootstrap** (default) - Non-parametric, robust
- **T-test** - Fast, provides effect sizes
- **Permutation** - Exact p-values
- **None** - No statistical testing

### How do I interpret p-values?

- `p < 0.05` - Statistically significant (95% confidence)
- `p < 0.01` - Highly significant (99% confidence)
- `p ≥ 0.05` - Not statistically significant

### Can I compare more than 2 runs?

Yes! Win/loss matrices show all pairwise comparisons:

```python
report = compare_runs(["run-1", "run-2", "run-3"], storage_path=".cache")
```

---

## Web Dashboard

### How do I start the dashboard?

```bash
themis serve
```

Then open `http://localhost:8080/dashboard`

### Do I need to install anything extra?

Yes:
```bash
pip install themis-eval[server]
```

### Can I access it remotely?

```bash
themis serve --host 0.0.0.0 --port 8080
```

Then access via your machine's IP address.

### Is there authentication?

Not by default. For production, add authentication (see [API Server docs](../reference/api-server.md#authentication)).

---

## Costs

### How much does it cost to run evaluations?

Depends on the model and dataset size:
- GPT-4: ~$0.03 per 1K tokens (input) + $0.06 per 1K tokens (output)
- GPT-3.5: ~$0.001 per 1K tokens
- Local models: Free (hardware costs only)

Example: GSM8K (8.5K samples) with GPT-4 ≈ $50-100

### How do I track costs?

```python
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)
cost = result.metadata.get("cost", {}).get("total_cost", 0.0)
print(f"Cost: ${cost:.2f}")
```

### Can I set a budget limit?

Not directly, but you can:
```python
# Estimate first
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=10)
cost = result.metadata.get("cost", {}).get("total_cost", 0.0)
estimated_full = cost * 850  # GSM8K has 8500 samples

if estimated_full < 100:  # Budget check
    result = evaluate(benchmark="gsm8k", model="gpt-4")
```

---

## Troubleshooting

### Evaluation hangs or is very slow

1. Check API key is set correctly
2. Reduce `workers` (may be hitting rate limits)
3. Try with `fake-math-llm` to isolate issue
4. Check internet connection

### Import errors

```bash
# Install missing optional dependencies
pip install themis-eval[math,nlp,code]
```

### "Run not found" error

```bash
# List available runs
themis list runs

# Check run ID spelling
```

### Server won't start

```bash
# Install server dependencies
pip install themis-eval[server]

# Check if port is available
lsof -i :8080
```

### Cache seems stale

```python
# Force re-evaluation
result = evaluate(benchmark="gsm8k", model="gpt-4", resume=False)
```

---

## Advanced

### Can I use custom storage backends?

Yes! Implement the `StorageBackend` interface:

```python
from themis.backends import StorageBackend

class MyStorage(StorageBackend):
    # Implement methods
    pass

result = evaluate(benchmark="gsm8k", model="gpt-4", storage_backend=MyStorage())
```

See [Extending Backends](EXTENDING_BACKENDS.md) for details.

### Can I use distributed execution?

The interface is defined, but Ray/Dask implementations are not included. You can implement your own:

```python
from themis.backends import ExecutionBackend

class RayBackend(ExecutionBackend):
    # Implement methods
    pass

result = evaluate(benchmark="gsm8k", model="gpt-4", execution_backend=RayBackend())
```

### Can I add custom metrics?

Yes:

```python
from themis.evaluation.metrics import Metric, MetricScore

class MyMetric(Metric):
    @property
    def name(self) -> str:
        return "MyMetric"
    
    def evaluate(self, response, reference) -> MetricScore:
        # Your logic
        return MetricScore(value=score)

result = evaluate(dataset=my_data, model="gpt-4", metrics=[MyMetric()])
```

### Can I add custom benchmarks?

Yes:

```python
from themis.presets.benchmarks import BenchmarkPreset, register_benchmark

preset = BenchmarkPreset(
    name="my-benchmark",
    # ... configuration
)

register_benchmark(preset)

# Now you can use it
result = evaluate(benchmark="my-benchmark", model="gpt-4")
```

---

## Support

### Where can I get help?

- **Documentation**: [docs/index.md](index.md)
- **Examples**: [examples-simple/](tutorials/examples.md)
- **GitHub Discussions**: Ask questions
- **GitHub Issues**: Report bugs

### How do I report a bug?

1. Check it's not already reported
2. Provide minimal reproduction
3. Include Python version, Themis version
4. Include error message and traceback

### How do I request a feature?

Open a GitHub issue with:
- Clear description of the feature
- Use case / motivation
- Example API you'd like to see

### Can I contribute?

Yes! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas we'd love help:
- Additional benchmarks
- New metrics
- Backend implementations
- Documentation improvements
- Bug fixes

---

## Comparison with Other Tools

### vs lm-evaluation-harness

- **Themis**: Simple API, fewer benchmarks, statistical comparison, web UI
- **lm-eval-harness**: Comprehensive benchmarks, no statistical comparison

### vs promptfoo

- **Themis**: Python-first, research-focused, statistical tests
- **promptfoo**: JavaScript, production-focused, no statistical tests

### vs Weights & Biases

- **Themis**: Local-first, open source, no account needed
- **W&B**: Cloud platform, experiment tracking, visualization

---

## Version Information

### What's new in v2.0?

- Simple `themis.evaluate()` API
- Built-in benchmarks
- Statistical comparison
- Web dashboard
- Breaking changes from v1.x

See [CHANGELOG.md](CHANGELOG.md) for details.

### Should I upgrade from v1.x?

Yes, if you want:
- Simpler API
- Statistical comparison
- Web dashboard

See [Migration Guide](MIGRATION.md) for help.

---

Have a question not answered here? Ask on [GitHub Discussions](https://github.com/pittawat2542/themis/discussions)!

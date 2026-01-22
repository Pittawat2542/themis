# Quick Start

Get up and running with Themis in 5 minutes!

## Your First Evaluation

The simplest way to use Themis is with the `evaluate()` function:

```python
from themis import evaluate

# Evaluate GPT-4 on GSM8K math problems
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    limit=10,  # Start with 10 samples
)

print(f"Accuracy: {result.metrics['ExactMatch']:.2%}")
```

That's it! Themis will:
1. Load the GSM8K benchmark
2. Generate responses using GPT-4
3. Evaluate with math metrics
4. Cache results for future runs

## Using the CLI

Prefer command-line tools?

```bash
# Run evaluation
themis eval gsm8k --model gpt-4 --limit 10

# View results
themis list runs

# Start web dashboard
themis serve
```

## Built-in Benchmarks

Themis includes 6 popular benchmarks:

```python
# Math reasoning
evaluate(benchmark="gsm8k", model="gpt-4")      # Grade school math
evaluate(benchmark="math500", model="gpt-4")    # Advanced math
evaluate(benchmark="aime24", model="gpt-4")     # Math competition

# Knowledge
evaluate(benchmark="mmlu_pro", model="gpt-4")   # General knowledge
evaluate(benchmark="supergpqa", model="gpt-4")  # Advanced reasoning

# Testing
evaluate(benchmark="demo", model="fake-math-llm")  # No API key needed!
```

List available benchmarks:

```python
from themis.presets import list_benchmarks
print(list_benchmarks())
```

## Custom Dataset

Evaluate on your own data:

```python
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

## Configuration Options

Customize the evaluation:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    
    # Sampling
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    
    # Execution
    num_samples=3,      # Sample 3 responses per prompt
    workers=8,          # Use 8 parallel workers
    
    # Storage
    storage=".cache",   # Where to store results
    run_id="my-exp",    # Unique identifier
    resume=True,        # Resume from cache
)
```

## Comparing Runs

Compare different configurations:

```python
from themis.comparison import compare_runs

# Run experiments
evaluate(benchmark="gsm8k", model="gpt-4", temperature=0.0, run_id="temp-0")
evaluate(benchmark="gsm8k", model="gpt-4", temperature=0.7, run_id="temp-0.7")

# Compare statistically
report = compare_runs(
    run_ids=["temp-0", "temp-0.7"],
    storage_path=".cache/experiments",
)

print(report.summary())
```

Or via CLI:

```bash
themis compare temp-0 temp-0.7 --output comparison.html
```

## Web Dashboard

Start the API server and view results in your browser:

```bash
themis serve
```

Then open:
- Dashboard: `http://localhost:8080/dashboard`
- API docs: `http://localhost:8080/docs`

## Multiple Providers

Themis works with 100+ LLM providers via [LiteLLM](https://docs.litellm.ai/):

```python
# OpenAI
evaluate(benchmark="gsm8k", model="gpt-4")
evaluate(benchmark="gsm8k", model="gpt-3.5-turbo")

# Anthropic
evaluate(benchmark="gsm8k", model="claude-3-opus-20240229")

# Azure OpenAI
evaluate(benchmark="gsm8k", model="azure/gpt-4")

# Local models
evaluate(benchmark="gsm8k", model="ollama/llama3")
```

## What's Next?

- **[Core Concepts](concepts.md)** - Understand how Themis works
- **[Evaluation Guide](../guides/evaluation.md)** - Deep dive into evaluation
- **[Comparison Guide](../COMPARISON.md)** - Statistical comparison
- **[Jupyter Tutorials](../tutorials/notebooks.md)** - Interactive learning
- **[API Reference](../api/overview.md)** - Complete API documentation

## Common Patterns

### Testing Before Running

Use the demo benchmark to test your setup:

```python
# No API key needed!
result = evaluate(
    benchmark="demo",
    model="fake-math-llm",
    limit=5,
)
```

### Resuming Failed Runs

If a run fails, resume from cache:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    run_id="my-experiment",
    resume=True,  # Skip already-evaluated samples
)
```

### Batch Processing

Evaluate multiple configurations:

```python
models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]

for model in models:
    result = evaluate(
        benchmark="gsm8k",
        model=model,
        limit=100,
        run_id=f"gsm8k-{model}",
    )
    print(f"{model}: {result.metrics['ExactMatch']:.2%}")
```

### Export Results

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    output="results.json",  # Export to JSON
)

# Also supports .csv and .html
```

## Tips

1. **Start Small**: Use `limit=10` for testing
2. **Use Run IDs**: Always specify `run_id` for reproducibility
3. **Enable Caching**: Set `resume=True` to avoid re-running
4. **Monitor Costs**: Check `result.cost` after evaluation
5. **Test Locally**: Use `fake-math-llm` model for testing

## Need Help?

- Check the [documentation](../index.md)
- View [examples](../tutorials/examples.md)
- Ask on [GitHub Discussions](https://github.com/pittawat2542/themis/discussions)
- Report issues on [GitHub Issues](https://github.com/pittawat2542/themis/issues)

Ready to dive deeper? Continue to [Core Concepts](concepts.md)!

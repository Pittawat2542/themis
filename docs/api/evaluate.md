# evaluate() API

The main entry point for running LLM evaluations.

## Signature

```python
def evaluate(
    benchmark_or_dataset: str | Sequence[dict[str, Any]],
    *,
    model: str,
    limit: int | None = None,
    prompt: str | None = None,
    metrics: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_samples: int = 1,
    distributed: bool = False,
    workers: int = 4,
    storage: str | Path | None = None,
    run_id: str | None = None,
    resume: bool = True,
    output: str | Path | None = None,
    on_result: Callable[[GenerationRecord], None] | None = None,
    **kwargs: Any,
) -> ExperimentReport:
```

## Parameters

### Required Parameters

**`benchmark_or_dataset`** : `str | Sequence[dict[str, Any]]`

Either:
- Name of a built-in benchmark (e.g., `"gsm8k"`, `"math500"`)
- Custom dataset as list of dictionaries

For custom datasets, each dict should have:
- `prompt` or `question` - The input prompt
- `answer` or `reference` - The expected output  
- Optional: `id`, additional fields for prompt template

**`model`** : `str`

Model identifier for LiteLLM. Examples:
- `"gpt-4"` - OpenAI GPT-4
- `"claude-3-opus-20240229"` - Anthropic Claude
- `"azure/gpt-4"` - Azure OpenAI
- `"ollama/llama3"` - Local Ollama model

See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for all options.

### Optional Parameters

**`limit`** : `int | None = None`

Maximum number of samples to evaluate. Useful for:
- Testing with small sample (`limit=10`)
- Quick experiments
- Budget constraints

If `None`, evaluates entire dataset.

**`prompt`** : `str | None = None`

Custom prompt template. Uses Python format strings:
- `"{prompt}"` - Insert prompt
- `"{question}"` - Insert question
- `"{context}"` - Insert context

Example: `"Question: {prompt}\nAnswer:"`

If `None`, uses benchmark's default prompt.

**`metrics`** : `list[str] | None = None`

List of metrics to compute. If `None`, uses benchmark's default metrics.

Available metrics:
- Math: `"ExactMatch"`, `"MathVerify"`
- NLP: `"BLEU"`, `"ROUGE"`, `"BERTScore"`, `"METEOR"`
- Code: `"PassAtK"`, `"CodeBLEU"`, `"ExecutionAccuracy"`

**`temperature`** : `float = 0.0`

Sampling temperature (0.0 = deterministic, 1.0+ = creative).

**`max_tokens`** : `int = 512`

Maximum tokens in model response.

**`num_samples`** : `int = 1`

Number of responses to generate per prompt. Useful for:
- Pass@K evaluation (`num_samples=10`)
- Ensembling
- Measuring variance

**`workers`** : `int = 4`

Number of parallel workers for generation. Higher values = faster execution but more API load.

**`storage`** : `str | Path | None = None`

Path to storage directory. Defaults to `.cache/experiments`.

**`run_id`** : `str | None = None`

Unique identifier for the run. If `None`, auto-generated from timestamp.

**`resume`** : `bool = True`

Whether to resume from cached results. Set to `False` to force re-evaluation.

**`output`** : `str | Path | None = None`

Export results to file. Supported formats:
- `.json` - JSON format
- `.csv` - CSV format
- `.html` - HTML report

**`on_result`** : `Callable[[GenerationRecord], None] | None = None`

Callback function called after each sample is generated. Useful for:
- Progress tracking
- Real-time monitoring
- Custom logging

**`**kwargs`** : `Any`

Additional keyword arguments passed to the model provider (e.g., `top_p`, `frequency_penalty`).

## Return Value

**`ExperimentReport`**

Object containing:
- `run_id` : `str` - Unique run identifier
- `metrics` : `dict[str, float]` - Metric scores
- `num_samples` : `int` - Number of samples evaluated
- `cost` : `float` - Estimated API cost
- `report` : `str` - Formatted text report
- Additional metadata

## Examples

### Basic Usage

```python
from themis import evaluate

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    limit=100,
)

print(f"Accuracy: {result.metrics['ExactMatch']:.2%}")
print(f"Cost: ${result.cost:.2f}")
```

### Custom Dataset

```python
dataset = [
    {"prompt": "What is 2+2?", "answer": "4"},
    {"prompt": "What is 5-3?", "answer": "2"},
]

result = evaluate(
    dataset,
    model="gpt-4",
    prompt="Solve: {prompt}",
    metrics=["ExactMatch"],
)
```

### Advanced Configuration

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    
    # Sampling
    temperature=0.7,
    max_tokens=1024,
    top_p=0.95,
    frequency_penalty=0.2,
    
    # Execution
    num_samples=3,
    workers=16,
    
    # Storage
    storage="~/experiments",
    run_id="gsm8k-gpt4-temp07",
    resume=True,
    
    # Output
    output="results.html",
)
```

### With Callback

```python
def log_progress(record):
    print(f"Completed: {record.id}")

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    limit=100,
    on_result=log_progress,
)
```

### Multiple Samples

```python
# Generate 10 responses per prompt
result = evaluate(
    benchmark="code-problems",
    model="gpt-4",
    num_samples=10,
    metrics=["PassAtK"],  # Evaluate Pass@K
)

print(f"Pass@1: {result.metrics['Pass@1']:.2%}")
print(f"Pass@10: {result.metrics['Pass@10']:.2%}")
```

## Error Handling

```python
from themis import evaluate

try:
    result = evaluate(
        benchmark="invalid-benchmark",
        model="gpt-4",
    )
except ValueError as e:
    print(f"Invalid input: {e}")
except FileNotFoundError as e:
    print(f"Storage error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## See Also

- [User Guide - Evaluation](../guides/evaluation.md) - Detailed usage guide
- [Comparison API](comparison.md) - Compare multiple runs
- [Presets API](presets.md) - Built-in benchmarks
- [Examples](../tutorials/examples.md) - Working code examples

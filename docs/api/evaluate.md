# evaluate() API

Primary entry point for running LLM evaluations.

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
    storage_backend: object | None = None,
    execution_backend: object | None = None,
    run_id: str | None = None,
    resume: bool = True,
    on_result: Callable[[GenerationRecord], None] | None = None,
    **kwargs: Any,
) -> ExperimentReport:
```

## Parameters

### Required

**`benchmark_or_dataset`** : `str | Sequence[dict[str, Any]]`

- Benchmark name (e.g., `"gsm8k"`, `"math500"`)
- Or custom dataset as a list of dictionaries

For custom datasets, each dict should include:
- `prompt` / `question` (input)
- `answer` / `reference` (expected output)
- Optional `id` / `unique_id`

**`model`** : `str`

Model identifier for provider routing. Examples:
- `"gpt-4"`
- `"claude-3-opus-20240229"`
- `"azure/gpt-4"`
- `"ollama/llama3"`

### Optional

**`limit`** : `int | None = None`

Maximum number of samples to evaluate.

**`prompt`** : `str | None = None`

Custom prompt template using Python format fields (e.g. `"Q: {question}\nA:"`).
If `None`, uses the benchmark preset template.

**`metrics`** : `list[str] | None = None`

Metric names to compute. If `None`, uses preset defaults. Example names:
- `"exact_match"`
- `"math_verify"`
- `"bleu"`, `"rouge1"`, `"bertscore"`, `"meteor"`
- `"pass_at_k"`, `"execution_accuracy"`, `"codebleu"`

Metric names are normalized (case-insensitive; `ExactMatch` and `exact_match` both work).

**`temperature`** : `float = 0.0`

Sampling temperature.

**`max_tokens`** : `int = 512`

Maximum tokens generated per response.

**`num_samples`** : `int = 1`

Number of samples per prompt. This is currently only partially wired in the spec/session flow.

**`distributed`** : `bool = False`

Reserved for future distributed execution. Currently ignored.

**`workers`** : `int = 4`

Parallel worker count for generation.

**`storage`** : `str | Path | None = None`

Storage location for runs and cache. Defaults to `.cache/experiments`.

**`storage_backend`** : `object | None = None`

Optional storage backend instance (typically `ExperimentStorage` or
`LocalFileStorageBackend`). Custom storage backends are not yet wired into
`ExperimentSession`.

**`execution_backend`** : `object | None = None`

Optional execution backend for custom parallelism.

**`run_id`** : `str | None = None`

Explicit run identifier. If `None`, one is generated automatically.

**`resume`** : `bool = True`

If `True`, reuse cached results when available.

**`on_result`** : `Callable[[GenerationRecord], None] | None = None`

Callback invoked per generation record.

**`**kwargs`** : `Any`

Currently used for `top_p` in sampling. Other provider-specific kwargs are
reserved for future wiring.

## Return Value

**`ExperimentReport`** containing:
- `generation_results`: list of `GenerationRecord`
- `evaluation_report`: `EvaluationReport` with aggregates and per-sample scores
- `failures`: generation failures
- `metadata`: run metadata

Access aggregate metrics via:

```python
report.evaluation_report.metrics["ExactMatch"].mean
```

## Examples

### Basic benchmark

```python
from themis import evaluate

report = evaluate(
    "gsm8k",
    model="gpt-4",
    limit=100,
)

accuracy = report.evaluation_report.metrics["ExactMatch"].mean
print(f"Accuracy: {accuracy:.2%}")
```

### Custom dataset

```python
dataset = [
    {"id": "1", "question": "2+2", "answer": "4"},
    {"id": "2", "question": "3+3", "answer": "6"},
]

report = evaluate(
    dataset,
    model="gpt-4",
    prompt="Q: {question}\nA:",
    metrics=["exact_match"],
)
```

### Advanced storage + execution

```python
from themis import evaluate
from themis.backends.execution import LocalExecutionBackend
from themis.backends.storage import LocalFileStorageBackend

report = evaluate(
    "math500",
    model="gpt-4",
    storage_backend=LocalFileStorageBackend(".cache/experiments"),
    execution_backend=LocalExecutionBackend(max_workers=8),
)
```

### Exporting results

`evaluate()` does not export files directly. Use export helpers:

```python
from pathlib import Path
from themis.experiment import export

report = evaluate("gsm8k", model="gpt-4", limit=50)
export.export_report_json(report, Path("report.json"))
export.export_report_csv(report, Path("report.csv"))
export.export_html_report(report, Path("report.html"))
```

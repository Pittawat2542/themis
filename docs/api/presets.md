# Presets API

Built-in benchmark configurations.

## Functions

### list_benchmarks()

List all available benchmark presets.

```python
from themis.presets import list_benchmarks

benchmarks = list_benchmarks()
print(benchmarks)
# ['demo', 'gsm8k', 'math500', 'aime24', ...]
```

**Returns**: `list[str]` - Benchmark names

---

### get_benchmark_preset()

Get configuration for a specific benchmark.

```python
from themis.presets import get_benchmark_preset

preset = get_benchmark_preset("gsm8k")
```

**Parameters:**
- **`name`** : `str` - Benchmark name

**Returns**: `BenchmarkPreset`

**Raises**: `ValueError` if benchmark not found

---

## Classes

### BenchmarkPreset

Complete benchmark configuration.

**Attributes:**
- **`name`** : `str` - Benchmark identifier
- **`prompt_template`** : `PromptTemplate` - Prompt formatting
- **`metrics`** : `list[Metric]` - Metric instances
- **`extractor`** : `Extractor` - Output extractor
- **`dataset_loader`** : `Callable` - Dataset loading function
- **`metadata_fields`** : `tuple[str, ...]` - Fields to preserve
- **`reference_field`** : `str` - Field containing reference answer
- **`dataset_id_field`** : `str` - Field containing sample ID
- **`description`** : `str` - Human-readable description

---

## Example

```python
from themis.evaluation import extractors, metrics
from themis.generation.templates import PromptTemplate
from themis.presets.core import BenchmarkPreset, register_benchmark


def my_dataset_loader(limit=None):
    data = [
        {"id": "1", "question": "2+2", "answer": "4"},
        {"id": "2", "question": "3+3", "answer": "6"},
    ]
    return data[:limit] if limit else data


preset = BenchmarkPreset(
    name="my-benchmark",
    prompt_template=PromptTemplate(name="my-template", template="Q: {question}\nA:"),
    metrics=[metrics.ExactMatch()],
    extractor=extractors.IdentityExtractor(),
    dataset_loader=my_dataset_loader,
    description="My custom benchmark",
)

register_benchmark(preset)
```

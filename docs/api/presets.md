# Presets API

Built-in benchmark configurations.

## Functions

### list_benchmarks()

List all available benchmark presets.

```python
from themis.presets import list_benchmarks

benchmarks = list_benchmarks()
print(benchmarks)
# ['demo', 'gsm8k', 'math500', 'aime24', 'mmlu_pro', 'supergpqa']
```

**Returns**: `list[str]` - List of benchmark names

**Example:**
```python
for benchmark in list_benchmarks():
    print(f"- {benchmark}")
```

---

### get_benchmark_preset()

Get configuration for a specific benchmark.

```python
from themis.presets import get_benchmark_preset

preset = get_benchmark_preset("gsm8k")
```

**Parameters:**
- **`name`** : `str` - Benchmark name

**Returns**: `BenchmarkPreset` - Benchmark configuration

**Raises**: `ValueError` if benchmark not found

**Example:**
```python
preset = get_benchmark_preset("gsm8k")
print(f"Prompt: {preset.prompt_template}")
print(f"Metrics: {[m.name for m in preset.metrics]}")
print(f"Reference field: {preset.reference_field}")
```

---

## Classes

### BenchmarkPreset

Complete benchmark configuration.

**Attributes:**

- **`name`** : `str` - Benchmark identifier
- **`prompt_template`** : `PromptTemplate` - Prompt formatting
- **`metrics`** : `list[Metric]` - Evaluation metrics
- **`extractor`** : `Extractor` - Answer extractor
- **`dataset_loader`** : `Callable` - Dataset loading function
- **`metadata_fields`** : `tuple[str, ...]` - Fields to preserve
- **`reference_field`** : `str` - Field containing reference answer (default: "answer")
- **`dataset_id_field`** : `str` - Field containing sample ID (default: "id")
- **`description`** : `str` - Human-readable description

**Example:**
```python
from themis.presets import get_benchmark_preset

preset = get_benchmark_preset("gsm8k")

# Access configuration
print(f"Name: {preset.name}")
print(f"Description: {preset.description}")
print(f"Metrics: {[m.name for m in preset.metrics]}")

# Use for evaluation
from themis import evaluate
result = evaluate(benchmark=preset.name, model="gpt-4")
```

---

## Built-in Benchmarks

### demo

**Description**: Quick testing benchmark (10 samples)

**Dataset**: Subset of GSM8K  
**Metrics**: ExactMatch, MathVerify  
**Prompt**: `"Solve the problem: {prompt}"`

```python
result = evaluate(benchmark="demo", model="fake-math-llm")
```

---

### gsm8k

**Description**: Grade School Math 8K - elementary math word problems

**Dataset**: 8,500+ problems  
**Metrics**: ExactMatch, MathVerify  
**Prompt**: `"Problem: {prompt}\nSolution:"`

```python
result = evaluate(benchmark="gsm8k", model="gpt-4", limit=100)
```

---

### math500

**Description**: MATH500 - advanced math competition problems

**Dataset**: 500 problems  
**Metrics**: ExactMatch, MathVerify  
**Prompt**: `"Solve this problem: {problem}"`

```python
result = evaluate(benchmark="math500", model="gpt-4")
```

---

### aime24

**Description**: AIME 2024 - American Invitational Mathematics Examination

**Dataset**: 30 problems  
**Metrics**: ExactMatch, MathVerify  
**Prompt**: `"Problem: {problem}\n\nSolution:"`

```python
result = evaluate(benchmark="aime24", model="gpt-4")
```

---

### mmlu_pro

**Description**: MMLU-Pro - massive multitask language understanding (professional)

**Dataset**: Multiple subjects  
**Metrics**: ExactMatch  
**Prompt**: Includes multiple-choice format

```python
result = evaluate(benchmark="mmlu_pro", model="gpt-4", limit=1000)
```

---

### supergpqa

**Description**: SuperGPQA - advanced reasoning and knowledge questions

**Dataset**: Challenging QA  
**Metrics**: ExactMatch  
**Prompt**: `"Question: {question}\nAnswer:"`

```python
result = evaluate(benchmark="supergpqa", model="gpt-4")
```

---

## Creating Custom Presets

You can register custom benchmarks:

```python
from themis.presets.benchmarks import BenchmarkPreset, register_benchmark
from themis.generation.prompt_template import PromptTemplate
from themis.evaluation.metrics.math import ExactMatch
from themis.evaluation.extractors import LastLineExtractor

def my_dataset_loader(limit=None):
    # Load your dataset
    return [{"prompt": "...", "answer": "..."}]

preset = BenchmarkPreset(
    name="my-benchmark",
    prompt_template=PromptTemplate(template="Q: {prompt}\nA:"),
    metrics=[ExactMatch()],
    extractor=LastLineExtractor(),
    dataset_loader=my_dataset_loader,
    description="My custom benchmark",
)

register_benchmark(preset)

# Now you can use it
result = evaluate(benchmark="my-benchmark", model="gpt-4")
```

---

## Model Parsing

### parse_model_name()

Parse model name to detect provider and options.

```python
from themis.presets.models import parse_model_name

provider, model, options = parse_model_name("gpt-4", temperature=0.7)
print(provider)  # "litellm"
print(model)     # "gpt-4"
print(options)   # {"temperature": 0.7}
```

---

## See Also

- [Evaluation Guide](../guides/evaluation.md) - Using benchmarks
- [Custom Datasets](../guides/evaluation.md#custom-datasets) - Create your own
- [API Reference](overview.md) - Complete API documentation

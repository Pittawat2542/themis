# Extension Quick Reference

One-page reference for extending Themis with custom components.

## Registration APIs

```python
import themis

# Register custom metrics
themis.register_metric("my_metric", MyMetric)
themis.get_registered_metrics()  # Query registered metrics

# Register custom datasets  
themis.register_dataset("my_dataset", create_dataset_function)
themis.list_datasets()  # List all datasets
themis.is_dataset_registered("my_dataset")  # Check if registered

# Register custom providers
themis.register_provider("my_provider", create_provider_function)

# Register benchmark presets
themis.register_benchmark(BenchmarkPreset(...))
themis.list_benchmarks()  # List all benchmarks
themis.get_benchmark_preset("my_benchmark")  # Get preset config
```

## Component Templates

### Custom Metric

```python
from dataclasses import dataclass
from themis.interfaces import Metric
from themis.core.entities import MetricScore

@dataclass
class MyMetric(Metric):
    threshold: float = 0.5  # Config params
    
    def __post_init__(self):
        self.name = "my_metric"
    
    def compute(self, *, prediction, references=None, metadata=None):
        score = self._score(prediction)
        return MetricScore(
            metric_name=self.name,
            value=score,  # 0.0 to 1.0
            details={},
            metadata=metadata or {},
        )
    
    def _score(self, prediction):
        return 1.0

themis.register_metric("my_metric", MyMetric)
```

### Custom Dataset

```python
def create_my_dataset(options: dict) -> list[dict]:
    """Load and return dataset samples.
    
    Args:
        options: {
            "limit": Max samples,
            "split": "train"/"test",
            ...custom options...
        }
    
    Returns:
        [{"id": "1", "question": "...", "answer": "..."}, ...]
    """
    samples = load_from_somewhere(
        limit=options.get("limit"),
        split=options.get("split", "test"),
    )
    
    return [
        {
            "id": s.id,
            "question": s.question,
            "answer": s.answer,
        }
        for s in samples
    ]

themis.register_dataset("my_dataset", create_my_dataset)
```

### Custom Provider

```python
from themis.interfaces import ModelProvider

def create_my_provider(**options):
    return MyProvider(
        api_key=options.get("api_key"),
        base_url=options.get("base_url"),
    )

class MyProvider(ModelProvider):
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
    
    def generate(self, model, prompt, temperature=0.0, max_tokens=512, **kwargs):
        # Call your API
        response = self._call_api(model, prompt, temperature, max_tokens)
        return response["text"]
    
    def _call_api(self, model, prompt, temperature, max_tokens):
        import requests
        return requests.post(
            f"{self.base_url}/generate",
            json={"model": model, "prompt": prompt},
        ).json()

themis.register_provider("my_provider", create_my_provider)
```

### Custom Extractor

```python
from themis.interfaces import Extractor

class MyExtractor(Extractor):
    def extract(self, output: str, **context) -> str:
        """Extract answer from raw output."""
        # Parse/extract logic
        return output.strip()

# Use directly (no registration)
extractor = MyExtractor()
```

### Custom Benchmark Preset

```python
from themis.presets import BenchmarkPreset
from themis.generation.templates import PromptTemplate
from themis.evaluation.metrics import ExactMatch
from themis.evaluation.extractors import IdentityExtractor

preset = BenchmarkPreset(
    name="my_benchmark",
    prompt_template=PromptTemplate(
        name="my_prompt",
        template="Q: {question}\nA:",
    ),
    metrics=[ExactMatch()],
    extractor=IdentityExtractor(),
    dataset_loader=lambda limit: load_samples(limit),
    reference_field="answer",
    dataset_id_field="id",
    description="My custom benchmark",
)

themis.register_benchmark(preset)
```

### Custom Prompt Template

```python
from themis.generation.templates import PromptTemplate

template = PromptTemplate(
    name="my_template",
    template="Question: {question}\n\nAnswer:",
    metadata={"author": "me"},
)

# Use directly (no registration)
prompt_text = template.render(question="What is AI?")
```

## Usage Patterns

### Using Registered Components

```python
import themis

# Use custom metric by name
report = themis.evaluate(
    "gsm8k",
    model="gpt-4",
    metrics=["my_metric", "exact_match"],
)

# Use custom dataset by name
report = themis.evaluate(
    "my_dataset",
    model="gpt-4",
    prompt="Q: {question}\nA:",
)

# Use custom provider by name
report = themis.evaluate(
    "gsm8k",
    model="my_provider/my_model",
    api_key="...",
)

# Use custom benchmark by name
report = themis.evaluate(
    "my_benchmark",
    model="gpt-4",
    limit=100,
)
```

### Using Direct Components

```python
from my_module import MyExtractor, MyTemplate

# Pass extractor directly
report = themis.evaluate(
    dataset,
    model="gpt-4",
    extractor=MyExtractor(),
)

# Use template directly
report = themis.evaluate(
    dataset,
    model="gpt-4",
    prompt=MyTemplate().template,
)
```

## What Needs Registration?

| Component | Register? | API | Use By |
|-----------|-----------|-----|--------|
| Metrics | ✅ Yes | `register_metric()` | Name: `metrics=["name"]` |
| Datasets | ✅ Yes | `register_dataset()` | Name: `evaluate("name")` |
| Providers | ✅ Yes | `register_provider()` | Name: `model="name/model"` |
| Benchmarks | ✅ Yes | `register_benchmark()` | Name: `evaluate("name")` |
| Extractors | ❌ No | - | Direct: `extractor=MyExtractor()` |
| Templates | ❌ No | - | Direct: `prompt=template.template` |

## Standard Field Names

### Dataset Samples

Required fields:
```python
{
    "id": "unique-id",           # Unique identifier
    "question": "...",            # Or "problem", "prompt"
    "answer": "...",              # Or "solution", "reference"
}
```

Optional metadata (preserved in evaluation):
```python
{
    "category": "...",
    "difficulty": "...",
    "subject": "...",
    # Any custom fields
}
```

### MetricScore

```python
MetricScore(
    metric_name="my_metric",     # Same as self.name
    value=0.85,                   # 0.0 to 1.0 (higher is better)
    details={"info": "..."},      # Debugging info (optional)
    metadata=metadata or {},      # Pass through from input
)
```

## Common Patterns

### Lazy Loading

```python
def create_dataset(options):
    # Import heavy dependencies only when needed
    from datasets import load_dataset
    
    ds = load_dataset(options.get("dataset"))
    return list(ds)
```

### Configuration Parameters

```python
@dataclass
class ConfigurableMetric(Metric):
    threshold: float = 0.5
    case_sensitive: bool = False
    
    def __post_init__(self):
        self.name = f"metric_t{self.threshold}"
```

### Error Handling

```python
def extract(self, output, **context):
    try:
        return self._parse(output)
    except ParseError as e:
        # Log warning and return fallback
        logger.warning(f"Parse failed: {e}")
        return output.strip()
```

### Chaining Components

```python
# Create benchmark that uses custom components
preset = BenchmarkPreset(
    name="my_benchmark",
    prompt_template=MyTemplate(),
    metrics=[MyMetric(), ExactMatch()],
    extractor=MyExtractor(),
    dataset_loader=lambda limit: create_my_dataset({"limit": limit}),
)
```

## Examples Location

```
themis/
├── evaluation/
│   ├── metrics/
│   │   ├── exact_match.py              ← Metric examples
│   │   ├── math_verify_accuracy.py
│   │   └── nlp/bleu.py
│   └── extractors/
│       ├── identity_extractor.py       ← Extractor examples
│       └── regex_extractor.py
├── datasets/
│   ├── math500.py                      ← Dataset examples
│   └── gsm8k.py
├── generation/
│   └── providers/
│       └── litellm_provider.py         ← Provider examples
└── presets/
    └── benchmarks.py                    ← Benchmark examples
```

## Testing Your Component

```python
import themis

# 1. Register
themis.register_metric("test_metric", TestMetric)

# 2. Test with minimal data
report = themis.evaluate(
    [{"id": "1", "question": "test", "answer": "test"}],
    model="fake",
    prompt="Q: {question}\nA:",
    metrics=["test_metric"],
)

# 3. Verify
assert "test_metric" in report.evaluation_report.metrics
print(f"Score: {report.evaluation_report.metrics['test_metric'].mean}")
```

## Full Example

```python
import themis
from dataclasses import dataclass
from themis.interfaces import Metric
from themis.core.entities import MetricScore

# Define
@dataclass
class ContainsSolution(Metric):
    keyword: str = "answer"
    
    def __post_init__(self):
        self.name = f"contains_{self.keyword}"
    
    def compute(self, *, prediction, references=None, metadata=None):
        contains = self.keyword.lower() in prediction.lower()
        return MetricScore(
            metric_name=self.name,
            value=1.0 if contains else 0.0,
            details={"keyword": self.keyword},
            metadata=metadata or {},
        )

# Register
themis.register_metric("contains_solution", ContainsSolution)

# Use
report = themis.evaluate(
    "gsm8k",
    model="gpt-4",
    metrics=["contains_solution", "math_verify"],
    limit=100,
)

print(f"Contains 'answer': {report.evaluation_report.metrics['contains_solution'].mean:.1%}")
```

---

**Need more details?** See [EXTENDING_THEMIS.md](EXTENDING_THEMIS.md) for complete guide with interfaces, examples, and best practices.

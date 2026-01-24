# Themis Extension Architecture

Visual overview of how to extend Themis with custom components.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│                                                             │
│    import themis                                           │
│    report = themis.evaluate("my_benchmark", model="gpt-4")│
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Extension Points                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Metrics    │  │  Datasets    │  │  Providers   │   │
│  │  (register)  │  │  (register)  │  │  (register)  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Benchmarks  │  │  Extractors  │  │  Templates   │   │
│  │  (register)  │  │   (direct)   │  │   (direct)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core Engine                               │
│                                                             │
│  Generation → Extraction → Evaluation → Aggregation       │
└─────────────────────────────────────────────────────────────┘
```

## Extension Points Detail

### 1. Metrics (Registration Required)

**Purpose**: Define how to score model outputs

```
┌─────────────────────────────────────┐
│  User Code                         │
│                                     │
│  @dataclass                        │
│  class MyMetric(Metric):           │
│      def compute(...):             │
│          return MetricScore(...)   │
│                                     │
│  themis.register_metric(           │
│      "my_metric", MyMetric         │
│  )                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Metrics Registry                   │
│                                     │
│  {                                  │
│    "exact_match": ExactMatch,      │
│    "math_verify": MathVerify,      │
│    "my_metric": MyMetric  ← NEW    │
│  }                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Usage                              │
│                                     │
│  themis.evaluate(                   │
│      metrics=["my_metric", "..."]  │
│  )                                  │
└─────────────────────────────────────┘
```

**Registry Location**: `themis.api._METRICS_REGISTRY`
**API**: `themis.register_metric(name, class)`
**Query**: `themis.get_registered_metrics()`

### 2. Datasets (Registration Required)

**Purpose**: Load evaluation datasets

```
┌─────────────────────────────────────┐
│  User Code                         │
│                                     │
│  def create_my_dataset(options):   │
│      samples = load_data(...)      │
│      return [                       │
│          {"id": "1", ...},         │
│          {"id": "2", ...},         │
│      ]                              │
│                                     │
│  themis.register_dataset(          │
│      "my_dataset",                 │
│      create_my_dataset             │
│  )                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Dataset Registry                   │
│                                     │
│  {                                  │
│    "math500": create_math500,      │
│    "gsm8k": create_gsm8k,          │
│    "my_dataset": create_my_dataset │
│  }                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Usage                              │
│                                     │
│  themis.evaluate("my_dataset", ...) │
└─────────────────────────────────────┘
```

**Registry Location**: `themis.datasets.registry._REGISTRY`
**API**: `themis.register_dataset(name, factory)`
**Query**: `themis.list_datasets()`, `themis.is_dataset_registered(name)`

### 3. Providers (Registration Required)

**Purpose**: Connect to LLM APIs

```
┌─────────────────────────────────────┐
│  User Code                         │
│                                     │
│  def create_my_provider(**options):│
│      return MyProvider(            │
│          api_key=options["api_key"],│
│          ...                        │
│      )                              │
│                                     │
│  class MyProvider(ModelProvider):  │
│      def generate(...):            │
│          # Call API                │
│          return response           │
│                                     │
│  themis.register_provider(         │
│      "my_provider",                │
│      create_my_provider            │
│  )                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Provider Registry                  │
│                                     │
│  {                                  │
│    "litellm": create_litellm,      │
│    "fake": create_fake,            │
│    "my_provider": create_my_prov   │
│  }                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Usage                              │
│                                     │
│  themis.evaluate(                   │
│      model="my_provider/model"     │
│  )                                  │
└─────────────────────────────────────┘
```

**Registry Location**: `themis.providers.registry._REGISTRY`
**API**: `themis.register_provider(name, factory)`

### 4. Benchmarks (Registration Required)

**Purpose**: Bundle dataset + prompt + metrics + extractor

```
┌─────────────────────────────────────┐
│  User Code                         │
│                                     │
│  preset = BenchmarkPreset(         │
│      name="my_benchmark",          │
│      prompt_template=...,          │
│      metrics=[...],                │
│      extractor=...,                │
│      dataset_loader=...,           │
│  )                                  │
│                                     │
│  themis.register_benchmark(preset) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Benchmark Registry                 │
│                                     │
│  {                                  │
│    "math500": <preset>,            │
│    "gsm8k": <preset>,              │
│    "my_benchmark": <preset>        │
│  }                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Usage                              │
│                                     │
│  themis.evaluate("my_benchmark")    │
│  # Auto-uses: prompt, metrics, etc │
└─────────────────────────────────────┘
```

**Registry Location**: `themis.presets.benchmarks._BENCHMARK_REGISTRY`
**API**: `themis.register_benchmark(preset)`
**Query**: `themis.list_benchmarks()`, `themis.get_benchmark_preset(name)`

### 5. Extractors (Direct Usage - No Registration)

**Purpose**: Parse/extract answers from model outputs

```
┌─────────────────────────────────────┐
│  User Code                         │
│                                     │
│  class MyExtractor(Extractor):     │
│      def extract(self, output):    │
│          # Parse output            │
│          return extracted          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Direct Usage (No Registry)        │
│                                     │
│  themis.evaluate(                   │
│      ...,                           │
│      extractor=MyExtractor()       │
│  )                                  │
└─────────────────────────────────────┘
```

**Location**: `themis/evaluation/extractors/`
**Usage**: Pass directly to `evaluate(extractor=...)`

### 6. Templates (Direct Usage - No Registration)

**Purpose**: Reusable prompt formats

```
┌─────────────────────────────────────┐
│  User Code                         │
│                                     │
│  template = PromptTemplate(        │
│      name="my_template",           │
│      template="Q: {q}\nA:"         │
│  )                                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Direct Usage (No Registry)        │
│                                     │
│  themis.evaluate(                   │
│      prompt=template.template      │
│  )                                  │
└─────────────────────────────────────┘
```

**Location**: `themis/generation/templates.py`
**Usage**: Pass directly to `evaluate(prompt=template.template)`

## Data Flow with Custom Components

```
┌──────────────────┐
│   User Request   │
│                  │
│  evaluate(       │
│    "my_dataset", │ ──┐
│    model="...",  │   │
│    metrics=[...] │   │
│  )               │   │
└──────────────────┘   │
                       │
                       ▼
        ┌──────────────────────────┐
        │  1. Resolve Dataset      │
        │                          │
        │  Registry lookup:        │
        │  "my_dataset" →          │
        │    create_my_dataset()   │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  2. Load Samples         │
        │                          │
        │  [                       │
        │    {"id": "1", ...},    │
        │    {"id": "2", ...},    │
        │  ]                       │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  3. Generate Responses   │
        │                          │
        │  For each sample:        │
        │    - Format prompt       │
        │    - Call provider       │
        │    - Store output        │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  4. Extract Answers      │
        │                          │
        │  extractor.extract(      │
        │    output                │
        │  ) → extracted_answer    │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  5. Compute Metrics      │
        │                          │
        │  For each metric:        │
        │    metric.compute(       │
        │      prediction,         │
        │      references          │
        │    ) → score             │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  6. Aggregate Results    │
        │                          │
        │  {                       │
        │    "my_metric": 0.85,   │
        │    "exact_match": 0.72, │
        │  }                       │
        └──────────────────────────┘
```

## Component Relationships

```
┌────────────────────────────────────────────────┐
│              Benchmark Preset                  │
│  (Bundles everything for easy use)            │
│                                                 │
│  ┌──────────────────────────────────────────┐ │
│  │  Dataset Loader                          │ │
│  │  (What to evaluate)                      │ │
│  └──────────────────────────────────────────┘ │
│                                                 │
│  ┌──────────────────────────────────────────┐ │
│  │  Prompt Template                         │ │
│  │  (How to format input)                   │ │
│  └──────────────────────────────────────────┘ │
│                                                 │
│  ┌──────────────────────────────────────────┐ │
│  │  Extractor                               │ │
│  │  (How to parse output)                   │ │
│  └──────────────────────────────────────────┘ │
│                                                 │
│  ┌──────────────────────────────────────────┐ │
│  │  Metrics                                 │ │
│  │  (How to score)                          │ │
│  └──────────────────────────────────────────┘ │
│                                                 │
└────────────────────────────────────────────────┘
```

## Registry Implementation Pattern

All registries follow the same pattern:

```python
# 1. Module-level registry
_REGISTRY = {}

# 2. Registration function
def register_component(name, factory):
    _REGISTRY[name] = factory

# 3. Creation function
def create_component(name, **options):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown: {name}")
    factory = _REGISTRY[name]
    return factory(options)

# 4. Query function
def list_components():
    return list(_REGISTRY.keys())
```

**Consistency**: All registries use this pattern for predictability

## When to Use Registration vs Direct Usage

### Use Registration When:

✅ You want to use components by name: `metrics=["my_metric"]`
✅ You're creating reusable components for multiple projects
✅ You want components available across your codebase
✅ You're building a library/framework on top of Themis

### Use Direct Usage When:

✅ One-off custom logic for a single evaluation
✅ Experimenting with different approaches
✅ Component is tightly coupled to specific evaluation
✅ You don't need named references

## Example: Full Custom Setup

```python
import themis
from dataclasses import dataclass
from themis.interfaces import Metric, Extractor
from themis.core.entities import MetricScore
from themis.presets import BenchmarkPreset
from themis.generation.templates import PromptTemplate

# 1. Custom Dataset
def create_qa_dataset(options):
    return [
        {"id": "q1", "question": "What is AI?", "answer": "Artificial Intelligence"},
        {"id": "q2", "question": "What is ML?", "answer": "Machine Learning"},
    ][:options.get("limit")]

themis.register_dataset("my_qa", create_qa_dataset)

# 2. Custom Metric
@dataclass
class ContainsKeyword(Metric):
    keyword: str = "learning"
    
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

themis.register_metric("contains_keyword", ContainsKeyword)

# 3. Custom Extractor (direct usage)
class UpperExtractor(Extractor):
    def extract(self, output, **context):
        return output.upper().strip()

# 4. Custom Benchmark (bundles everything)
preset = BenchmarkPreset(
    name="my_qa_benchmark",
    prompt_template=PromptTemplate(
        name="qa",
        template="Question: {question}\nAnswer:",
    ),
    metrics=[ContainsKeyword()],
    extractor=UpperExtractor(),
    dataset_loader=lambda limit: create_qa_dataset({"limit": limit}),
    reference_field="answer",
    dataset_id_field="id",
    description="My QA benchmark",
)

themis.register_benchmark(preset)

# 5. Use it all together!
report = themis.evaluate(
    "my_qa_benchmark",  # Uses registered benchmark
    model="gpt-4",
    limit=10,
)

# Or use components individually
report = themis.evaluate(
    "my_qa",  # Use registered dataset
    model="gpt-4",
    prompt="Q: {question}\nA:",
    metrics=["contains_keyword"],  # Use registered metric
    extractor=UpperExtractor(),  # Use extractor directly
)
```

## Summary

| Component | Registry? | Top-level API | Location |
|-----------|-----------|---------------|----------|
| **Metrics** | ✅ Yes | `themis.register_metric()` | `themis.api._METRICS_REGISTRY` |
| **Datasets** | ✅ Yes | `themis.register_dataset()` | `themis.datasets.registry._REGISTRY` |
| **Providers** | ✅ Yes | `themis.register_provider()` | `themis.providers.registry._REGISTRY` |
| **Benchmarks** | ✅ Yes | `themis.register_benchmark()` | `themis.presets.benchmarks._BENCHMARK_REGISTRY` |
| **Extractors** | ❌ No | Pass directly | `themis.evaluation.extractors/` |
| **Templates** | ❌ No | Pass directly | `themis.generation.templates` |

**Design Principle**: Components used by name need registration. Components passed as objects don't.

---

## Next Steps

1. **[Read the complete guide](EXTENDING_THEMIS.md)** - Detailed interfaces and examples
2. **[Check the quick reference](EXTENSION_QUICK_REFERENCE.md)** - One-page cheat sheet
3. **[Look at built-in examples](../themis/)** - Reference implementations
4. **[Try the example](../examples-simple/06_custom_metrics.py)** - Working custom metric example

---

**The goal**: Make it obvious where to add things and how to add them. If it's not obvious, that's a bug - please let us know!

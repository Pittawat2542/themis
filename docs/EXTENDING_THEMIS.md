# Extending Themis: Complete Guide

This guide shows you how to extend Themis with custom components. All extension points have clear APIs and are designed to be intuitive.

## Table of Contents

1. [Custom Metrics](#1-custom-metrics)
2. [Custom Datasets](#2-custom-datasets)
3. [Custom Model Providers](#3-custom-model-providers)
4. [Custom Benchmark Presets](#4-custom-benchmark-presets)
5. [Custom Extractors](#5-custom-extractors)
6. [Custom Prompt Templates](#6-custom-prompt-templates)
7. [Quick Reference](#quick-reference)

---

## 1. Custom Metrics

**What**: Define how to score model outputs (e.g., accuracy, BLEU, custom domain metrics)

**When**: You need evaluation criteria beyond built-in metrics like ExactMatch or MathVerify

**API**: `themis.register_metric(name, metric_class)`

### Interface

```python
from dataclasses import dataclass
from themis.interfaces import Metric
from themis.core.entities import MetricScore

@dataclass
class MyMetric(Metric):
    # Optional: configuration parameters
    threshold: float = 0.5
    
    def __post_init__(self):
        # Required: set metric name
        self.name = "my_metric"
    
    def compute(self, *, prediction, references=None, metadata=None) -> MetricScore:
        """Compute metric score.
        
        Args:
            prediction: Model's output (string)
            references: List of correct answers (optional)
            metadata: Additional info from dataset sample (optional)
            
        Returns:
            MetricScore with value between 0.0-1.0
        """
        # Your scoring logic
        score = self._compute_score(prediction, references, metadata)
        
        return MetricScore(
            metric_name=self.name,
            value=score,  # 0.0 to 1.0
            details={"any": "debugging info"},
            metadata=metadata or {},
        )
    
    def _compute_score(self, prediction, references, metadata) -> float:
        # Implementation
        return 1.0
```

### Example: Word Count Metric

```python
import themis
from dataclasses import dataclass
from themis.interfaces import Metric
from themis.core.entities import MetricScore

@dataclass
class WordCountMetric(Metric):
    min_words: int = 10
    
    def __post_init__(self):
        self.name = f"word_count_{self.min_words}"
    
    def compute(self, *, prediction, references=None, metadata=None):
        word_count = len(str(prediction).split())
        meets_requirement = word_count >= self.min_words
        
        return MetricScore(
            metric_name=self.name,
            value=1.0 if meets_requirement else 0.0,
            details={"word_count": word_count},
            metadata=metadata or {},
        )

# Register
themis.register_metric("word_count", WordCountMetric)

# Use
report = themis.evaluate(
    dataset,
    model="gpt-4",
    metrics=["word_count", "exact_match"],
)
```

### Built-in Metrics You Can Reference

Look at these for examples:
- `themis.evaluation.metrics.ExactMatch` - String matching
- `themis.evaluation.metrics.MathVerifyAccuracy` - Math answer verification
- `themis.evaluation.metrics.ResponseLength` - Length tracking
- `themis.evaluation.metrics.nlp.BLEU` - NLP metric example

---

## 2. Custom Datasets

**What**: Load your own evaluation datasets

**When**: You have custom data or want to add a new benchmark

**API**: `themis.register_dataset(name, factory_function)`

### Interface

```python
def create_my_dataset(options: dict) -> list[dict]:
    """Load dataset and return samples.
    
    Args:
        options: Configuration dict with keys like:
            - limit: Max samples to load
            - split: 'train', 'test', 'validation'
            - source: 'local', 'huggingface', etc.
            - Any custom options you define
    
    Returns:
        List of sample dicts with required fields:
            - id: Unique identifier (string)
            - question/problem/prompt: Input text
            - answer/solution: Reference answer
            - Any additional metadata fields
    """
    # Your loading logic
    samples = load_from_somewhere(
        limit=options.get("limit"),
        split=options.get("split", "test"),
    )
    
    # Convert to standard format
    return [
        {
            "id": sample.id,
            "question": sample.question,
            "answer": sample.answer,
            # Optional metadata
            "category": sample.category,
            "difficulty": sample.difficulty,
        }
        for sample in samples
    ]
```

### Example: JSON File Dataset

```python
import themis
import json
from pathlib import Path

def create_json_dataset(options: dict) -> list[dict]:
    """Load dataset from JSON file."""
    path = options.get("path", "data.json")
    limit = options.get("limit")
    
    with open(path) as f:
        data = json.load(f)
    
    samples = data["samples"][:limit] if limit else data["samples"]
    
    # Ensure standard format
    return [
        {
            "id": s.get("id", f"sample-{i}"),
            "question": s["question"],
            "answer": s["answer"],
            **{k: v for k, v in s.items() if k not in ["id", "question", "answer"]}
        }
        for i, s in enumerate(samples)
    ]

# Register
themis.register_dataset("my_json_dataset", create_json_dataset)

# Use
report = themis.evaluate(
    "my_json_dataset",  # Use by name
    model="gpt-4",
    prompt="Q: {question}\nA:",
    options={"path": "my_data.json", "limit": 100},
)
```

### Example: HuggingFace Dataset

```python
import themis
from datasets import load_dataset

def create_hf_dataset(options: dict) -> list[dict]:
    """Load from HuggingFace Hub."""
    dataset_name = options.get("dataset", "squad")
    split = options.get("split", "validation")
    limit = options.get("limit")
    
    ds = load_dataset(dataset_name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    return [
        {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answers"]["text"][0],  # First answer
            "context": item["context"],
        }
        for item in ds
    ]

themis.register_dataset("squad", create_hf_dataset)
```

### Built-in Datasets You Can Reference

Look at these for examples:
- `themis/datasets/math500.py` - HuggingFace loader
- `themis/datasets/gsm8k.py` - Simple dataset
- `themis/datasets/competition_math.py` - Multiple subsets

---

## 3. Custom Model Providers

**What**: Connect to custom LLM APIs or local models

**When**: You have a proprietary API or want to add a new provider

**API**: `themis.register_provider(name, factory_function)`

### Interface

```python
from themis.interfaces import ModelProvider

def create_my_provider(**options) -> ModelProvider:
    """Factory function that creates a provider instance.
    
    Args:
        **options: Configuration from user, e.g.:
            - api_key: Authentication
            - base_url: API endpoint
            - timeout: Request timeout
            - Any custom options
    
    Returns:
        ModelProvider instance
    """
    return MyProvider(
        api_key=options.get("api_key"),
        base_url=options.get("base_url", "https://api.example.com"),
        timeout=options.get("timeout", 30),
    )


class MyProvider(ModelProvider):
    """Custom provider implementation."""
    
    def __init__(self, api_key, base_url, timeout):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Generate response from the model.
        
        Args:
            model: Model identifier
            prompt: Input text
            temperature: Sampling temperature
            max_tokens: Max response length
            **kwargs: Additional generation options
        
        Returns:
            Generated text
        
        Raises:
            Exception: If generation fails
        """
        # Your API call
        response = self._call_api(
            endpoint=f"{self.base_url}/generate",
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["text"]
    
    def _call_api(self, **params):
        import requests
        response = requests.post(
            params["endpoint"],
            json={k: v for k, v in params.items() if k != "endpoint"},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
```

### Example: OpenAI-Compatible Provider

```python
import themis
from themis.interfaces import ModelProvider
import requests

def create_openai_compatible_provider(**options):
    return OpenAICompatibleProvider(
        base_url=options.get("base_url", "http://localhost:1234/v1"),
        api_key=options.get("api_key", "not-needed"),
    )


class OpenAICompatibleProvider(ModelProvider):
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
    
    def generate(self, model, prompt, temperature=0.0, max_tokens=512, **kwargs):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# Register
themis.register_provider("openai_compatible", create_openai_compatible_provider)

# Use
report = themis.evaluate(
    "gsm8k",
    model="openai_compatible/llama3",
    base_url="http://localhost:1234/v1",
    limit=10,
)
```

### Built-in Providers You Can Reference

Look at these for examples:
- `themis/generation/clients.py` - Fake provider for testing
- `themis/generation/providers/litellm_provider.py` - LiteLLM integration
- `themis/generation/providers/vllm_provider.py` - vLLM integration

---

## 4. Custom Benchmark Presets

**What**: Pre-configured evaluation setups (prompt + metrics + extractor + dataset)

**When**: You want a named benchmark that's easy to run: `themis.evaluate("my_benchmark", ...)`

**API**: `themis.register_benchmark(preset)`

### Interface

```python
from themis.presets import BenchmarkPreset
from themis.generation.templates import PromptTemplate

def create_my_benchmark_preset() -> BenchmarkPreset:
    """Create a complete benchmark configuration."""
    
    # 1. Define how to load the dataset
    def load_dataset(limit: int | None = None) -> list[dict]:
        from my_module import load_data
        samples = load_data()
        if limit:
            samples = samples[:limit]
        return [
            {
                "id": s.id,
                "question": s.question,
                "answer": s.answer,
            }
            for s in samples
        ]
    
    # 2. Define the prompt template
    prompt = PromptTemplate(
        name="my_benchmark",
        template="Question: {question}\n\nAnswer:",
    )
    
    # 3. Choose metrics
    from themis.evaluation.metrics import ExactMatch
    metrics = [ExactMatch()]
    
    # 4. Choose extractor
    from themis.evaluation.extractors import IdentityExtractor
    extractor = IdentityExtractor()
    
    # 5. Create preset
    return BenchmarkPreset(
        name="my_benchmark",
        prompt_template=prompt,
        metrics=metrics,
        extractor=extractor,
        dataset_loader=load_dataset,
        metadata_fields=("category", "difficulty"),  # Optional
        reference_field="answer",
        dataset_id_field="id",
        description="My custom benchmark for X",
    )
```

### Example: Custom Math Benchmark

```python
import themis
from themis.presets import BenchmarkPreset
from themis.generation.templates import PromptTemplate
from themis.evaluation.metrics import MathVerifyAccuracy
from themis.evaluation.extractors import MathVerifyExtractor

def create_my_math_preset():
    def load_my_math(limit=None):
        # Load your math problems
        problems = [
            {"id": "p1", "problem": "2+2=?", "answer": "4"},
            {"id": "p2", "problem": "10*5=?", "answer": "50"},
        ]
        return problems[:limit] if limit else problems
    
    return BenchmarkPreset(
        name="my_math",
        prompt_template=PromptTemplate(
            name="my_math",
            template="Solve: {problem}\n\nAnswer:",
        ),
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_my_math,
        reference_field="answer",
        dataset_id_field="id",
        description="My custom math problems",
    )

# Register
preset = create_my_math_preset()
themis.register_benchmark(preset)

# Use - now it's a named benchmark!
report = themis.evaluate("my_math", model="gpt-4", limit=10)
```

### Built-in Presets You Can Reference

Look at `themis/presets/benchmarks.py`:
- `_create_math500_preset()` - Math benchmark
- `_create_mmlu_pro_preset()` - Multiple choice
- `_create_demo_preset()` - Simple example

---

## 5. Custom Extractors

**What**: Parse/extract answers from model outputs before scoring

**When**: Model outputs need processing (e.g., extract number from text, parse JSON)

**Location**: No registry needed - create class and pass to evaluation

### Interface

```python
from themis.interfaces import Extractor

class MyExtractor(Extractor):
    """Extract structured data from model output."""
    
    def extract(self, output: str, **context) -> str:
        """Extract answer from raw output.
        
        Args:
            output: Raw model response
            **context: Additional info (metadata, etc.)
        
        Returns:
            Extracted answer (string)
        
        Raises:
            Exception: If extraction fails
        """
        # Your extraction logic
        return self._parse_output(output)
    
    def _parse_output(self, output: str) -> str:
        # Implementation
        return output.strip()
```

### Example: JSON Extractor

```python
from themis.interfaces import Extractor
import json
import re

class JSONAnswerExtractor(Extractor):
    """Extract answer from JSON response."""
    
    def extract(self, output: str, **context) -> str:
        # Try to find JSON in output
        json_match = re.search(r'\{[^}]+\}', output)
        if not json_match:
            return output.strip()
        
        try:
            data = json.loads(json_match.group())
            return data.get("answer", "")
        except json.JSONDecodeError:
            return output.strip()

# Use directly in evaluation
from themis import evaluate
from themis.evaluation.metrics import ExactMatch

report = evaluate(
    dataset,
    model="gpt-4",
    prompt="Return JSON: {question}",
    # Pass extractor directly (no registration needed)
    extractor=JSONAnswerExtractor(),
    metrics=[ExactMatch()],
)
```

### Example: Regex Extractor

```python
from themis.evaluation.extractors import RegexExtractor

# Built-in regex extractor
extractor = RegexExtractor(
    pattern=r"Answer: (.*)",
    group=1,
)

# Or extend it
class NumberExtractor(RegexExtractor):
    def __init__(self):
        super().__init__(
            pattern=r"[-+]?\d*\.?\d+",
            group=0,
        )

extractor = NumberExtractor()
```

### Built-in Extractors

- `IdentityExtractor` - Pass-through (no extraction)
- `RegexExtractor` - Regex-based extraction
- `MathVerifyExtractor` - Extract math answers from \\boxed{}
- `JSONFieldExtractor` - Extract field from JSON

Location: `themis/evaluation/extractors/`

---

## 6. Custom Prompt Templates

**What**: Reusable prompt formats

**When**: You want to standardize prompts across experiments

**Location**: No registry - create and use directly

### Interface

```python
from themis.generation.templates import PromptTemplate

template = PromptTemplate(
    name="my_template",
    template="Your template string with {variables}",
    metadata={"author": "me", "version": "1.0"},  # Optional
)

# Render with context
prompt_text = template.render(variables="values")

# Or use in evaluation
report = themis.evaluate(
    dataset,
    model="gpt-4",
    prompt=template.template,  # Use template string
)
```

### Example: Few-Shot Template

```python
from themis.generation.templates import PromptTemplate

few_shot_template = PromptTemplate(
    name="few_shot_math",
    template="""Solve the following math problems:

Example 1:
Q: What is 2 + 2?
A: 4

Example 2:
Q: What is 10 * 5?
A: 50

Now solve:
Q: {question}
A:""",
    metadata={"type": "few_shot", "examples": 2},
)

# Use
report = themis.evaluate(
    "gsm8k",
    model="gpt-4",
    prompt=few_shot_template.template,
    limit=100,
)
```

### Example: Chain-of-Thought Template

```python
cot_template = PromptTemplate(
    name="chain_of_thought",
    template="""Solve this step by step:

Problem: {problem}

Let's think through this:
1) First, I'll identify what's being asked
2) Then, I'll break down the problem
3) Finally, I'll compute the answer

Solution:""",
)
```

---

## Quick Reference

### All Registration APIs

```python
import themis

# Metrics
themis.register_metric(name, metric_class)
themis.get_registered_metrics()

# Datasets
themis.register_dataset(name, factory_function)
themis.list_datasets()
themis.is_dataset_registered(name)

# Providers
themis.register_provider(name, factory_function)

# Benchmarks
themis.register_benchmark(preset)
themis.list_benchmarks()
themis.get_benchmark_preset(name)
```

### What Needs Registration vs What Doesn't

| Component | Needs Registration? | Why |
|-----------|-------------------|-----|
| **Metrics** | ✅ Yes | To use by name in `metrics=["my_metric"]` |
| **Datasets** | ✅ Yes | To use by name in `evaluate("my_dataset")` |
| **Providers** | ✅ Yes | To use by name in `model="my_provider/model"` |
| **Benchmarks** | ✅ Yes | To bundle dataset + prompt + metrics as named config |
| **Extractors** | ❌ No | Pass directly: `extractor=MyExtractor()` |
| **Templates** | ❌ No | Use directly: `prompt=template.template` |

### File Locations for Examples

```
themis/
├── evaluation/
│   ├── metrics/
│   │   ├── exact_match.py          # Metric example
│   │   ├── math_verify_accuracy.py  # Metric example
│   │   └── nlp/bleu.py              # NLP metric example
│   └── extractors/
│       ├── identity_extractor.py    # Extractor example
│       ├── regex_extractor.py       # Extractor example
│       └── math_verify_extractor.py # Extractor example
├── datasets/
│   ├── registry.py                  # Dataset registry
│   ├── math500.py                   # Dataset example
│   └── gsm8k.py                     # Dataset example
├── providers/
│   └── registry.py                  # Provider registry
├── generation/
│   ├── providers/
│   │   └── litellm_provider.py     # Provider example
│   └── templates.py                 # Template class
└── presets/
    └── benchmarks.py                # Benchmark presets
```

---

## Best Practices

### 1. Naming Conventions

- **Metrics**: Use snake_case: `"word_count"`, `"f1_score"`
- **Datasets**: Use kebab-case: `"my-dataset"`, `"custom-qa"`
- **Providers**: Use lowercase: `"openai"`, `"anthropic"`
- **Benchmarks**: Use lowercase: `"math500"`, `"my_benchmark"`

### 2. Return Value Ranges

- **Metrics**: Return 0.0-1.0 (0% to 100%)
- Higher is better by convention
- Use `details={}` for debugging info

### 3. Error Handling

- Raise descriptive exceptions
- Don't silently fail
- Log warnings for non-critical issues

### 4. Documentation

- Add docstrings to all public functions
- Include usage examples
- Document all options in factory functions

### 5. Testing

```python
# Test your custom component
import themis

# Register
themis.register_metric("my_metric", MyMetric)

# Test with small dataset
report = themis.evaluate(
    [{"id": "1", "question": "test", "answer": "test"}],
    model="fake",
    prompt="Q: {question}\nA:",
    metrics=["my_metric"],
)

# Check results
assert "my_metric" in report.evaluation_report.metrics
```

---

## Complete Example: Adding Everything

Here's a complete example that adds a custom dataset, metric, extractor, and benchmark:

```python
import themis
from dataclasses import dataclass
from themis.interfaces import Metric, Extractor
from themis.core.entities import MetricScore
from themis.presets import BenchmarkPreset
from themis.generation.templates import PromptTemplate

# 1. Custom Dataset
def create_my_qa_dataset(options):
    return [
        {"id": "q1", "question": "What is AI?", "answer": "Artificial Intelligence"},
        {"id": "q2", "question": "What is ML?", "answer": "Machine Learning"},
    ][:options.get("limit")]

themis.register_dataset("my_qa", create_my_qa_dataset)

# 2. Custom Metric
@dataclass
class LengthRatioMetric(Metric):
    def __post_init__(self):
        self.name = "length_ratio"
    
    def compute(self, *, prediction, references=None, metadata=None):
        if not references:
            return MetricScore(self.name, 0.0, {}, {})
        
        pred_len = len(prediction)
        ref_len = len(references[0])
        ratio = min(pred_len, ref_len) / max(pred_len, ref_len)
        
        return MetricScore(
            metric_name=self.name,
            value=ratio,
            details={"pred_len": pred_len, "ref_len": ref_len},
            metadata=metadata or {},
        )

themis.register_metric("length_ratio", LengthRatioMetric)

# 3. Custom Extractor
class UppercaseExtractor(Extractor):
    def extract(self, output, **context):
        return output.upper().strip()

# 4. Custom Benchmark
def create_my_qa_preset():
    return BenchmarkPreset(
        name="my_qa_benchmark",
        prompt_template=PromptTemplate(
            name="qa",
            template="Q: {question}\nA:",
        ),
        metrics=[LengthRatioMetric()],
        extractor=UppercaseExtractor(),
        dataset_loader=lambda limit: create_my_qa_dataset({"limit": limit}),
        reference_field="answer",
        dataset_id_field="id",
        description="My custom Q&A benchmark",
    )

themis.register_benchmark(create_my_qa_preset())

# Now use it all together!
report = themis.evaluate(
    "my_qa_benchmark",  # Use the preset
    model="gpt-4",
    limit=10,
)

print(f"Length Ratio: {report.evaluation_report.metrics['length_ratio'].mean:.2%}")
```

---

## Getting Help

- **Examples**: Check `examples/` directory for working code
- **Built-ins**: Look at `themis/evaluation/`, `themis/datasets/` for reference implementations
- **Issues**: Open an issue on GitHub for questions
- **Docs**: See other guides in `docs/` directory

---

**Remember**: The goal is to make extension obvious and easy. If something isn't clear, that's a bug in our design - please let us know!

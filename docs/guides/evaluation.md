# Evaluation Guide

This guide covers advanced evaluation features in Themis, including custom metrics, reference handling, and extractors.

## Table of Contents

1. [Multi-Value References](#multi-value-references)
2. [Custom Reference Selectors](#custom-reference-selectors)
3. [Extractor Contract](#extractor-contract)
4. [Custom Metrics](#custom-metrics)
5. [Best Practices](#best-practices)

---

## Multi-Value References

References can now hold complex data structures using dict values, perfect for tasks requiring multiple reference values.

### Basic Example

```python
from themis.core.entities import Reference

# Simple reference
ref = Reference(kind="answer", value="42")

# Multi-value reference using dict
ref = Reference(
    kind="countdown_task",
    value={
        "target": 122,
        "numbers": [25, 50, 75, 100]
    }
)

# List reference
ref = Reference(
    kind="valid_answers",
    value=["yes", "no", "maybe"]
)
```

### Using in Tasks

```python
from themis.core.entities import GenerationTask, PromptRender, PromptSpec, Reference

task = GenerationTask(
    prompt=PromptRender(
        spec=PromptSpec(
            name="countdown",
            template="Using {numbers_str}, make {target}"
        ),
        text="Using 25, 50, 75, 100, make 122",
        context={"numbers_str": "25, 50, 75, 100", "target": 122}
    ),
    model=model_spec,
    sampling=sampling_config,
    reference=Reference(
        kind="countdown",
        value={"target": 122, "numbers": [25, 50, 75, 100]}
    ),
    metadata={"numbers": [25, 50, 75, 100]}  # Also in metadata for reference selector
)
```

### Accessing in Metrics

```python
from themis.interfaces import Metric
from themis.core import entities

class CountdownAccuracy(Metric):
    name = "countdown_accuracy"
    
    def compute(self, *, prediction, references, metadata=None):
        # references is a list (normalized by pipeline)
        ref = references[0]
        
        if isinstance(ref, dict):
            # Multi-value reference
            target = ref["target"]
            numbers = ref["numbers"]
        else:
            # Fallback to metadata
            target = ref
            numbers = metadata.get("numbers", [])
        
        # Validate prediction uses only allowed numbers
        is_valid = self.validate_expression(prediction, numbers, target)
        
        return entities.MetricScore(
            metric_name=self.name,
            value=1.0 if is_valid else 0.0,
            details={"target": target, "numbers": numbers}
        )
```

---

## Custom Reference Selectors

Custom reference selectors allow you to extract and transform reference data before metrics receive it.

### Basic Usage

```python
from themis.evaluation import EvaluationPipeline

def my_reference_selector(record):
    """Extract reference from task metadata."""
    return {
        "target": record.task.reference.value,
        "numbers": record.task.metadata.get("numbers", [])
    }

pipeline = EvaluationPipeline(
    extractor=my_extractor,
    metrics=[my_metric],
    reference_selector=my_reference_selector
)
```

**Important:** Custom reference selectors take precedence over the default behavior. You'll see a warning if using with `DefaultEvaluationStrategy` - this is normal and the selector will work correctly.

### Reference Selector Patterns

**Pattern 1: Multi-Field Extraction**
```python
def extract_multi_field_reference(record):
    """Combine multiple metadata fields into reference dict."""
    return {
        "answer": record.task.reference.value,
        "explanation": record.task.metadata.get("explanation"),
        "difficulty": record.task.metadata.get("difficulty")
    }
```

**Pattern 2: Conditional References**
```python
def conditional_reference(record):
    """Select reference based on task type."""
    task_type = record.task.metadata.get("type")
    
    if task_type == "multiple_choice":
        return record.task.metadata.get("correct_option")
    elif task_type == "math":
        return {
            "answer": record.task.reference.value,
            "steps": record.task.metadata.get("steps")
        }
    else:
        return record.task.reference.value
```

**Pattern 3: Multiple Valid Answers**
```python
def multiple_answers_reference(record):
    """Return list of valid answers."""
    primary = record.task.reference.value
    alternatives = record.task.metadata.get("alternative_answers", [])
    return [primary] + alternatives
```

### Precedence Rules

The evaluation pipeline uses this precedence order:

1. **Custom reference_selector** (if provided) - Always takes precedence
2. **item.reference** (from evaluation strategy)
3. **Default reference selector** (extracts from task.reference)

```python
# Custom selector ALWAYS takes precedence
pipeline = EvaluationPipeline(
    extractor=extractor,
    metrics=[metric],
    reference_selector=my_custom_selector  # Will be used
)
```

---

## Extractor Contract

Understanding the extractor contract prevents common bugs and ensures metrics work correctly.

### What Extractors Do

Extractors parse raw model output and extract the relevant answer:

```
Raw Output (from model):
"<think>Let me solve this... 2+2=4</think><answer>4</answer>"

↓ Extractor processes ↓

Extracted Output (to metric):
"4"
```

### Metric Receives Extracted Output

**Critical: Metrics receive EXTRACTED output, not raw text!**

```python
from themis.interfaces import Metric

class MyMetric(Metric):
    name = "my_metric"
    
    def compute(self, *, prediction, references, metadata=None):
        # ✅ CORRECT: prediction is already extracted
        # prediction = "4" (not "<think>...</think><answer>4</answer>")
        is_correct = prediction == references[0]
        
        # ❌ WRONG: Don't try to extract again!
        # answer = self.extract_answer(prediction)  # DON'T DO THIS
        
        return MetricScore(
            metric_name=self.name,
            value=1.0 if is_correct else 0.0
        )
```

### Pipeline Flow

```
1. Model generates: "<think>reasoning</think><answer>42</answer>"
2. Extractor extracts: "42"
3. Metric receives: prediction="42" (ALREADY EXTRACTED)
```

### Common Extractor Types

**JSON Field Extractor:**
```python
from themis.evaluation.extractors import JsonFieldExtractor

extractor = JsonFieldExtractor("answer")
# Input: '{"answer": "42", "explanation": "..."}'
# Output: "42"
```

**Regex Extractor:**
```python
from themis.evaluation.extractors import RegexExtractor

extractor = RegexExtractor(r"<answer>(.*?)</answer>")
# Input: "<think>...</think><answer>42</answer>"
# Output: "42"
```

**Identity Extractor:**
```python
from themis.evaluation.extractors import IdentityExtractor

extractor = IdentityExtractor()
# Input: "42"
# Output: "42" (no transformation)
```

### Creating Custom Extractors

```python
from themis.evaluation.extractors import FieldExtractionError

class MyExtractor:
    def extract(self, raw_output: str):
        """Extract answer from custom format."""
        if "ANSWER:" in raw_output:
            parts = raw_output.split("ANSWER:")
            return parts[-1].strip()
        raise FieldExtractionError("No ANSWER: marker found")

# Use in pipeline
pipeline = EvaluationPipeline(
    extractor=MyExtractor(),
    metrics=[my_metric]
)
```

---

## Custom Metrics

### Basic Metric Template

```python
from themis.interfaces import Metric
from themis.core import entities

class MyMetric(Metric):
    name = "my_metric"
    requires_reference = True  # Set False if no reference needed
    
    def compute(self, *, prediction, references, metadata=None):
        """Compute metric score.
        
        Args:
            prediction: Extracted answer (str, int, dict, etc.)
            references: List of reference values (always a list)
            metadata: Dict with sample_id and task metadata
            
        Returns:
            MetricScore with value and optional details
        """
        # Your logic here
        score_value = self._calculate_score(prediction, references)
        
        return entities.MetricScore(
            metric_name=self.name,
            value=score_value,
            details={"debug_info": "..."},
            metadata={"processing_time_ms": 10}
        )
```

### Example: Exact Match with Multiple Valid Answers

```python
class MultiAnswerExactMatch(Metric):
    name = "multi_answer_exact_match"
    
    def compute(self, *, prediction, references, metadata=None):
        # Check if prediction matches any reference
        prediction_clean = prediction.strip().lower()
        
        is_correct = any(
            prediction_clean == str(ref).strip().lower()
            for ref in references
        )
        
        return entities.MetricScore(
            metric_name=self.name,
            value=1.0 if is_correct else 0.0,
            details={
                "prediction": prediction,
                "valid_answers": references,
                "matched": is_correct
            }
        )
```

### Example: Math Evaluation with Steps

```python
class MathWithSteps(Metric):
    name = "math_with_steps"
    
    def compute(self, *, prediction, references, metadata=None):
        ref = references[0]
        
        if isinstance(ref, dict):
            expected_answer = ref["answer"]
            expected_steps = ref.get("steps", [])
        else:
            expected_answer = ref
            expected_steps = []
        
        # Check answer correctness
        answer_correct = self._check_answer(prediction, expected_answer)
        
        # Check if solution includes expected steps
        steps_correct = all(
            step in prediction for step in expected_steps
        )
        
        return entities.MetricScore(
            metric_name=self.name,
            value=1.0 if (answer_correct and steps_correct) else 0.0,
            details={
                "answer_correct": answer_correct,
                "steps_correct": steps_correct,
                "expected_steps": expected_steps
            }
        )
```

---

## Best Practices

### 1. Use Multi-Value References for Complex Tasks

```python
# ✅ GOOD: All data in reference
reference = Reference(
    kind="task",
    value={
        "answer": 42,
        "constraints": [1, 2, 3],
        "format": "integer"
    }
)

# ❌ AVOID: Scattered across metadata
reference = Reference(kind="answer", value=42)
metadata = {"constraints": [1, 2, 3], "format": "integer"}
```

### 2. Don't Re-Extract in Metrics

```python
# ✅ CORRECT: Use prediction directly
def compute(self, *, prediction, references, metadata=None):
    return MetricScore(
        metric_name=self.name,
        value=1.0 if prediction == references[0] else 0.0
    )

# ❌ WRONG: Trying to extract again
def compute(self, *, prediction, references, metadata=None):
    # DON'T DO THIS - prediction is already extracted!
    answer = extract_from_tags(prediction)  
    return MetricScore(...)
```

### 3. Handle Multiple Reference Formats Gracefully

```python
def compute(self, *, prediction, references, metadata=None):
    ref = references[0]
    
    # Handle both dict and scalar references
    if isinstance(ref, dict):
        answer = ref["answer"]
        extra_data = ref.get("extra", None)
    else:
        answer = ref
        extra_data = None
    
    # Use answer and extra_data...
```

### 4. Provide Detailed Error Information

```python
def compute(self, *, prediction, references, metadata=None):
    try:
        result = self._complex_validation(prediction)
        return MetricScore(
            metric_name=self.name,
            value=result.score,
            details={
                "validation_details": result.details,
                "sample_id": metadata.get("sample_id")
            }
        )
    except Exception as e:
        # Return score with error details
        return MetricScore(
            metric_name=self.name,
            value=0.0,
            details={
                "error": str(e),
                "prediction": prediction[:100]  # First 100 chars
            }
        )
```

### 5. Test with Edge Cases

```python
# Test your metrics with:
# - Empty predictions
# - Missing references
# - Malformed data
# - Multiple valid answers
# - Edge cases in your domain

def test_metric_with_edge_cases():
    metric = MyMetric()
    
    # Empty prediction
    score = metric.compute(prediction="", references=["42"])
    assert score.value == 0.0
    
    # Multiple valid answers
    score = metric.compute(prediction="yes", references=["yes", "y", "true"])
    assert score.value == 1.0
```

---

For more examples, see:
- [`examples/advanced/`](../examples/advanced/) - Custom metrics and evaluation strategies
- [`tests/evaluation/`](../tests/evaluation/) - Comprehensive test suite
- [`themis/evaluation/metrics/`](../themis/evaluation/metrics/) - Built-in metric implementations

# Metrics API

Evaluation metrics for different domains.

## Overview

Themis includes metrics for:
- **Math**: ExactMatch, MathVerify
- **NLP**: BLEU, ROUGE, BERTScore, METEOR
- **Code**: PassAtK, CodeBLEU, ExecutionAccuracy

## Base Interface

### Metric

All metrics inherit from this abstract base:

```python
from themis.evaluation.metrics import Metric, MetricScore

class CustomMetric(Metric):
    @property
    def name(self) -> str:
        return "MyMetric"
    
    def evaluate(self, response: str, reference: str) -> MetricScore:
        # Your evaluation logic
        score = compute_score(response, reference)
        return MetricScore(value=score, metadata={})
```

---

## Math Metrics

### ExactMatch

Exact string matching after normalization.

```python
from themis.evaluation.metrics.math import ExactMatch

metric = ExactMatch()
score = metric.evaluate("4", "4")
print(score.value)  # 1.0
```

**Normalization:**
- Strip whitespace
- Lowercase
- Remove punctuation

---

### MathVerify

Symbolic and numeric math verification.

```python
from themis.evaluation.metrics.math import MathVerify

metric = MathVerify()
score = metric.evaluate("2.0", "2")  # Numerically equal
print(score.value)  # 1.0
```

**Features:**
- Symbolic equivalence
- Numeric tolerance
- LaTeX parsing

---

## NLP Metrics

Requires: `pip install themis-eval[nlp]`

### BLEU

N-gram precision metric.

```python
from themis.evaluation.metrics.nlp import BLEU

metric = BLEU()
score = metric.evaluate(
    response="The cat is on the mat",
    reference="The cat is on the mat",
)
print(score.value)  # High score (similar)
```

---

### ROUGE

Recall-oriented metric.

```python
from themis.evaluation.metrics.nlp import ROUGE

metric = ROUGE()
score = metric.evaluate(response, reference)
print(score.metadata)  # Contains ROUGE-1, ROUGE-2, ROUGE-L
```

---

### BERTScore

Semantic similarity using BERT embeddings.

```python
from themis.evaluation.metrics.nlp import BERTScore

metric = BERTScore()
score = metric.evaluate(
    response="The cat sits on the mat",
    reference="A feline rests on the rug",
)
print(score.value)  # Semantic similarity
```

---

### METEOR

Alignment-based metric with synonyms.

```python
from themis.evaluation.metrics.nlp import METEOR

metric = METEOR()
score = metric.evaluate(response, reference)
```

---

## Code Metrics

Requires: `pip install themis-eval[code]`

### PassAtK

Pass rate for K samples.

```python
from themis.evaluation.metrics.code import PassAtK

metric = PassAtK(k=10)

# Evaluate multiple samples
scores = []
for sample in samples:
    score = metric.evaluate(sample, reference)
    scores.append(score.value)

pass_rate = sum(scores) / len(scores)
```

**Use with:**
```python
result = evaluate(
    benchmark="humaneval",
    model="gpt-4",
    num_samples=10,  # Generate 10 samples
    metrics=["PassAtK"],
)
```

---

### CodeBLEU

BLEU adapted for code with syntax awareness.

```python
from themis.evaluation.metrics.code import CodeBLEU

metric = CodeBLEU(lang="python")
score = metric.evaluate(response_code, reference_code)
```

---

### ExecutionAccuracy

Functional correctness through execution.

```python
from themis.evaluation.metrics.code import ExecutionAccuracy

metric = ExecutionAccuracy()
score = metric.evaluate(
    response=code_solution,
    reference=test_cases,
)
```

---

## Custom Metrics

Implement your own metric:

```python
from themis.evaluation.metrics import Metric, MetricScore

class LengthRatio(Metric):
    """Metric that compares response length to reference."""
    
    @property
    def name(self) -> str:
        return "LengthRatio"
    
    def evaluate(self, response: str, reference: str) -> MetricScore:
        if not reference:
            return MetricScore(value=0.0)
        
        ratio = len(response) / len(reference)
        
        # Score: 1.0 if within 20% of reference length
        if 0.8 <= ratio <= 1.2:
            score = 1.0
        else:
            score = max(0.0, 1.0 - abs(ratio - 1.0))
        
        return MetricScore(
            value=score,
            metadata={"response_len": len(response), "reference_len": len(reference)}
        )

# Use in evaluation
result = evaluate(
    dataset=my_dataset,
    model="gpt-4",
    metrics=[LengthRatio(), "ExactMatch"],
)
```

---

## MetricScore

Result of a metric evaluation.

```python
@dataclass
class MetricScore:
    value: float              # Numeric score (0.0 to 1.0 typically)
    metadata: dict = {}       # Additional information
```

**Example:**
```python
score = MetricScore(
    value=0.85,
    metadata={
        "precision": 0.9,
        "recall": 0.8,
        "f1": 0.85
    }
)
```

---

## Using Metrics

### In evaluate()

```python
from themis import evaluate

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "MathVerify", "BLEU"],
)

print(result.metrics)
# {'ExactMatch': 0.85, 'MathVerify': 0.87, 'BLEU': 0.72}
```

### Standalone

```python
from themis.evaluation.metrics.math import ExactMatch

metric = ExactMatch()

responses = ["4", "7", "12"]
references = ["4", "8", "12"]

for resp, ref in zip(responses, references):
    score = metric.evaluate(resp, ref)
    print(f"{resp} vs {ref}: {score.value}")
```

---

## Best Practices

### 1. Choose Appropriate Metrics

- **Exact tasks**: Use `ExactMatch`
- **Math**: Use `MathVerify`
- **Text generation**: Use `BLEU`, `ROUGE`, `BERTScore`
- **Code**: Use `PassAtK`, `CodeBLEU`

### 2. Combine Multiple Metrics

Don't rely on a single metric:

```python
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    metrics=["ExactMatch", "MathVerify", "BLEU"],
)
```

### 3. Understand Metric Limitations

- `ExactMatch`: Sensitive to formatting
- `BLEU`: Only measures n-gram overlap
- `BERTScore`: Requires model download (slow first time)
- `PassAtK`: Needs multiple samples

### 4. Test Metrics First

```python
# Test your metric
metric = CustomMetric()
score = metric.evaluate("test response", "test reference")
print(f"Score: {score.value}")
print(f"Metadata: {score.metadata}")
```

---

## Metric Properties

### Score Range

Most metrics return scores in [0.0, 1.0]:
- 0.0 = Completely incorrect
- 1.0 = Perfect match

Some metrics (BLEU, ROUGE) may use different ranges—check documentation.

### Metadata

Metrics can include additional information:

```python
score = metric.evaluate(response, reference)
print(score.value)     # Main score
print(score.metadata)  # Additional details
```

---

## See Also

- [Evaluation Guide](../guides/evaluation.md) - Using metrics in practice
- [Custom Metrics Tutorial](../EVALUATION.md) - Implement your own
- [Comparison Guide](../COMPARISON.md) - Compare across metrics

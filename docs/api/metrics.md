# Metrics API

Evaluation metrics for different domains.

## Overview

Themis includes metrics for:
- **Core**: `ExactMatch`, `ResponseLength`
- **Math**: `MathVerifyAccuracy`
- **NLP**: `BLEU`, `ROUGE`, `BERTScore`, `METEOR`
- **Code**: `PassAtK`, `ExecutionAccuracy`, `CodeBLEU`

Metric names used in reports are taken from each metric's `name` attribute (e.g., `"ExactMatch"`).

## Dependencies and optional extras

- **MathVerifyAccuracy** requires `themis-eval[math]` (package: `math-verify`).
- **NLP metrics** require `themis-eval[nlp]` (`sacrebleu`, `rouge-score`, `bert-score`, `nltk`).
- **CodeBLEU** requires `themis-eval[code]` (package: `codebleu`).

Install extras:
```bash
uv add "themis-eval[math,nlp,code]"
```

## Base Interface

### Metric

All metrics implement `themis.interfaces.Metric`:

```python
from themis.core.entities import MetricScore
from themis.interfaces import Metric

class CustomMetric(Metric):
    name = "MyMetric"

    def compute(self, *, prediction, references, metadata=None) -> MetricScore:
        score = 1.0 if prediction in references else 0.0
        return MetricScore(metric_name=self.name, value=score)
```

Notes:
- `prediction` is already extracted by the pipeline's extractor.
- `references` is a list of normalized reference values.

---

## Core Metrics

### ExactMatch

Exact string matching after normalization.

```python
from themis.evaluation.metrics import ExactMatch

metric = ExactMatch()
score = metric.compute(prediction="4", references=["4"])
print(score.value)  # 1.0
```

### ResponseLength

Length-based metric useful for detecting verbosity or truncation.

```python
from themis.evaluation.metrics import ResponseLength

metric = ResponseLength()
score = metric.compute(prediction="short answer", references=[""])
print(score.value)
```

---

## Math Metrics

### MathVerifyAccuracy

Symbolic and numeric math verification.

```python
from themis.evaluation.metrics import MathVerifyAccuracy

metric = MathVerifyAccuracy()
score = metric.compute(prediction="2.0", references=["2"])
print(score.value)  # 1.0
```

---

## NLP Metrics

Requires: `uv add "themis-eval[nlp]"`

### BLEU

```python
from themis.evaluation.metrics.nlp import BLEU

metric = BLEU()
score = metric.compute(prediction="The cat is on the mat", references=["The cat is on the mat"])
print(score.value)
```

### ROUGE

```python
from themis.evaluation.metrics.nlp import ROUGE, ROUGEVariant

metric = ROUGE(variant=ROUGEVariant.ROUGE_L)
score = metric.compute(prediction="A", references=["A"])
print(score.value)
```

### BERTScore

```python
from themis.evaluation.metrics.nlp import BERTScore

metric = BERTScore()
score = metric.compute(prediction="The cat sits on the mat", references=["A feline rests on the rug"])
print(score.value)
```

### METEOR

```python
from themis.evaluation.metrics.nlp import METEOR

metric = METEOR()
score = metric.compute(prediction="A", references=["A"])
```

---

## Code Metrics

Requires: `uv add "themis-eval[code]"`

### PassAtK

```python
from themis.evaluation.metrics.code.pass_at_k import PassAtK

metric = PassAtK(k=10)
score = metric.compute(prediction="code", references=["expected"])
```

### ExecutionAccuracy

```python
from themis.evaluation.metrics.code.execution import ExecutionAccuracy

metric = ExecutionAccuracy()
score = metric.compute(prediction="code", references=["expected"])
```

### CodeBLEU

```python
from themis.evaluation.metrics.code.codebleu import CodeBLEU

metric = CodeBLEU()
score = metric.compute(prediction="code", references=["expected"])
```

---

## Using metrics in `evaluate()`

```python
from themis import evaluate

report = evaluate(
    "gsm8k",
    model="gpt-4",
    metrics=["exact_match", "math_verify"],
)
```

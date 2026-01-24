# Themis Simple Examples

These examples demonstrate the new simplified API introduced in Themis 2.0.

## Quick Start

Run your first evaluation:

```bash
cd examples-simple
python 01_quickstart.py
```

## Examples

### 01_quickstart.py
Your first evaluation in 10 lines. Shows how to run a benchmark with minimal code.

```python
import themis
report = themis.evaluate("demo", model="fake-math-llm", limit=3)
```

### 02_custom_dataset.py
Evaluate on your own data. Shows how to use custom datasets with custom prompts.

```python
dataset = [{"id": "q1", "question": "...", "answer": "..."}]
report = themis.evaluate(dataset=dataset, model="gpt-4", prompt="Q: {question}\nA:")
```

### 03_distributed.py
Scale with distributed execution (Phase 3). Shows how to run evaluations across multiple workers.

```python
report = themis.evaluate(
    "gsm8k",
    model="gpt-4",
    distributed=True,
    workers=8,
    storage="s3://my-bucket/runs"
)
```

### 06_custom_metrics.py
Define and register custom evaluation metrics. Shows how to create your own metrics beyond the built-in ones.

```python
# Define a custom metric
@dataclass
class WordCountMetric(Metric):
    min_words: int = 10
    
    def compute(self, *, prediction, references=None, metadata=None):
        word_count = len(str(prediction).split())
        return MetricScore(
            metric_name=self.name,
            value=1.0 if word_count >= self.min_words else 0.0,
            details={"word_count": word_count},
            metadata=metadata or {},
        )

# Register and use it
themis.register_metric("word_count", WordCountMetric)
report = themis.evaluate(dataset, model="gpt-4", metrics=["word_count"])
```

## Running with Real Models

To use real LLM providers (not the fake model), set your API keys:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Then run with real models
python -c "
import themis
report = themis.evaluate('demo', model='gpt-4', limit=3)
print(report.evaluation_report.aggregates)
"
```

## Next Steps

- See full documentation: https://pittawat2542.github.io/themis/
- Run CLI evaluations: `themis eval demo --model gpt-4`
- Explore benchmarks: `themis list benchmarks`

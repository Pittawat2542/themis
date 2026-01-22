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

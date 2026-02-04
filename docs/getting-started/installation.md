# Installation

## Requirements

- Python 3.12+

## Install

```bash
pip install themis-eval
```

With optional extras:

```bash
pip install "themis-eval[math,nlp,code,server]"
```

Using `uv`:

```bash
uv pip install themis-eval
uv pip install "themis-eval[math,nlp,code,server]"
```

## Verify

```bash
themis demo --limit 3
```

Or Python:

```python
from themis import evaluate

report = evaluate("demo", model="fake-math-llm", limit=3)
print(len(report.generation_results))
```

## Optional Provider Setup

Set provider keys only when using real hosted models:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

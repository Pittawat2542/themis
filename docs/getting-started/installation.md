# Installation

## Requirements

- Python 3.12+

## Install

```bash
uv add themis-eval
```

With optional extras:

```bash
uv add "themis-eval[math,nlp,code,server]"
```

## Contributor Environment (Recommended)

For local development and full validation (tests + docs + server + metrics extras):

```bash
uv sync --extra dev --extra docs --extra server --extra nlp --extra code --extra math
```

## Verify

```bash
# From a source checkout
uv run python -m themis.cli demo --limit 3

# If installed as a package with console scripts
themis demo --limit 3
```

Or Python:

```python
from themis import evaluate

report = evaluate("demo", model="fake-math-llm", limit=3)
print(len(report.generation_results))
```

Validate docs build:

```bash
uv run python -m mkdocs build --strict
```

## Optional Provider Setup

Set provider keys only when using real hosted models:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

# Providers Guide

Themis routes generation via provider + model.

## Model String Formats

- Auto-detected model name: `"gpt-4"`, `"claude-3-opus-20240229"`
- Canonical key: `"provider:model_id"` (recommended for vNext specs)

Examples:

```python
# Auto-detected provider (litellm)
evaluate("gsm8k", model="gpt-4")

# Explicit provider:model_id for ExperimentSpec
spec = ExperimentSpec(..., model="litellm:gpt-4", ...)
```

## Fake Provider

Use fake models for local smoke tests without credentials:

```python
evaluate("demo", model="fake-math-llm", limit=5)
# or
spec = ExperimentSpec(..., model="fake:fake-math-llm", ...)
```

## Credentials

Set API keys only for real providers:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

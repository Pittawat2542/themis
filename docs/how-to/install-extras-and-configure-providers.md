---
title: Install extras and configure providers
diataxis: how-to
audience: users adopting provider-backed integrations
goal: Show how to install optional extras and prepare provider-backed workflows safely.
---

# Install extras and configure providers

Goal: install optional dependencies and set up provider-backed adapters.

When to use this:

Use this guide before adopting OpenAI, vLLM, LangGraph, Hugging Face dataset loading, or external store backends.

## Procedure

Install only the extras you need:

```bash
uv pip install -e ".[openai]"
uv pip install -e ".[langgraph]"
uv pip install -e ".[datasets]"
```

Provider specifics:

- OpenAI: install `.[openai]` and either inject a client or provide credentials to the adapter
- vLLM: use Linux where the `vllm` package is supported, or inject an OpenAI-compatible client
- LangGraph: install `.[langgraph]` and pass a graph with `invoke()` or `ainvoke()`
- Hugging Face quick-eval: install `.[datasets]`

## Variants

- deterministic docs/test examples: inject fake clients or use builtin demo components
- production-style provider flows: install the matching extras and point adapters at the real service

## Expected result

You should know which extra package to install and whether a real provider is required for the workflow you want to run.

## Troubleshooting

- [Configure generators](configure-generators.md)
- [Adapters reference](../reference/adapters.md)

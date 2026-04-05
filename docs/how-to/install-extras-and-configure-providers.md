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
uv add "themis-eval[openai]"
uv add "themis-eval[vllm]"  # Linux only
uv add "themis-eval[langgraph]"
uv add "themis-eval[datasets]"
uv add "themis-eval[mongodb]"
uv add "themis-eval[postgres]"
```

Provider specifics:

- OpenAI: install `themis-eval[openai]` and either inject a client or provide credentials to the adapter
- vLLM: install `themis-eval[vllm]` on Linux, or inject an OpenAI-compatible client when you cannot install the Linux-only dependency
- LangGraph: install `themis-eval[langgraph]` and pass a graph with `invoke()` or `ainvoke()`
- Hugging Face quick-eval: install `themis-eval[datasets]`
- MongoDB store: install `themis-eval[mongodb]` before using the `mongodb` backend
- Postgres store: install `themis-eval[postgres]` before using the `postgres` backend

If you only need builtin components with `memory` or `sqlite`, the base install is enough.

## Variants

- deterministic docs/test examples: inject fake clients or use builtin demo components
- production-style provider flows: install the matching extras and point adapters at the real service

## Expected result

You should know which extra package to install and whether a real provider is required for the workflow you want to run.

## Troubleshooting

- [Configure generators](configure-generators.md)
- [Choose the right store backend](choose-the-right-store-backend.md)
- [Adapters reference](../reference/adapters.md)

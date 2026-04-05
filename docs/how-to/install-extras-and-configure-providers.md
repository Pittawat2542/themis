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

| Option | Best for | Persistence / runtime behavior | Caveats |
| --- | --- | --- | --- |
| OpenAI | OpenAI or OpenAI-compatible generation | Themis still owns evaluation and storage while the adapter calls the provider | Install `themis-eval[openai]` and either inject a client or provide credentials |
| vLLM | Local or self-hosted OpenAI-compatible generation on Linux | Behaves like another provider endpoint inside Themis runtime controls | Install `themis-eval[vllm]` on Linux, or inject a compatible client |
| LangGraph | Graph-backed generation flows | Themis treats the graph as the generator and still owns the rest of the runtime | Install `themis-eval[langgraph]` and pass a graph with `invoke()` or `ainvoke()` |
| Hugging Face quick-eval | Remote dataset-backed quick evaluations and benchmark materialization | Enables dataset loading rather than provider generation | Install `themis-eval[datasets]` |
| MongoDB store | External persistent storage with MongoDB | Adds a non-local store backend | Install `themis-eval[mongodb]` before using `mongodb` |
| Postgres store | External persistent storage with Postgres | Adds a non-local store backend | Install `themis-eval[postgres]` before using `postgres` |

If you only need builtin components with `memory` or `sqlite`, the base install is enough.

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Deterministic docs or test examples | You want local reproducibility without real providers | Does not validate real provider integration | Builtin demo components, injected fake clients |
| Production-style provider flows | You want real endpoint-backed generation or storage | Requires matching extras, credentials, and service setup | `uv add "themis-eval[openai]"`, `uv add "themis-eval[vllm]"`, `uv add "themis-eval[langgraph]"` |

## Expected result

You should know which extra package to install and whether a real provider is required for the workflow you want to run.

## Troubleshooting

- [Configure generators](configure-generators.md)
- [Choose the right store backend](choose-the-right-store-backend.md)
- [Adapters reference](../reference/adapters.md)

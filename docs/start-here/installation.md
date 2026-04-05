---
title: Installation
diataxis: landing
audience: new Themis users
goal: Explain the default install path, optional extras, and the prerequisites needed for common provider-backed workflows.
---

# Installation

Install the base package when you want the core runtime, builtin components, and deterministic local examples.

```bash
uv add themis-eval
```

## Optional extras

| Option | Best for | Persistence / runtime behavior | Caveats |
| --- | --- | --- | --- |
| `themis-eval[openai]` | OpenAI or OpenAI-compatible provider-backed generation | Keeps Themis in charge of planning, storage, parsing, and scoring while an endpoint handles generation | Requires credentials or an injected client |
| `themis-eval[vllm]` | Local or self-hosted vLLM generation on Linux | Behaves like another provider endpoint inside Themis runtime controls | The packaged extra is Linux-only |
| `themis-eval[langgraph]` | LangGraph-backed generation flows | Themis still owns evaluation and persistence while the graph supplies outputs | Requires a graph with `invoke()` or `ainvoke()` |
| `themis-eval[datasets]` | Quick-eval Hugging Face loading and catalog benchmark materialization | Enables remote dataset-backed loading paths | Install before using shipped benchmark recipes that fetch datasets |
| `themis-eval[mongodb]` or `themis-eval[postgres]` | External persistent stores | Adds non-local storage options for multi-process or environment-driven persistence | Choose the backend that matches your operational environment |
| `themis-eval[docs]` | Local docs builds and docs-site verification | Adds documentation build dependencies only | Not needed for normal runtime usage |

Add an extra with `uv add "themis-eval[extra]"`, for example `uv add "themis-eval[openai]"`.

Use the base install first unless you already know you need a provider integration. The tutorials in this docs set default to deterministic local examples and builtin demo components.

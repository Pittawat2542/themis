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

Optional extras:

- `themis-eval[openai]` for the OpenAI adapter
- `themis-eval[vllm]` for vLLM on Linux
- `themis-eval[langgraph]` for the LangGraph adapter
- `themis-eval[datasets]` for Hugging Face quick-eval loading
- `themis-eval[mongodb]` or `themis-eval[postgres]` for external persistent stores
- `themis-eval[docs]` for local docs builds

Add an extra with `uv add "themis-eval[extra]"`, for example `uv add "themis-eval[openai]"`.

Use the base install first unless you already know you need a provider integration. The tutorials in this docs set default to deterministic local examples and builtin demo components.

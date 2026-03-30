---
title: Installation
diataxis: landing
audience: new Themis users
goal: Explain the default install path, optional extras, and the prerequisites needed for common provider-backed workflows.
---

# Installation

Install the base package when you want the core runtime, builtin components, docs tooling, and local deterministic examples.

```bash
uv pip install -e ".[dev]"
```

Optional extras:

- `.[openai]` for the OpenAI adapter
- `.[vllm]` for vLLM on Linux
- `.[langgraph]` for the LangGraph adapter
- `.[datasets]` for Hugging Face quick-eval loading
- `.[mongodb]` or `.[postgres]` for external persistent stores

Use the base install first unless you already know you need a provider integration. The tutorials in this docs set default to deterministic local examples and builtin demo components.

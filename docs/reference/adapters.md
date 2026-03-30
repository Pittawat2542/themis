---
title: Adapters reference
diataxis: reference
audience: users configuring provider-backed generators
goal: Document the generator adapter entry points and their intended use.
---

# Adapters reference

Available adapters:

- `themis.adapters.openai(...)`
- `themis.adapters.vllm(...)`
- `themis.adapters.langgraph(...)`

Use adapters when Themis should still own planning, reduction, parsing, scoring, storage, and inspection, but a provider or graph runtime should perform generation.

Provider notes:

- OpenAI: inject a client or install the `openai` extra
- vLLM: install the `vllm` extra on Linux or inject a compatible client
- LangGraph: pass a graph with `invoke()` or `ainvoke()`; trace capture is supported when `astream_events()` exists

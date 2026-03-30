---
title: Builtins and adapters
diataxis: reference
audience: users selecting shipped components
goal: Summarize builtin component ids and adapter families with their primary usage.
---

# Builtins and adapters

Builtin component ids:

- `builtin/demo_generator`
- `builtin/majority_vote`
- `builtin/best_of_n`
- `builtin/json_identity`
- `builtin/exact_match`
- `builtin/f1`
- `builtin/bleu`
- `builtin/demo_judge`
- `builtin/llm_rubric`
- `builtin/pairwise_judge`
- `builtin/panel_of_judges`
- `builtin/majority_vote_judge`

Adapter families:

- OpenAI Responses API
- vLLM OpenAI-compatible APIs
- LangGraph graphs

Use builtin ids for deterministic examples, smoke tests, and common scoring patterns. Use adapters when generation should be delegated to an external provider or graph runtime.

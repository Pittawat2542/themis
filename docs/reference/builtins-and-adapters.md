---
title: Builtins and adapters
diataxis: reference
audience: users selecting shipped components
goal: Summarize builtin component ids and adapter families with their primary usage.
---

# Builtins and adapters

The catalog now exposes reusable shipped components directly through
`themis.catalog.load(...)` and `themis.catalog.list_component_ids(...)`.

Builtin component ids:

- `builtin/demo_generator`
- `builtin/majority_vote` (reducer)
- `builtin/best_of_n` (selector)
- `builtin/json_identity`
- `builtin/text`
- `builtin/choice_letter`
- `builtin/math_answer`
- `builtin/code_text`
- `builtin/exact_match`
- `builtin/f1`
- `builtin/bleu`
- `builtin/choice_accuracy`
- `builtin/math_equivalence`
- `builtin/procbench_final_accuracy`
- `builtin/codeforces_pass_rate`
- `builtin/aethercode_pass_rate`
- `builtin/livecodebench_pass_rate`
- `builtin/demo_judge`
- `builtin/llm_rubric`
- `builtin/pairwise_judge`
- `builtin/panel_of_judges`
- `builtin/majority_vote_judge`

Adapter families:

- OpenAI Responses API
- vLLM OpenAI-compatible APIs
- LangGraph graphs

Use builtin ids for deterministic examples, smoke tests, common scoring patterns,
and benchmark-family reuse. Use adapters when generation should be delegated to
an external provider or graph runtime.

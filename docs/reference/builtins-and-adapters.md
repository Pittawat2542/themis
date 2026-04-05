---
title: Builtins and adapters
diataxis: reference
audience: users selecting shipped components
goal: Summarize builtin component ids and adapter families with their primary usage.
---

# Builtins and adapters

The catalog now exposes reusable shipped components directly through
`themis.catalog.load(...)` and `themis.catalog.list_component_ids(...)`.

## Builtin component ids

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `builtin/demo_generator` | Generator | You need deterministic local output for tutorials, smoke tests, or fixture-backed examples | Not a real provider integration |
| `builtin/majority_vote` | Reducer | Multiple samples should collapse to the most common answer | Works best when outputs normalize to the same representation |
| `builtin/best_of_n` | Selector | A judge should choose the strongest candidate before reduction | Requires a judge-backed flow |
| `builtin/json_identity` | Parser | The model already emits structured JSON in the shape you want to score | Minimal normalization |
| `builtin/text` | Parser | You want the reduced output treated as plain text | Useful for rubric-style scoring or simple text metrics |
| `builtin/choice_letter` | Parser | The answer should resolve to a discrete option label | Best for MCQ benchmarks |
| `builtin/math_answer` | Parser | Math answers need normalization before deterministic scoring | Pairs with `builtin/math_equivalence` |
| `builtin/code_text` | Parser | The output is code, including fenced code blocks | Used by code-generation benchmarks and reusable execution metrics |
| `builtin/exact_match` | Metric | Parsed output should match the expected value exactly | Good default for deterministic tasks with stable output format |
| `builtin/f1` | Metric | Token overlap is a better fit than exact string equality | Still deterministic; no judge model required |
| `builtin/bleu` | Metric | You need surface-form overlap for longer text outputs | Better for rough similarity than strict correctness |
| `builtin/choice_accuracy` | Metric | Parsed option labels should score as correct or incorrect | Expects parser output compatible with multiple choice |
| `builtin/math_equivalence` | Metric | Equivalent math expressions or normalized answers should count as correct | Best for math families such as AIME or HMMT |
| `builtin/procbench_final_accuracy` | Metric | You want deterministic final-answer checking for procedure-style tasks | Only use when the recipe is not already judge-backed |
| `builtin/codeforces_pass_rate` | Metric | You need Codeforces-style code execution scoring | Requires an execution backend such as `piston` or `sandbox_fusion` |
| `builtin/aethercode_pass_rate` | Metric | You need AetherCode-specific execution scoring | Requires an execution backend such as `piston` or `sandbox_fusion` |
| `builtin/livecodebench_pass_rate` | Metric | You need LiveCodeBench-style execution scoring | Requires an execution backend such as `piston` or `sandbox_fusion` |
| `builtin/demo_judge` | Judge model | You need a deterministic local judge for examples and tests | Replace with a real judge model for meaningful evaluation |
| `builtin/llm_rubric` | Workflow metric | One judge should score against a rubric | Requires judge models plus optional rubric overrides |
| `builtin/pairwise_judge` | Workflow metric | Two candidates should be compared head-to-head | Useful for selection or pairwise preference evaluation |
| `builtin/panel_of_judges` | Workflow metric | Multiple judges should score the same output and aggregate | Higher cost than a single-judge rubric |
| `builtin/majority_vote_judge` | Workflow metric | Several judge votes should collapse to a majority decision | Useful when categorical consensus matters more than scalar averaging |

## Adapter families

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| OpenAI Responses API | Provider adapter | Themis should own evaluation and storage, while an OpenAI-compatible endpoint handles generation | Install the `openai` extra or inject a compatible client |
| vLLM OpenAI-compatible APIs | Provider adapter | You run a local or self-hosted OpenAI-compatible model endpoint | Install the `vllm` extra on Linux or inject a compatible client |
| LangGraph graphs | Graph adapter | A LangGraph workflow already exists and should act as the generator | Pass a graph with `invoke()` or `ainvoke()`; trace capture improves when `astream_events()` exists |

Use builtin ids for deterministic examples, smoke tests, common scoring patterns,
and benchmark-family reuse. Use adapters when generation should be delegated to
an external provider or graph runtime.

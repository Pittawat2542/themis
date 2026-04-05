# Work At The Extension Boundary

Use this file when the task changes generation, parsing, scoring, adapters, or other runtime extension points.

## Builtin Components

Common shipped component ids:

- generator: `builtin/demo_generator`
- reducer: `builtin/majority_vote`
- selector: `builtin/best_of_n`
- parser: `builtin/json_identity`
- metrics: `builtin/exact_match`, `builtin/f1`, `builtin/bleu`
- workflow-backed judges: `builtin/demo_judge`, `builtin/llm_rubric`, `builtin/pairwise_judge`, `builtin/panel_of_judges`, `builtin/majority_vote_judge`

Use builtin ids for deterministic tests, docs examples, smoke tests, and simple baselines.

## Provider Adapters

Available adapter entry points:

- `themis.adapters.openai(...)`
- `themis.adapters.vllm(...)`
- `themis.adapters.langgraph(...)`

Use adapters when Themis should still own planning, reduction, parsing, scoring, storage, and inspection, but generation should come from an external model or graph runtime.

### Extras

- `.[openai]` for the OpenAI adapter
- `.[vllm]` for vLLM on Linux
- `.[langgraph]` for LangGraph
- `.[datasets]` for Hugging Face dataset loading
- `.[mongodb]` and `.[postgres]` for external stores

Prefer fake or injected clients in tests when the task is about integration shape instead of live provider behavior.

## Prompt And Generation Notes

- `GenerationConfig.prompt_spec` is identity-bearing. Prompt changes can invalidate generation-stage cache reuse.
- `PromptSpec.blocks` is generic structured prompt material, not a special case/example abstraction.
- Prompt specs flow into `GenerateContext`, so custom generators can consume them directly.

## Config Target Syntax

Config files can reference components as:

- builtin ids such as `builtin/exact_match`
- importable module paths such as `package.module:factory` or `package.module:Class`

When a config target resolves to a class, Themis instantiates it without constructor arguments. Do not rely on config loading for stateful constructor injection unless the project already wraps that behavior.

## Protocol Surface

Use the smallest protocol that solves the problem:

- `Generator` for candidate production
- `CandidateReducer` for post-fanout selection or synthesis
- `Parser` for reduced-output normalization
- `PureMetric` for deterministic scoring from parsed outputs
- `LLMMetric`, `SelectionMetric`, or `TraceMetric` for workflow-backed evaluation

Instrumentation contracts:

- `LifecycleSubscriber` for observing stage callbacks and emitted events
- `TracingProvider` for run-level and stage-level tracing spans

Prefer implementing protocols over patching orchestration internals when the change belongs at the extension boundary.

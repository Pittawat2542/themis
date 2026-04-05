# Work At The Extension Boundary

Use this file when the task changes generation, parsing, scoring, adapters, or other runtime extension points.

## Builtin Components

Common shipped component ids:

- generator: `builtin/demo_generator`
- reducer: `builtin/majority_vote`
- selector: `builtin/best_of_n`
- parsers: `builtin/json_identity`, `builtin/text`, `builtin/choice_letter`, `builtin/math_answer`, `builtin/code_text`
- deterministic metrics: `builtin/exact_match`, `builtin/f1`, `builtin/bleu`, `builtin/choice_accuracy`, `builtin/math_equivalence`, `builtin/procbench_final_accuracy`
- code-execution metrics: `builtin/codeforces_pass_rate`, `builtin/aethercode_pass_rate`, `builtin/livecodebench_pass_rate`
- workflow-backed metrics: `builtin/llm_rubric`, `builtin/pairwise_judge`, `builtin/panel_of_judges`, `builtin/majority_vote_judge`
- judge helper: `builtin/demo_judge`

Use builtin ids for deterministic tests, docs examples, smoke tests, and simple baselines.

Use `themis.catalog.list_component_ids(...)` when you need the exact current
shipped ids from code instead of relying on memory.

## Catalog Registry Notes

- `themis.catalog.load(...)` loads builtin shipped components by id.
- `themis.catalog.list_component_ids(kind=...)` lists ids filtered by kind.
- `themis.catalog.builtin_component_refs()` exposes builtin component refs.
- Benchmark recipes live in the catalog too, but they are a separate surface
  from reusable component ids.

## Provider Adapters

Available adapter entry points:

- `themis.adapters.openai(...)`
- `themis.adapters.vllm(...)`
- `themis.adapters.langgraph(...)`

Use adapters when Themis should still own planning, reduction, parsing, scoring, storage, and inspection, but generation should come from an external model or graph runtime.

### Extras

- `themis-eval[openai]` for the OpenAI adapter
- `themis-eval[vllm]` for vLLM on Linux
- `themis-eval[langgraph]` for LangGraph
- `themis-eval[datasets]` for Hugging Face dataset loading and benchmark materialization
- `themis-eval[mongodb]` and `themis-eval[postgres]` for external stores
- `themis-eval[docs]` for local docs builds when working in the Themis repo itself

Prefer fake or injected clients in tests when the task is about integration shape instead of live provider behavior.

## Prompt And Generation Notes

- `GenerationConfig.prompt_spec` is identity-bearing. Prompt changes can invalidate generation-stage cache reuse.
- `PromptSpec.blocks` is generic structured prompt material, not a special case/example abstraction.
- Prompt specs flow into `GenerateContext`, so custom generators can consume them directly.
- Builtin workflow-backed metrics also consume evaluation prompt specs; prompt changes for judge workflows are identity-bearing too.

## Code-Execution Backends

Reusable sandbox-backed metrics default to local HTTP services:

- Piston: `THEMIS_CODE_PISTON_URL`, default `http://localhost:2000`
- Sandbox Fusion: `THEMIS_CODE_SANDBOX_FUSION_URL`, default `http://localhost:8080`

Prefer a fake sandbox executor in tests when you only need to validate metric or
benchmark wiring.

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

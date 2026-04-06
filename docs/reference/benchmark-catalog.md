---
title: Benchmark catalog
diataxis: reference
audience: users running named benchmark entries
goal: Summarize the shipped benchmark names and point readers to adapter-specific constraints.
---

# Benchmark catalog

## Catalog surfaces

| Surface | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| Reusable catalog components | Loadable parsers, metrics, reducers, selectors, generators, and judge workflows | You want a shipped building block such as `builtin/choice_letter` without going through a named benchmark | Load with `themis.catalog.load(...)` or list with `themis.catalog.list_component_ids(...)` |
| Benchmark recipes | Named dataset-backed benchmark definitions | You want a catalog entry such as `mmlu_pro` to materialize a real dataset and wire the right parser and metric stack | Recipes stay cheap to inspect until `materialize_dataset(...)` or `themis.catalog.run(...)` is called |

## Python entry points

| Entry point | Kind | Use when | Notes |
| --- | --- | --- | --- |
| `themis.catalog.list_component_ids(...)` | Discovery helper | You want to see the reusable shipped component ids before deciding what to load | Returns component ids only; benchmark discovery still starts from the benchmark manifest docs |
| `themis.catalog.list_benchmark_ids(...)` | Discovery helper | You want the canonical shipped benchmark ids without inspecting manifests | Returns benchmark ids only |
| `themis.catalog.list_benchmarks(...)` | Metadata listing | You want structured benchmark metadata such as support tier, variants, and version notes | Best source for docs, CLIs, and validation layers |
| `themis.catalog.load(...)` | Resolver | You want to inspect a reusable component or a `BenchmarkDefinition` before running anything | Use `load("builtin/choice_letter")` for a parser or `load("mmlu_pro")` for a benchmark recipe |
| `themis.catalog.run(...)` | Convenience executor | You want the catalog to materialize the dataset and run the benchmark in one call | Best for benchmark execution; for custom slicing, load first and build your own `Dataset` |
| `themis.catalog.validate_benchmark(...)` | Validation helper | You want to confirm a shipped benchmark loads, materializes, and is ready for score smoke checks | Ready code-execution benchmarks run a score smoke check; experimental ones report a skipped score smoke check |

Use `themis.catalog.load("builtin/choice_letter")` when you want a reusable parser
directly. Use `themis.catalog.load("mmlu_pro")` when you want to inspect a
benchmark definition first, including `materialize_dataset(...)`. Use
`themis.catalog.run("mmlu_pro", model=..., store=...)` when you want catalog
convenience without going through the CLI. Use
`themis.catalog.list_benchmark_ids(...)` or `themis.catalog.list_benchmarks(...)`
when you need benchmark discovery or catalog metadata instead of component
discovery.

## Reusable component ids

| Name | Kind | Use when | Key constraints / notes |
| --- | --- | --- | --- |
| `builtin/choice_letter` | Parser | The model should end in an option label such as `A` or `B` | Pair with MCQ benchmarks and metrics such as `builtin/choice_accuracy` |
| `builtin/math_answer` | Parser | You need short-answer math normalization before scoring | Pairs with `builtin/math_equivalence` |
| `builtin/code_text` | Parser | The model emits raw or fenced code that should be scored as source text | Common in code-generation benchmarks |
| `builtin/choice_accuracy` | Metric | You want deterministic correctness for parsed MCQ outputs | Expects parsed option labels rather than long free-form answers |
| `builtin/math_equivalence` | Metric | You want symbolic or normalized math equivalence instead of string equality | Best for AIME-style numeric and short-answer math |
| `builtin/procbench_final_accuracy` | Metric | You want deterministic final-answer checking for procbench-like outputs | Use only when the benchmark recipe is not already using a judge-backed rubric |

## Named benchmark entries

| Benchmark | Shape | Parser / Metric | Variants | Support tier | Notes |
| --- | --- | --- | --- | --- | --- |
| `aime_2025` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Install `themis-eval[datasets]` when materializing from Hugging Face |
| `aime_2026` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Install `themis-eval[datasets]` when materializing from Hugging Face |
| `aethercode` | Code generation | `builtin/code_text` + `builtin/aethercode_pass_rate` | None | ready | Requires `piston` or `sandbox_fusion` plus dataset access |
| `apex_2025` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Install `themis-eval[datasets]` when materializing from Hugging Face |
| `babe` | Multiple choice | `builtin/choice_letter` + `builtin/choice_accuracy` | None | ready | Dataset access only |
| `beyond_aime` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Dataset access only |
| `codeforces` | Code generation | `builtin/code_text` + `builtin/codeforces_pass_rate` | None | ready | Requires `piston` or `sandbox_fusion` plus dataset access |
| `encyclo_k` | Multiple choice | `builtin/choice_letter` + `builtin/choice_accuracy` | None | ready | Dataset access only |
| `frontierscience` | Judge-backed QA | `builtin/json_identity` + `builtin/llm_rubric` | None | ready | Use a real judge model for non-demo scoring |
| `gpqa_diamond` | Multiple choice | `builtin/choice_letter` + `builtin/choice_accuracy` | None | ready | Dataset access only |
| `healthbench` | Judge-backed QA | `builtin/json_identity` + `builtin/llm_rubric` | None | ready | Use a real judge model for non-demo scoring |
| `hle` | Judge-backed expert QA | `builtin/json_identity` + `builtin/panel_of_judges` | Recipe-defined | ready | Check the recipe for supported domain variants before choosing one |
| `hmmt_feb_2025` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Dataset access only |
| `hmmt_nov_2025` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Dataset access only |
| `humaneval` | Code generation | `builtin/code_text` + `builtin/humaneval_pass_rate` | Recipe-defined | ready | Requires `piston` or `sandbox_fusion`; inspect the recipe before choosing a variant |
| `humaneval_plus` | Code generation | `builtin/code_text` + `builtin/humaneval_pass_rate` | Recipe-defined | ready | Requires `piston` or `sandbox_fusion`; inspect the recipe before choosing a variant |
| `imo_answerbench` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Dataset access only |
| `livecodebench` | Code generation | `builtin/code_text` + `builtin/livecodebench_pass_rate` | None | ready | Targets LiveCodeBench release_v6. Requires `piston` or `sandbox_fusion` plus dataset access |
| `lpfqa` | Judge-backed QA | `builtin/json_identity` + `builtin/llm_rubric` | None | ready | Use a real judge model for non-demo scoring |
| `mmlu_pro` | Multiple choice | `builtin/choice_letter` + `builtin/choice_accuracy` | None | ready | Good default catalog benchmark for MCQ smoke checks |
| `mmmlu` | Multiple choice | `builtin/choice_letter` + `builtin/choice_accuracy` | Recipe-defined | ready | Inspect the recipe for supported language or config variants |
| `phybench` | Math short-answer | `builtin/math_answer` + `builtin/math_equivalence` | None | ready | Dataset access only |
| `procbench` | Procedural QA | `builtin/text` + `builtin/llm_rubric` | Recipe-defined | ready | Check task-specific variants before choosing a slice |
| `rolebench` | Role-following judged QA | `builtin/json_identity` + `builtin/llm_rubric` | `instruction_generalization_eng`, `role_generalization_eng` | ready | Use a real judge model for non-demo scoring |
| `simpleqa_verified` | Judge-backed QA | `builtin/json_identity` + `builtin/panel_of_judges` | None | ready | Uses a panel-style judge workflow rather than single-rubric scoring |
| `superchem` | Multiple choice | `builtin/choice_letter` + `builtin/choice_accuracy` | `en`, `zh` | ready | Choose the language variant before materializing the dataset |
| `supergpqa` | Multiple choice | `builtin/choice_letter` + `builtin/choice_accuracy` | None | ready | Dataset access only |

Benchmark recipes now materialize real benchmark datasets instead of a synthetic
placeholder case at run time. Check the benchmark manifest and
[Benchmark adapters](../explanation/benchmark-adapters.md) for adapter-specific
execution requirements such as code execution backends or dataset variants.

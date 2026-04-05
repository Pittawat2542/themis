# Work With The Catalog Surface

Use this file when the task touches shipped benchmarks, reusable builtin
components, or `themis.catalog`.

## Two Catalog Surfaces

The shipped catalog now has two public surfaces:

- reusable catalog components such as parsers, metrics, reducers, selectors,
  generators, and judge workflows
- benchmark recipes that materialize real benchmark datasets and wire those
  components together

The public Python entry points are:

- `themis.catalog.list_component_ids(...)`: list reusable builtin component ids
- `themis.catalog.load(...)`: load either a builtin component or a named
  benchmark recipe
- `themis.catalog.run(...)`: execute a named benchmark through the catalog
  convenience layer
- `themis.catalog.builtin_component_refs()`: inspect builtin component refs

## How Loading Behaves

Use `themis.catalog.load("builtin/choice_letter")` when the task needs a
reusable shipped parser or metric directly.

Use `themis.catalog.load("mmlu_pro")` when the task needs a
`BenchmarkDefinition`. A benchmark definition stays cheap to inspect until you
materialize it. The important benchmark methods are:

- `materialize_dataset(...)`: fetch or construct the real `Dataset`
- `build_experiment(...)`: build an `Experiment` from the benchmark recipe,
  optionally around a pre-materialized dataset

Use `themis.catalog.run("mmlu_pro", model=..., store=...)` when the task wants
catalog convenience without hand-authoring the experiment.

CLI parity for the common benchmark path is:

- `themis quick-eval benchmark --name <benchmark-id>`

## Reusable Builtin Components

Common shipped parsers:

- `builtin/json_identity`
- `builtin/text`
- `builtin/choice_letter`
- `builtin/math_answer`
- `builtin/code_text`

Common shipped deterministic metrics:

- `builtin/exact_match`
- `builtin/f1`
- `builtin/bleu`
- `builtin/choice_accuracy`
- `builtin/math_equivalence`
- `builtin/procbench_final_accuracy`
- `builtin/codeforces_pass_rate`
- `builtin/aethercode_pass_rate`
- `builtin/livecodebench_pass_rate`

Common shipped workflow-backed metrics and judge helpers:

- `builtin/demo_judge`
- `builtin/llm_rubric`
- `builtin/pairwise_judge`
- `builtin/panel_of_judges`
- `builtin/majority_vote_judge`

Use `list_component_ids(kind="parser" | "metric" | "generator" | "reducer" |
"selector" | "judge_model")` when you need the exact available ids from code.

## Benchmark Families

Math short-answer benchmarks:

- `aime_2025`
- `aime_2026`
- `apex_2025`
- `beyond_aime`
- `hmmt_feb_2025`
- `hmmt_nov_2025`
- `imo_answerbench`
- `phybench`

These normally use `builtin/math_answer` plus `builtin/math_equivalence`.

Multiple-choice benchmarks:

- `babe`
- `encyclo_k`
- `gpqa_diamond`
- `mmlu_pro`
- `mmmlu`
- `superchem`
- `supergpqa`

These normally use `builtin/choice_letter` plus `builtin/choice_accuracy`.

Judge-backed QA benchmarks:

- `frontierscience`
- `healthbench`
- `hle`
- `lpfqa`
- `procbench`
- `rolebench`
- `simpleqa_verified`

These typically use `builtin/llm_rubric` or `builtin/panel_of_judges`, often
with `builtin/json_identity` or `builtin/text`.

Code-generation benchmarks:

- `aethercode`
- `codeforces`
- `humaneval`
- `humaneval_plus`
- `livecodebench`

These require code-execution setup and use either reusable code-execution
metrics or benchmark-specific recipe wiring.

## Known Variants And Special Cases

Use these when you need the current shipped benchmark ids without external docs:

- `rolebench` supports `instruction_generalization_eng` and
  `role_generalization_eng`
- `superchem` supports `en` and `zh`
- `hle`, `humaneval`, `humaneval_plus`, `mmmlu`, and `procbench` use
  recipe-validated variant handling; inspect the benchmark definition or the
  source tree before assuming an arbitrary variant string is valid

When the task is to add or fix a benchmark, inspect both:

- `themis/catalog/benchmarks/manifests/benchmarks.toml`
- `themis/catalog/benchmarks/`

Keep static coordinates in the manifest and behavior in Python code.

## Code-Execution Benchmark Setup

Reusable code-execution metrics default to local HTTP sandbox backends:

- Piston: `THEMIS_CODE_PISTON_URL`, default `http://localhost:2000`
- Sandbox Fusion: `THEMIS_CODE_SANDBOX_FUSION_URL`, default
  `http://localhost:8080`

Shipped code-execution metrics currently include:

- `builtin/codeforces_pass_rate`
- `builtin/aethercode_pass_rate`
- `builtin/livecodebench_pass_rate`

If the task is about benchmark wiring rather than sandbox correctness, prefer a
fake executor or fixture-backed metric test over a live sandbox dependency.

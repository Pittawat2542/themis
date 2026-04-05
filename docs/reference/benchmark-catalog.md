---
title: Benchmark catalog
diataxis: reference
audience: users running named benchmark entries
goal: Summarize the shipped benchmark names and point readers to adapter-specific constraints.
---

# Benchmark catalog

The shipped catalog has two public surfaces:

- reusable catalog components such as parsers, metrics, reducers, selectors, and judge workflows
- benchmark recipes that materialize real benchmark datasets and wire those components together

Python catalog entry points:

- `themis.catalog.list_component_ids(...)`: list reusable shipped component ids
- `themis.catalog.load(...)`: resolve a shipped benchmark name to a `BenchmarkDefinition`
- `themis.catalog.run(...)`: build and execute the named benchmark through the catalog runtime

Use `themis.catalog.load("builtin/choice_letter")` when you want a reusable parser
directly. Use `themis.catalog.load("mmlu_pro")` when you want to inspect a
benchmark definition first, including `materialize_dataset(...)`. Use
`themis.catalog.run("mmlu_pro", model=..., store=...)` when you want catalog
convenience without going through the CLI.

Common reusable component ids include:

- `builtin/choice_letter`
- `builtin/math_answer`
- `builtin/code_text`
- `builtin/choice_accuracy`
- `builtin/math_equivalence`
- `builtin/procbench_final_accuracy`

Named benchmark entries currently shipped in the manifest include:

- `aime_2025`
- `aime_2026`
- `aethercode`
- `apex_2025`
- `babe`
- `beyond_aime`
- `encyclo_k`
- `frontierscience`
- `gpqa_diamond`
- `healthbench`
- `hle`
- `hmmt_feb_2025`
- `hmmt_nov_2025`
- `humaneval`
- `humaneval_plus`
- `imo_answerbench`
- `livecodebench`
- `lpfqa`
- `mmlu_pro`
- `mmmlu`
- `codeforces`
- `phybench`
- `procbench`
- `rolebench`
- `simpleqa_verified`
- `superchem`
- `supergpqa`

Benchmark recipes now materialize real benchmark datasets instead of a synthetic
placeholder case at run time. Check the benchmark manifest and
[Benchmark adapters](../explanation/benchmark-adapters.md) for adapter-specific
execution requirements such as code execution backends or dataset variants.

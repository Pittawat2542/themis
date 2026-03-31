---
title: Benchmark catalog
diataxis: reference
audience: users running named benchmark entries
goal: Summarize the shipped benchmark names and point readers to adapter-specific constraints.
---

# Benchmark catalog

Python catalog entry points:

- `themis.catalog.load(...)`: resolve a shipped benchmark name to a `BenchmarkDefinition`
- `themis.catalog.run(...)`: build and execute the named benchmark through the catalog runtime

Use `themis.catalog.load("mmlu_pro")` when you want to inspect the resolved benchmark definition first. Use `themis.catalog.run("mmlu_pro", model=..., store=...)` when you want catalog convenience without going through the CLI.

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

Check the benchmark manifest and [Benchmark adapters](../explanation/benchmark-adapters.md) for adapter-specific execution requirements such as code execution backends or dataset variants.

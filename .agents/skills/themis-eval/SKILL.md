---
name: themis-eval
description: Use when working with the themis-eval Python package to build, run, inspect, compare, hand off, or export benchmark-first LLM evaluations and config reports. This skill is for package users writing workflows with ProjectSpec, BenchmarkSpec, SliceSpec, DatasetProvider, PromptVariantSpec, ParseSpec, ScoreSpec, PluginRegistry, Orchestrator, BenchmarkResult, config report export, custom engines, metrics, extractors, hooks, judge-backed metrics, project files, external handoffs, or the themis-quickcheck CLI. Use it whenever the user asks for benchmark setup, slice design, result aggregation, paired comparisons, config export, or quick SQLite inspection.
---

# themis-eval

Treat Themis as a benchmark-first evaluation framework with one small public
surface:

- `ProjectSpec` defines shared storage and execution policy.
- `BenchmarkSpec` defines the benchmark.
- `SliceSpec` defines one dataset slice.
- `DatasetQuerySpec` defines subset and filter intent.
- `PromptVariantSpec`, `ParseSpec`, and `ScoreSpec` define prompting, parsing, and scoring.
- `PluginRegistry` binds dataset providers, engines, extractors, metrics, judges, and hooks.
- `Orchestrator` plans, runs, resumes, exports, and imports benchmark work.
- `BenchmarkResult` reads projections, aggregates, paired comparisons, artifact bundles, and timelines.

This skill is for consumers of `themis-eval`. Prefer the public package surface
and these references. Do not send the user into retired experiment/task APIs.

## Read The Right Reference

- Read `references/getting-started.md` for installation, the core benchmark flow, and example selection.
- Read `references/plugins-and-specs.md` when defining dataset providers, slices, prompt variants, parse pipelines, engines, metrics, hooks, or judge-backed metrics.
- Read `references/results-and-ops.md` when inspecting aggregates, timelines, artifact bundles, run progress, config reports, or `themis-quickcheck`.
- Read `references/advanced-workflows.md` for project files, external handoffs, benchmark evolution, scaling, and telemetry.

## Working Rules

- Start from the smallest bundled benchmark pattern and adapt it.
- Use `DatasetProvider.scan(slice_spec, query)` for data access.
- Use `BenchmarkSpec` plus `SliceSpec`; do not propose `ExperimentSpec` or `TaskSpec`.
- Treat prompt rendering as orchestration-owned for benchmark runs. Engines receive rendered `trial.prompt.messages` plus preserved prompt metadata such as `trial.prompt.id`, `trial.prompt.family`, and `trial.prompt.variables`.
- Use `ParseSpec` for parsing and keep metrics focused on scoring parsed outputs.
- Use `BenchmarkResult.aggregate(...)` and `paired_compare(...)` before reaching for lower-level report APIs.
- Prefer `from themis import generate_config_report` for one-shot reproducibility exports.
- Use `themis.specs` for supporting public spec imports such as `DatasetSpec`, `GenerationSpec`, and `JudgeInferenceSpec`.
- Use `progress=` plus `themis.progress.ProgressConfig` when the user wants live logs or snapshots.
- Use `themis-quickcheck` when the user wants SQLite inspection without importing benchmark code.
- If the workspace does not contain Themis source or examples, continue using only this skill's references and the installed package surface.

## Default Workflow

1. Install `themis-eval` and needed extras.
2. Implement or adapt a `DatasetProvider` plus the minimum plugin set.
3. Define `ProjectSpec`.
4. Define `BenchmarkSpec`, `SliceSpec`, prompt variants, parses, and scores.
5. Build `Orchestrator` from a project spec or project file.
6. Run with `run_benchmark(...)`, or export/import external work as needed.
7. Inspect the returned `BenchmarkResult`, query progress, or inspect SQLite with `themis-quickcheck`.

## Pattern Map

- Hello world: `references/getting-started.md`
- Project files: `references/advanced-workflows.md`
- Custom parsing and metrics: `references/plugins-and-specs.md`
- Aggregation and artifact bundles: `references/results-and-ops.md`
- Config reports: `references/results-and-ops.md`
- Hooks and judge-backed metrics: `references/plugins-and-specs.md`
- External handoffs and benchmark evolution: `references/advanced-workflows.md`
- Telemetry and observability: `references/advanced-workflows.md`

## Output Expectations

When helping the user, produce runnable code that includes:

- concrete imports
- a dataset provider
- a minimal registry
- `ProjectSpec` and `BenchmarkSpec`
- the right extras to install
- `BenchmarkResult` inspection or export calls after execution
- `generate_config_report(...)` or `themis report` examples when they ask for config export

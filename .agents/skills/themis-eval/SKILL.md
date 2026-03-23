---
name: themis-eval
description: Benchmark-first workflows for `themis-eval`. Use when building, running, inspecting, comparing, or debugging Themis benchmarks; using `quick-eval`, `init`, the built-in benchmark catalog, agent/tool or MCP-enabled evaluations, config reports, external handoffs, or `themis-quickcheck`.
---

# themis-eval

Treat Themis as a benchmark-first evaluation framework with one small public
surface:

- `ProjectSpec` defines shared storage and execution policy.
- `BenchmarkSpec` is the benchmark configuration.
- `SliceSpec` is one dataset slice.
- `DatasetQuerySpec` carries subset and filter intent.
- `PromptVariantSpec`, `ParseSpec`, and `ScoreSpec` cover prompting, parsing, and scoring.
- `BenchmarkDefinition` and `build_benchmark_definition_project(...)` cover reusable benchmark packaging and starter project assembly.
- `ToolSpec` plus `SliceSpec.tool_ids` cover first-class local tool declaration and selection for agent-capable engines.
- `McpServerSpec` plus `SliceSpec.mcp_server_ids` cover provider-hosted remote MCP server selection for MCP-capable engines.
- `PluginRegistry` binds dataset providers, engines, extractors, metrics, judges, and hooks.
- `Orchestrator` plans, runs, resumes, exports, and imports benchmark work.
- `BenchmarkResult` reads projections, aggregates, paired comparisons, artifact bundles, and timelines.

This skill is for consumers of `themis-eval`. Prefer the public package surface
and these references. Do not send the user into retired experiment/task APIs.

## Read The Right Reference

- Read `references/getting-started.md` for installation, quick-eval and init starters, the core benchmark flow, and example selection.
- Consult `references/plugins-and-specs.md` when defining dataset providers, slices, prompt variants, parse pipelines, engines, metrics, hooks, or judge-backed metrics.
- See `references/agent-evals-and-tools.md` when the user needs bootstrap message sequences, follow-up turns, first-class tool passing, or OpenAI-hosted MCP server selection.
- Refer to `references/results-and-ops.md` when inspecting aggregates, timelines, artifact bundles, streamed runs, estimates, run progress, config reports, or `themis-quickcheck`.
- Use `references/advanced-workflows.md` for project files, built-in benchmark catalog projects, external handoffs, benchmark evolution, scaling, and telemetry.
- Use `references/project-structure.md` when the user wants the ideal manual project layout that matches `themis init`.

## Working Rules

- Start from the smallest bundled benchmark pattern and adapt it.
- Prefer `themis quick-eval inline ...` when the user wants a smoke test and `themis init ...` when they want a scaffolded project before writing Python code.
- Prefer built-in catalog benchmarks through `themis quick-eval benchmark ...`, `themis init ... --benchmark <id>`, or `themis.catalog.build_catalog_benchmark_project(...)` when the benchmark already exists in the shipped catalog.
- When the user wants to author a project manually instead of running `themis init`, use `scripts/generate_project_structure.py` or mirror the layouts in `references/project-structure.md`.
- Use `DatasetProvider.scan(slice_spec, query)` for data access.
- Use `BenchmarkSpec` plus `SliceSpec`; do not propose `ExperimentSpec` or `TaskSpec`.
- Use `BenchmarkSpec.simple(...)` and `BenchmarkSpec.preview(...)` for quick authoring and prompt inspection when they fit the task.
- Treat count-based sampling without an explicit seed as deterministic and
  order-based. Add `seed=` only when the user wants a reproducible pseudo-random
  subset instead of the provider's stable prefix.
- Treat prompt rendering as orchestration-owned for benchmark runs. Engines receive rendered `trial.prompt.messages`, rendered `trial.prompt.follow_up_turns`, selected `trial.tools`, selected `trial.mcp_servers`, matching `runtime.tool_handlers`, and preserved prompt metadata such as `trial.prompt.id`, `trial.prompt.family`, and `trial.prompt.variables`.
- Treat MCP as OpenAI-first in this codebase: `McpServerSpec` is for provider-hosted remote tools, not for generic MCP resource browsing.
- Use `ParseSpec` for parsing and keep metrics focused on scoring parsed outputs.
- Use `BenchmarkResult.aggregate(...)` and `paired_compare(...)` before reaching for lower-level report APIs.
- Prefer `from themis import generate_config_report` for one-shot reproducibility exports.
- Use `Orchestrator.run_benchmark_iter(...)` or `estimate(...)` when the user needs streamed progress, trial-matrix inspection, or resume-impact visibility before a full run.
- Use `themis.specs` for supporting public spec imports such as `DatasetSpec`, `GenerationSpec`, and `JudgeInferenceSpec`.
- Use `progress=` plus `themis.progress.ProgressConfig` when the user wants live logs or snapshots.
- Use `themis-quickcheck` when the user wants SQLite inspection without importing benchmark code.
- When reproducibility depends on local runtime helpers, capture `RuntimeContext.tool_handler_versions` so tool implementations are traceable in stored trial records.
- When discussing engine seeding, note that `trial.params.seed` may be wider
  than 32 bits; some providers require truncation such as
  `trial.params.seed & 0xFFFFFFFF`.
- If the workspace does not contain Themis source or examples, continue using only this skill's references and the installed package surface.

## Default Workflow

1. Install `themis-eval` and needed extras.
2. Choose the fastest fitting entry point: `themis quick-eval`, `themis init`, a built-in catalog benchmark, or custom Python authoring.
3. Implement or adapt a `DatasetProvider` plus the minimum plugin set when the built-in catalog is not enough.
4. Define `ProjectSpec`.
5. Define `BenchmarkSpec`, `SliceSpec`, prompt variants, parses, and scores.
6. If the benchmark is agent-style, add bootstrap messages, optional follow-up turns, and explicit local tools plus `tool_ids` or MCP servers plus `mcp_server_ids` as needed.
7. Build `Orchestrator` from a project spec or project file.
8. Run with `run_benchmark(...)` or `run_benchmark_iter(...)`, or export/import external work as needed.
9. Inspect the returned `BenchmarkResult`, estimates, progress, or SQLite state with `themis-quickcheck`.

## Pattern Map

- Hello world: `references/getting-started.md`
- Quick CLI smoke test or scaffold: `references/getting-started.md`
- Built-in benchmark catalog and starter projects: `references/getting-started.md`
- Project files: `references/advanced-workflows.md`
- Custom parsing and metrics: `references/plugins-and-specs.md`
- Agent evaluation and tool passing: `references/agent-evals-and-tools.md`
- OpenAI MCP server selection: `references/agent-evals-and-tools.md`
- Aggregation, streaming runs, estimates, and artifact bundles: `references/results-and-ops.md`
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
- a CLI path or built-in catalog path first when it is the smallest correct answer
- bootstrap messages and follow-up turns when the user is building an agent benchmark
- `ToolSpec` plus slice-level selection when the user needs local runtime tools
- `McpServerSpec` plus slice-level selection and `RuntimeContext.secrets` when the user needs OpenAI-hosted MCP tools
- the right extras to install
- `BenchmarkResult` inspection, `run_benchmark_iter(...)`, or `estimate(...)` calls after execution when they help answer the task
- `generate_config_report(...)` or `themis report` examples when they ask for config export

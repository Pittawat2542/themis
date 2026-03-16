---
name: themis-eval
description: Use when working with the themis-eval Python package to build, run, inspect, resume, compare, or export LLM evaluations and project configs. This skill is for package users writing evaluation code or debugging evaluation workflows with ProjectSpec, ExperimentSpec, PluginRegistry, Orchestrator, ExperimentResult, config report export, custom engines, metrics, extractors, hooks, judge-backed metrics, project files, external handoffs, or the themis-quickcheck CLI. Use it whenever the user asks for a reproducibility snapshot, config export, project/experiment report, or `themis report`, even if they do not name the config-report API directly.
---

# themis-eval

Treat Themis as a code-first evaluation framework with a small public surface:

- `ProjectSpec` defines shared storage and execution policy.
- `ExperimentSpec` defines the experiment matrix.
- `PluginRegistry` binds runtime engines, metrics, extractors, judges, and hooks.
- `Orchestrator` plans, runs, resumes, and imports work.
- `ExperimentResult` reads projections, timelines, reports, and comparisons.

This skill is for a consumer of `themis-eval`. Prefer the public package surface
and this skill's bundled references. Do not assume the user has the Themis repo,
local docs, or local example files available at runtime. Do not send the user
into package internals unless the public package surface requires it.

## Read The Right Reference

- Read `references/getting-started.md` for installation, the core workflow, and
  example selection.
- Read `references/plugins-and-specs.md` when defining dataset loaders, specs,
  engines, metrics, extractors, hooks, or judge-backed metrics.
- Read `references/results-and-ops.md` when inspecting trials, timelines,
  reports, comparisons, run progress, resume behavior, operator-facing logging,
  or `themis-quickcheck`.
- Read `references/advanced-workflows.md` for project files, external handoffs,
  experiment evolution, scaling, telemetry, and observability callbacks.

## Working Rules

- Start from the smallest bundled pattern in these references, then adapt it
  instead of inventing a new pattern.
- Prefer `from themis import generate_config_report` for one-shot export helpers.
  Use `themis report` when the user wants a CLI artifact instead of Python code.
- `themis.contracts.protocols`, `themis.progress`, and
  `themis.specs.foundational` may be imported directly when those modules are
  part of the documented public workflow.
- If the current workspace does not contain Themis source code, examples, or
  docs, continue using only this skill's references and the installed package
  surface.
- Keep `ProjectSpec` stable across reruns. Put storage, retry policy, and
  backend choices there.
- Keep models, prompts, tasks, transforms, evaluations, and parameter sweeps in
  `ExperimentSpec`.
- Register provider names, metric IDs, extractor IDs, and hook IDs explicitly in
  one `PluginRegistry` instance.
- Use `progress=` plus `themis.progress.ProgressConfig` for operator-facing
  status, logging, and callbacks. Use `TelemetryBus` or Langfuse only when the
  user needs event-level observability or external telemetry sinks.
- Use `result.for_transform(...)` or `result.for_evaluation(...)` before
  comparing or exporting when multiple overlays exist.
- Use `orchestrator.get_run_progress(run_id)` for persisted run state after
  `plan()`, `submit()`, `resume()`, or stage-specific execution.
- Use timeline views for one bad example and aggregate helpers for overall
  behavior.

## Default Workflow

1. Install `themis-eval` and any needed extras.
2. Pick the nearest pattern from this skill's references.
3. Implement or adapt a dataset loader plus the minimum plugin set.
4. Define `ProjectSpec`, then `ExperimentSpec`.
5. Build `Orchestrator` from a project spec or project file.
6. Run, submit, resume, or export work, adding `progress=` when the user wants
   live status or logs.
7. Inspect the returned `ExperimentResult`, query
   `orchestrator.get_run_progress(run_id)`,
   compare overlays, or query SQLite summaries with `themis-quickcheck`.

## Pattern Map

- Hello world: `references/getting-started.md`
- Project files: `references/advanced-workflows.md`
- Custom extractors and metrics: `references/plugins-and-specs.md`
- Comparison and reports: `references/results-and-ops.md`
- Config report export: `references/results-and-ops.md`
- Progress logging and run snapshots: `references/results-and-ops.md`
- Resume and reruns: `references/results-and-ops.md`
- Hooks and judge-backed metrics: `references/plugins-and-specs.md`
- External handoffs and experiment evolution: `references/advanced-workflows.md`
- Telemetry and observability callbacks: `references/advanced-workflows.md`

## Output Expectations

When helping the user, produce code that is runnable in their project:

- concrete imports
- a minimal registry and dataset loader
- `ProjectSpec` and `ExperimentSpec`
- the right extras to install
- `generate_config_report(...)` or `themis report` examples when the user asks
  for config export
- `ProgressConfig` or `orchestrator.get_run_progress()` calls when the user asks
  for logging,
  status inspection, or monitoring
- the matching inspection or export calls after execution

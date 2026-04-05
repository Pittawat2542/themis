---
name: themis-eval
description: Guidance for agents working in codebases that depend on the `themis-eval` package or import `themis`. Use whenever a task mentions `evaluate(...)`, `Experiment(...)`, `Experiment.from_config(...)`, `themis` CLI commands, `themis.catalog`, shipped benchmarks, builtin ids such as `builtin/exact_match`, `run_id`, replay/resume, stores, adapters, prompt specs, or custom generators/parsers/reducers/metrics. Also use it when the user asks about benchmark wiring, `quick-eval benchmark`, reusable parser or metric components, or code-execution sandbox backends, even if they do not explicitly ask for "Themis docs."
---

# Themis Eval

Use this skill when a project depends on `themis-eval` but the implementation surface is `themis`.

Treat these as separate but related names:

- distribution/package dependency: `themis-eval`
- Python import namespace: `themis`
- CLI command: `themis`

Assume external docs are unavailable unless the needed information is bundled
inside this skill folder or visible in checked-out source code. Prefer the
bundled reference files below over telling the user to go read docs elsewhere.

## Start With Triage

1. Confirm the project really uses Themis.
2. Identify which authoring surface the project uses most heavily.
3. Load only the reference file that matches the task boundary.

Read [references/triage.md](references/triage.md) first when the repo shape is unclear.

## Pick The Right Surface

Use `evaluate(...)` for the shortest path from dataset plus a few inline arguments to a completed run.

Use `Experiment(...)` when the code needs `compile()`, `run()`, `replay()`, config loading, reusable experiment objects, or persistent execution workflows.

Use `themis.catalog` when the task is about shipped benchmarks, reusable builtin
components, or catalog-backed quick evaluation.

Use config and CLI when the project checks experiment definitions into source control, drives runs from shell automation, or uses worker/batch submission flows.

Use custom extension protocols only when builtin components and adapters are not sufficient.

Read [references/authoring.md](references/authoring.md) when you need to decide between these surfaces or modify experiment/config shape.

## Respect Themis Boundaries

- Preserve the authoring layer already used by the project unless the task explicitly asks for a migration.
- Keep compile-time identity changes separate from runtime-only changes. `run_id` changes come from identity inputs, not from concurrency or retry tuning.
- Remember that config-backed experiments use builtin ids or importable module paths, not live Python objects.
- Prefer persistent stores when the workflow needs resume, reporting, export, comparison, replay, or cross-process cache reuse.
- Prefer builtin components or fake injected clients for deterministic validation when the task is about wiring rather than live provider behavior.
- When the checked-out source includes Themis itself, treat manifests, public
  exports, and tests as higher-signal than prose.

Read [references/catalog.md](references/catalog.md) when the task touches
shipped benchmarks, `themis.catalog.load(...)`, `themis.catalog.run(...)`,
`list_component_ids(...)`, `materialize_dataset(...)`, benchmark variants, or
code-execution benchmark setup.

Read [references/extensions.md](references/extensions.md) when the task touches adapters, provider extras, prompt specs, or custom generators/parsers/reducers/metrics.

Read [references/execution.md](references/execution.md) when the task touches `compile()`, `run()`, `replay()`, `rejudge()`, `run_id`, stores, inspection helpers, or CLI commands.

Read [references/validation.md](references/validation.md) when you need the
smallest meaningful smoke test or need to choose between deterministic, fixture,
catalog, provider, or sandbox-backed validation.

## Validation

- Start with the smallest run that exercises the changed boundary.
- Use `builtin/demo_generator`, `builtin/exact_match`, and `builtin/json_identity` when provider behavior is irrelevant.
- Reuse existing project tests first; otherwise add or run a minimal smoke path through `evaluate(...)`, `Experiment(...)`, or the relevant CLI command.
- For provider-backed code, prefer fake or injected clients before live credentials.

## Reference Files

- [references/triage.md](references/triage.md): detect Themis usage and map repo patterns to the correct internal references and examples.
- [references/authoring.md](references/authoring.md): choose between `evaluate(...)`, `Experiment(...)`, and config/CLI authoring.
- [references/catalog.md](references/catalog.md): shipped catalog components, benchmark recipes, variants, and code-execution benchmark setup.
- [references/extensions.md](references/extensions.md): builtin ids, adapters, extras, config target syntax, and extension protocols.
- [references/execution.md](references/execution.md): compile vs run, identity vs provenance, stores, replay/rejudge, and CLI boundaries.
- [references/validation.md](references/validation.md): minimal smoke-test patterns for deterministic, config, catalog, provider, and sandbox-backed changes.

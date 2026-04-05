---
name: themis-eval
description: Guidance for agents working in codebases that depend on the `themis-eval` package or import `themis`. Use when you need to inspect or modify Themis-based evaluation code, experiment definitions, config files, CLI workflows, stores, adapters, or custom generators/parsers/metrics built around `evaluate(...)`, `Experiment(...)`, or `themis` commands.
---

# Themis Eval

Use this skill when a project depends on `themis-eval` but the implementation surface is `themis`.

Treat these as separate but related names:

- distribution/package dependency: `themis-eval`
- Python import namespace: `themis`
- CLI command: `themis`

## Start With Triage

1. Confirm the project really uses Themis.
2. Identify which authoring surface the project uses most heavily.
3. Load only the reference file that matches the task boundary.

Read [references/triage.md](references/triage.md) first when the repo shape is unclear.

## Pick The Right Surface

Use `evaluate(...)` for the shortest path from dataset plus a few inline arguments to a completed run.

Use `Experiment(...)` when the code needs `compile()`, `run()`, `replay()`, config loading, reusable experiment objects, or persistent execution workflows.

Use config and CLI when the project checks experiment definitions into source control, drives runs from shell automation, or uses worker/batch submission flows.

Use custom extension protocols only when builtin components and adapters are not sufficient.

Read [references/authoring.md](references/authoring.md) when you need to decide between these surfaces or modify experiment/config shape.

## Respect Themis Boundaries

- Preserve the authoring layer already used by the project unless the task explicitly asks for a migration.
- Keep compile-time identity changes separate from runtime-only changes. `run_id` changes come from identity inputs, not from concurrency or retry tuning.
- Remember that config-backed experiments use builtin ids or importable module paths, not live Python objects.
- Prefer persistent stores when the workflow needs resume, reporting, export, comparison, replay, or cross-process cache reuse.
- Prefer builtin components or fake injected clients for deterministic validation when the task is about wiring rather than live provider behavior.

Read [references/extensions.md](references/extensions.md) when the task touches adapters, provider extras, prompt specs, or custom generators/parsers/reducers/metrics.

Read [references/execution.md](references/execution.md) when the task touches `compile()`, `run()`, `replay()`, `rejudge()`, `run_id`, stores, inspection helpers, or CLI commands.

## Validation

- Start with the smallest run that exercises the changed boundary.
- Use `builtin/demo_generator`, `builtin/exact_match`, and `builtin/json_identity` when provider behavior is irrelevant.
- Reuse existing project tests first; otherwise add or run a minimal smoke path through `evaluate(...)`, `Experiment(...)`, or the relevant CLI command.
- For provider-backed code, prefer fake or injected clients before live credentials.

## Reference Files

- [references/triage.md](references/triage.md): detect Themis usage and map repo patterns to the correct docs/examples.
- [references/authoring.md](references/authoring.md): choose between `evaluate(...)`, `Experiment(...)`, and config/CLI authoring.
- [references/extensions.md](references/extensions.md): builtin ids, adapters, extras, config target syntax, and extension protocols.
- [references/execution.md](references/execution.md): compile vs run, identity vs provenance, stores, replay/rejudge, and CLI boundaries.

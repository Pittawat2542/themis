# Triage The Project

Use this file first when you know a repo depends on `themis-eval` but you do not yet know how it is wired.

## Naming Map

- PyPI or dependency name: `themis-eval`
- Python imports: `themis`, `themis.adapters`, `themis.core`
- CLI entry point: `themis`

Do not confuse a missing `themis-eval` string in source code with absence of the package. Most project code only mentions `themis`.

## Fast Search Patterns

Use these searches to determine how the project integrates Themis:

- imports: `rg -n "from themis|import themis|themis\\.adapters|themis\\.core" .`
- config-backed experiments: `rg -n "generation:|evaluation:|storage:|runtime:|builtin/" .`
- CLI usage: `rg -n "\\bthemis\\b|quick-eval|replay|resume|report|inspect" .`
- custom components: `rg -n "Generator|CandidateReducer|Parser|PureMetric|LLMMetric|SelectionMetric|TraceMetric" .`
- persistence and inspection: `rg -n "RunStore|InMemoryRunStore|SqliteRunStore|sqlite_store|get_run_snapshot|get_execution_state|get_evaluation_execution" .`

## Identify The Main Authoring Style

Treat the first matching style as the one to preserve unless the user asks for a migration:

- `evaluate(...)` calls: likely a small Python-first workflow.
- `Experiment(...)` or `Experiment.from_config(...)`: likely the primary authoring surface for serious or repeatable work.
- YAML or TOML files with `generation`, `evaluation`, `storage`, `runtime`, and `datasets`: config-backed workflow.
- `themis` shell commands: CLI-driven workflow.
- implementations of protocol types or module-path component targets: custom extension workflow.

## Choose The Next Reference File

- Need to decide whether code should use `evaluate(...)`, `Experiment(...)`, or config/CLI: read [authoring.md](authoring.md).
- Need to wire a provider, builtin component, or custom runtime component: read [extensions.md](extensions.md).
- Need to reason about `run_id`, persistence, replay, reports, or CLI command behavior: read [execution.md](execution.md).

## Upstream Source Map

These upstream document names are the highest-signal references when they are available locally or linked from the project:

- `Choose your API layer`
- `Run from Python vs config and CLI`
- `Author custom components`
- `Config schema reference`
- `Experiment lifecycle reference`
- `Stores and inspection reference`
- `Install extras and configure providers`

Useful upstream example names to look for:

- `first_evaluate.py`
- `first_experiment.py`
- `external_execution.py`
- `provider_openai.py`
- `custom_generator.py`
- `custom_parser.py`
- `custom_reducer.py`
- `custom_metric.py`
- `rejudge_bundle.py`

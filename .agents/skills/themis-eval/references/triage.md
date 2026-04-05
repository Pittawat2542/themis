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
- catalog and benchmarks: `rg -n "themis\\.catalog|quick-eval benchmark|materialize_dataset|list_component_ids|builtin_component_refs" .`
- config-backed experiments: `rg -n "generation:|evaluation:|storage:|runtime:|builtin/" .`
- CLI usage: `rg -n "\\bthemis\\b|quick-eval|replay|resume|report|inspect" .`
- custom components: `rg -n "Generator|CandidateReducer|Parser|PureMetric|LLMMetric|SelectionMetric|TraceMetric" .`
- persistence and inspection: `rg -n "RunStore|InMemoryRunStore|SqliteRunStore|sqlite_store|get_run_snapshot|get_execution_state|get_evaluation_execution" .`

## Identify The Main Authoring Style

Treat the first matching style as the one to preserve unless the user asks for a migration:

- `evaluate(...)` calls: likely a small Python-first workflow.
- `Experiment(...)` or `Experiment.from_config(...)`: likely the primary authoring surface for serious or repeatable work.
- `themis.catalog.load(...)`, `themis.catalog.run(...)`, or `quick-eval benchmark`: likely catalog-backed benchmark or reusable-component workflow.
- YAML or TOML files with `generation`, `evaluation`, `storage`, `runtime`, and `datasets`: config-backed workflow.
- `themis` shell commands: CLI-driven workflow.
- implementations of protocol types or module-path component targets: custom extension workflow.

## Choose The Next Reference File

- Need to decide whether code should use `evaluate(...)`, `Experiment(...)`, or config/CLI: read [authoring.md](authoring.md).
- Need to reason about shipped benchmarks, builtin ids, or catalog APIs: read [catalog.md](catalog.md).
- Need to wire a provider, builtin component, or custom runtime component: read [extensions.md](extensions.md).
- Need to reason about `run_id`, persistence, replay, reports, or CLI command behavior: read [execution.md](execution.md).
- Need to choose a smoke-test strategy: read [validation.md](validation.md).

## Useful Source Artifacts When Present

If the checked-out source includes Themis itself, these source artifacts are
high-signal:

- `themis/__init__.py`
- `themis/catalog/__init__.py`
- `themis/catalog/manifests/components.toml`
- `themis/catalog/benchmarks/manifests/benchmarks.toml`
- `themis/core/config.py`
- `themis/core/experiment.py`
- `themis/cli/commands/`
- `tests/catalog/`
- `tests/cli/`

Example files are also useful when they exist:

- `first_evaluate.py`
- `first_experiment.py`
- `external_execution.py`
- `provider_openai.py`
- `custom_generator.py`
- `custom_parser.py`
- `custom_reducer.py`
- `custom_metric.py`
- `rejudge_bundle.py`

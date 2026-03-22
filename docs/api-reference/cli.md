# CLI

## `themis`

Primary commands:

- `quick-eval`
- `init`
- `quickcheck`
- `report`

### `themis quick-eval`

Modes:

- `inline`
- `file`
- `huggingface`
- `benchmark`

Shared flags:

- `--model`
- `--provider`
- `--metric`
- `--prompt`
- `--preview`
- `--estimate-only`
- `--format table|json`

Benchmark mode adds:

- `--benchmark`
- `--subset`
- `--revision`
- `--judge-model`
- `--judge-provider`

### `themis init`

Default mode generates a local-data catalog project with:

- `project.toml`
- a runnable package scaffold
- sample local data
- README commands for `quickcheck` and `report`

Use `--benchmark <id>` to scaffold a builtin benchmark-backed starter instead
of the default local-data template.

Builtin benchmark mode generates:

- `project.toml`
- `.env.example` with catalog benchmark defaults
- a runnable package scaffold
- builtin benchmark and dataset wiring through `themis.catalog`
- README commands for `quickcheck` and `report`

### `themis quickcheck`

Primary subcommands:

- `scores`
- `failures`
- `latency`

Benchmark-first filters:

- `--slice`
- `--dimension key=value`

`themis-quickcheck` remains available as a compatibility alias for the same subcommands.

### `themis report`

Supports:

- `--factory`
- `--project-file` with `--run-id`
- `--format json|yaml|markdown|latex`

::: themis.cli.quickcheck

::: themis.cli.quick_eval

::: themis.cli.init

::: themis.cli.report

::: themis.cli.main

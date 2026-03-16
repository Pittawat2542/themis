# CLI

The CLI surface includes the parent `themis` command plus the legacy
`themis-quickcheck` entry point. This page is manual reference first, with the
auto-generated parser helpers left as an appendix instead of being the only
lookup surface.

## `themis`

Top-level command that exposes the current subcommands under one namespace.

```text
usage: themis [-h] {quickcheck,report} ...
```

## `themis report`

Generate a config report from either a Python factory or a persisted run
manifest.

Required input:

- `--factory MODULE:FUNCTION`, or
- `--project-file PATH` plus `--run-id RUN_ID`

Common flags:

- `--format {json,yaml,markdown,latex}` default `markdown`
- `--verbosity {default,full}` default `default`
- `--output PATH` to write the rendered report instead of printing it

Representative success:

```bash
themis report \
  --project-file project.toml \
  --run-id run_123 \
  --format markdown \
  --output config-report.md
```

Representative failures:

```text
--run-id is required with --project-file.
Factory must use the form module:function.
Run manifest 'run_123' was not found.
Project files must use .toml or .json.
```

## `themis quickcheck`

Parent-CLI form of the SQLite summary inspector. It exposes the same
subcommands as the legacy entry point:

- `failures`
- `scores`
- `latency`

Representative usage:

```bash
themis quickcheck scores \
  --db .cache/themis-examples/01-hello-world/themis.sqlite3 \
  --metric exact_match
```

## `themis-quickcheck`

Legacy standalone entry point for the same SQLite summary queries.

```text
usage: themis-quickcheck [-h] {failures,scores,latency} ...
```

Flags by subcommand:

- `failures`: `--db`, optional `--limit`, `--transform-hash`, `--evaluation-hash`
- `scores`: `--db`, optional `--metric`, `--task`, `--evaluation-hash`
- `latency`: `--db`, optional `--transform-hash`, `--evaluation-hash`

Representative failures:

```text
the following arguments are required: --db
invalid choice: 'summary'
```

## Parser API Appendix

::: themis.cli.main
    options:
      show_root_heading: false

::: themis.cli.quickcheck
    options:
      show_root_heading: false

::: themis.cli.report
    options:
      show_root_heading: false

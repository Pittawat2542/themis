# Generate Config Reports

Use this guide when you need a reproducibility-oriented snapshot of the exact
project and experiment configuration used by Themis.

`generate_config_report(...)` walks nested config objects bottom-up, preserving
the hierarchy of `ProjectSpec`, `ExperimentSpec`, and their sub-configs. It
captures field values, types, defaults, source locations, and best-effort
comments/docstrings in one canonical structure that can be rendered in multiple
formats.

It also supports two verbosity levels:

- `default`: compact, paper-facing output that keeps meaningful experimental
  parameters and prunes empty branches
- `full`: exhaustive output for audits and debugging

## Preferred Python API

The preferred root input is a bundle containing both `project` and
`experiment`:

```python
from pathlib import Path

from themis import generate_config_report

bundle = {"project": project, "experiment": experiment}

markdown = generate_config_report(bundle, format="markdown")
json_payload = generate_config_report(bundle, format="json")
paper_view = generate_config_report(bundle, format="markdown", verbosity="default")
full_view = generate_config_report(bundle, format="markdown", verbosity="full")
generate_config_report(bundle, format="latex", output=Path("config-report.tex"))
```

Supported formats:

- `json`
- `yaml`
- `markdown`
- `latex`

If `output` is omitted, the function returns a string. If `output` is set, it
also writes the rendered report to disk. `output` accepts a string path or any
`PathLike` value.

`verbosity` defaults to `default`. Use `verbosity="full"` when you want the
complete collected tree instead of the paper-oriented summary.

## CLI Usage

The parent CLI now exposes config reporting under `themis report`.

### Factory Mode

Use this when your experiment config is defined directly in Python and you want
to import a factory that returns either a config object or a `{project,
experiment}` bundle:

```bash
themis report \
  --factory my_package.evals.paper_run:build_config_bundle \
  --format markdown \
  --verbosity default \
  --output config-report.md
```

### Persisted Run Manifest Mode

Use this when the run already exists and you want the exact persisted
`RunManifest` snapshot:

```bash
themis report \
  --project-file project.json \
  --run-id run_123 \
  --format latex \
  --verbosity full \
  --output config-report.tex
```

This loads the project storage config from the project file, reads the manifest
by `run_id`, and reports on the persisted `project_spec` and `experiment_spec`.

## Worked Example

Assume a small experiment bundle:

```python
bundle = {
    "project": project,
    "experiment": experiment,
}
```

### JSON

```json
{
  "header": {
    "project_name": "report-demo",
    "entrypoint": "my_package.evals.paper_run:build_config_bundle"
  },
  "root": {
    "name": "config",
    "children": [
      {"name": "project"},
      {"name": "experiment"}
    ]
  }
}
```

### YAML

```yaml
header:
  project_name: report-demo
  entrypoint: my_package.evals.paper_run:build_config_bundle
root:
  name: config
  children:
    -
      name: project
    -
      name: experiment
```

### Markdown

```markdown
# Configuration Report

- Project Name: report-demo
- Verbosity: default

<details>
<summary><strong>experiment</strong> <code>$.experiment</code></summary>

| Parameter | Value | Type | Default | Declared In | Source | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| num_samples | 1 | int | 1 | ExperimentSpec | experiment.py:397 | How many samples per trial to draw by default. |
```

### LaTeX

```latex
\section*{Configuration Report}
\item[Verbosity] default
\subsection*{experiment}
\begin{longtable}{...}
```

All four outputs represent the same underlying nested report tree. Only the
renderer changes.

## Default vs Full

For the same config bundle:

- `verbosity="default"` keeps experiment-defining settings such as model IDs,
  prompts, sampling overrides, datasets, metrics, and task structure
- `verbosity="full"` also includes bookkeeping and infrastructure detail such
  as `schema_version`, storage policy, and unchanged default fields

Use `default` for papers and appendices, and `full` when you need a complete
audit trail.

## Report Header and Metadata

Each report includes:

- generation timestamp in UTC
- best-effort git commit hash
- project name when available
- entrypoint used to create the report
- root object type
- selected verbosity level

Each parameter row also carries best-effort source metadata:

- declared config class
- source file and line
- field description
- inline or leading comments when available in source

When a class has no retrievable Python source, the report still renders and
leaves source metadata empty instead of failing.

## Extending Formats

Renderer selection is pluggable through the public registry helpers:

```python
from themis.config_report import register_config_report_renderer


class XmlRenderer:
    def render(self, document) -> str:
        return f"<report root='{document.root.name}' />\n"


register_config_report_renderer("xml", XmlRenderer())
```

Custom config classes should use `config_reportable(...)` or `ConfigReportMixin`
to control default/full visibility. The built-in field-name heuristics are
retained for backwards compatibility with shipped Themis specs.

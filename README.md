# Themis

> Typed, code-first orchestration for reproducible LLM evaluation.

[![CI](https://github.com/Pittawat2542/themis/actions/workflows/ci.yml/badge.svg)](https://github.com/Pittawat2542/themis/actions/workflows/ci.yml)
[![Docs](https://github.com/Pittawat2542/themis/actions/workflows/docs.yml/badge.svg)](https://github.com/Pittawat2542/themis/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/themis-eval.svg)](https://pypi.org/project/themis-eval/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Themis centers on a small public surface:

- `ProjectSpec` for shared storage and execution policy
- `ExperimentSpec` for the experiment matrix
- `PluginRegistry` for engines, extractors, metrics, judges, and hooks
- `Orchestrator` for planning, running, and materializing trials
- `ExperimentResult` for timelines, reports, and paired comparisons
- `generate_config_report(...)` for reproducibility-focused config snapshots
- `themis-quickcheck` for fast SQLite summary inspection

## Why Themis

- **Deterministic planning**: typed specs expand into stable trial hashes.
- **Local-first storage**: append-only events plus projection tables in SQLite.
- **Extensible runtime**: register your own engines, extractors, metrics, judges, and hooks.
- **Inspectable outputs**: read trials, timelines, reports, and paired comparisons from one result object.
- **Predictable resume behavior**: completed trials are skipped when storage, specs, and revision match.
- **Config-as-documentation reporting**: snapshot nested experiment parameters into JSON, YAML, Markdown, or LaTeX.

## Installation

`uv` is required for the documented install and example workflow in this
repository.

```bash
uv add themis-eval

# add extras as needed
uv add "themis-eval[stats,compression]"
```

For the full optional-extra matrix, including `datasets`, provider SDKs,
telemetry, docs tooling, and the contributor toolchain, see
[docs/installation-setup/index.md](docs/installation-setup/index.md).

## Hello World

The runnable quick-start script lives at
[`examples/01_hello_world.py`](examples/01_hello_world.py). It uses local demo
components, so it runs without API keys or provider extras.

Run it with:

```bash
uv run python examples/01_hello_world.py
```

The workflow is:

- create a `PluginRegistry` with one fake engine and one metric
- build `ProjectSpec` and `ExperimentSpec` from top-level `themis` imports
- run `Orchestrator.from_project_spec(...)`
- inspect scores from the returned `ExperimentResult`

For the full script, use the example file directly or the
[Quick Start guide](docs/quick-start/index.md), which embeds that same file.

## Package Namespaces

The root `themis` package stays intentionally small. Additional convenience
namespaces are available when you want lower-level types without long module
paths:

- `themis.records` re-exports persisted record models such as `TrialRecord`
  and `CandidateRecord`
- `themis.types` re-exports shared enums and event/value types used across the
  runtime
- `themis.stats` re-exports paired-comparison tooling and requires the
  `stats` extra

These namespaces are lazy-loaded so the base install keeps a small import
surface and clear optional-dependency boundaries.

## Examples

Runnable examples live in [`examples/`](examples/):

- `01_hello_world.py`
- `02_project_file.py`
- `03_custom_extractor_metric.py`
- `04_compare_models.py`
- `05_resume_run.py`
- `06_hooks_and_timeline.py`
- `07_judge_metric.py`
- `08_external_stage_handoff.py`
- `09_experiment_evolution.py`

## Config Reports

Use `generate_config_report(...)` when you need a human-readable snapshot of the
exact nested config used for an experiment:

```python
from pathlib import Path

from themis import generate_config_report

bundle = {"project": project, "experiment": experiment}
markdown = generate_config_report(bundle, format="markdown")
latex = generate_config_report(bundle, format="latex", output=Path("config-report.tex"))
full_json = generate_config_report(bundle, format="json", verbosity="full")
```

The same collected structure can be rendered as `json`, `yaml`, `markdown`, or
`latex`, with `verbosity="default"` for a paper-facing summary and
`verbosity="full"` for the exhaustive view. Source metadata is complete for
local Python classes with retrievable source, partial for dynamically generated
classes, and empty for third-party or compiled implementations that do not
expose source text. Custom formats can be registered with
`register_config_report_renderer(...)`. For CLI usage and a full worked example, see
[docs/guides/config-reports.md](docs/guides/config-reports.md).

## Documentation

- Docs site: https://pittawat2542.github.io/themis/
- Quick Start: [docs/quick-start/index.md](docs/quick-start/index.md)
- Concepts: [docs/concepts/index.md](docs/concepts/index.md)
- Guides: [docs/guides/index.md](docs/guides/index.md)
- Release Checklist: [docs/guides/releasing.md](docs/guides/releasing.md)
- API Reference: [docs/api-reference/index.md](docs/api-reference/index.md)
- FAQ: [docs/faq/index.md](docs/faq/index.md)

## Development

```bash
# install all dev + feature dependencies
uv sync --all-extras --dev

# test
uv run pytest

# strict docs build
uv run mkdocs build --strict

# baseline lint
uv run ruff check
```

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use Themis in research, cite via [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).

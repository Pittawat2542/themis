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
- `themis-quickcheck` for fast SQLite summary inspection

## Why Themis

- **Deterministic planning**: typed specs expand into stable trial hashes.
- **Local-first storage**: append-only events plus projection tables in SQLite.
- **Extensible runtime**: register your own engines, extractors, metrics, judges, and hooks.
- **Inspectable outputs**: read trials, timelines, reports, and paired comparisons from one result object.
- **Predictable resume behavior**: completed trials are skipped when storage, specs, and revision match.

## Installation

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
[`examples/01_hello_world.py`](examples/01_hello_world.py). It uses only local
fake components, so it works without API keys or provider extras.

The workflow is:

- create a `PluginRegistry` with one fake engine and one metric
- build `ProjectSpec` and `ExperimentSpec` from top-level `themis` imports
- run `Orchestrator.from_project_spec(...)`
- inspect scores from the returned `ExperimentResult`

For the full script, use the example file directly or the
[Quick Start guide](docs/quick-start/index.md), which embeds that same file.

## Examples

Runnable examples live in [`examples/`](examples/):

- `01_hello_world.py`
- `02_project_file.py`
- `03_custom_extractor_metric.py`
- `04_compare_models.py`
- `05_resume_run.py`
- `06_hooks_and_timeline.py`
- `07_judge_metric.py`

## Documentation

- Docs site: https://pittawat2542.github.io/themis/
- Quick Start: [docs/quick-start/index.md](docs/quick-start/index.md)
- Concepts: [docs/concepts/index.md](docs/concepts/index.md)
- Guides: [docs/guides/index.md](docs/guides/index.md)
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

# Repository Guidelines

## Project Structure & Module Organization
- `themis/` – core library (generation, evaluation, experiment orchestration, provider registry, storage).
- `experiments/` – runnable pipelines: `example`, `advanced_example`, `agentic_example` (each with config, CLI, README).
- `tests/` – pytest suites mirroring the module layout plus experiment-level coverage.
- `docs/` – diagrams, component guides, and other references.
- `main.py` / `themis/cli/` – top-level CLI entry points.

## Build, Test, and Development Commands
- `uv run python -m themis.cli demo` – smoke test the core CLI.
- `uv run python -m experiments.example.cli run --dry-run` – preview the simple experiment.
- `uv run pytest` – execute the full test suite (generation/evaluation/experiments).
- `uv run python -m experiments.advanced_example.cli run --storage .cache/foo --run-id bar` – sample cached run.

## Coding Style & Naming Conventions
- Python 3.12+, PEP8 (4-space indent). Favor dataclasses and Pydantic models for configs/entities.
- File names are snake_case; classes in PascalCase; CLI commands dashed (Cyclopts handles parsing).
- Provider names, run IDs, dataset IDs should be lowercase (e.g., `fake-math-llm`, `run-2024Q1`).

## Testing Guidelines
- Framework: pytest (configured via `pyproject.toml`).
- Tests live under `tests/` mirroring module paths (`tests/generation/test_strategies.py`).
- Use descriptive test names (`test_pipeline_returns_metric_aggregates`).
- Aim to cover new abstractions (strategies, builder hooks, storage). Run `uv run pytest` before PRs.

## Commit & Pull Request Guidelines
- Commits: present-tense, imperative summaries (e.g., `Add agentic runner`), group related changes.
- PRs should include: summary, test evidence (`uv run pytest`), mention of new docs/CLIs, and links to issues.
- When touching experiments, describe how to reproduce (e.g., CLI command) and note storage impacts.

## Security & Configuration Tips
- No secrets checked in; configs point to fake/default providers by design.
- Use `--storage` within the repo (e.g., `.cache/...`) to keep cached data versioned or gitignored locally.

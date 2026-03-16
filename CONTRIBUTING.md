# Contributing to Themis

This repository now centers on the typed spec + orchestrator runtime. Use this
guide when you need the current local workflow, quality gates, and release
checks.

## Development Setup

- Python `3.12+`
- [`uv`](https://github.com/astral-sh/uv) for environment and dependency management

Clone the repository and install the current dev environment:

```bash
git clone https://github.com/yourusername/themis.git
cd themis
uv sync --all-extras --dev
```

## Project Shape

- `themis/`: runtime, storage, specs, registry, telemetry, and reporting code
- `tests/`: pytest coverage for contracts, orchestration, records, storage, CLI, and telemetry
- `examples/`: progressive runnable examples for the current implementation
- `docs/`: MkDocs site and API reference
- `scripts/ci/`: release, example, and lint helper scripts used by automation

## Local Checks

Run these before opening a pull request:

```bash
# Full test suite
uv run pytest -q

# Example smoke tests
uv run python scripts/ci/run_examples.py

# Lint and format
uv run ruff format --check .
uv run ruff check .

# Docs must build in strict mode
uv run mkdocs build --strict
```

When you change docs or public API reference surfaces, also run:

```bash
uv run pytest tests/docs/test_docs_consistency.py \
  tests/docs/test_public_docstrings.py \
  tests/docs/test_documented_workflows.py
```

If you only changed a few Python files, matching CI's changed-file checks is
fine too:

```bash
uv run ruff format --check path/to/file.py
uv run ruff check path/to/file.py
```

To enable the repository Git hook for staged Python validation:

```bash
git config core.hooksPath .githooks
```

That hook runs `scripts/ci/check_staged_python.sh`.

## Pull Requests

1. Create a branch from `main`.
2. Make the smallest coherent change you can.
3. Run the relevant local checks.
4. Open a pull request against `main` with a clear summary and any migration notes.

## Docs QA

Before merging docs changes, do one rendered-site pass for:

- Beginner flow: install, run the quick start, and confirm the expected output matches the page.
- Researcher flow: find compare/export/reproduce guidance without reading the whole site.
- Power-user flow: find errors, progress, run-planning, and types reference pages directly from nav/search.

Maintenance rules:

- Any new example requires a docs link, expected output, and API-reference decision before merge.
- Any new public namespace requires a docs link, expected output, and API-reference decision before merge.

## Versioning & Releases

- Release tags follow `vX.Y.Z` or `vX.Y.Z.postN`.
- Keep `pyproject.toml` and `CHANGELOG.md` in sync.
- Validate release metadata locally:

```bash
uv run python scripts/ci/validate_release.py --tag vX.Y.Z
```

- Preview generated release notes:

```bash
uv run python scripts/ci/extract_release_notes.py \
  --tag vX.Y.Z \
  --changelog CHANGELOG.md \
  --output /tmp/release_notes.md
```

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.

# Contributing to Themis

Themis treats code, docs, runnable examples, and release metadata as one product surface. A change is not finished until the user-facing docs, examples, and checks reflect it.

## Setup

Install the dev environment with all optional integrations and docs tooling:

```bash
uv sync --frozen --all-extras --dev
```

The published package name is `themis-eval`. The Python import namespace and CLI command are both `themis`.

## Contributor references

Start with these project docs:

- Product docs overview: [docs/project/index.md](docs/project/index.md)
- Docs architecture: [docs/project/docs-architecture.md](docs/project/docs-architecture.md)
- Writing guide: [docs/project/writing-guide.md](docs/project/writing-guide.md)
- Example authoring: [docs/project/example-authoring.md](docs/project/example-authoring.md)
- Coverage expectations: [docs/project/adding-new-docs-coverage.md](docs/project/adding-new-docs-coverage.md)
- Release/versioning policy: [docs/project/release-and-versioning.md](docs/project/release-and-versioning.md)

## Minimum checks

Run these before opening a PR:

```bash
uv run --extra dev ruff check themis tests examples scripts
uv run --extra dev mypy themis tests
uv run --extra dev pytest -q
uv run --extra dev pytest tests/test_docs_site.py tests/test_docs_examples.py tests/test_docs_inventory.py -q
uv run --extra docs mkdocs build --strict
```

If you changed packaging or release metadata, also run:

```bash
uv build
uv run --extra dev python scripts/ci/check_built_package.py
```

## Docs and examples

- Keep public docs in `docs/` aligned with the exported Python and CLI surface.
- Prefer runnable examples under `examples/docs/` over pseudo-code snippets.
- Keep examples deterministic when provider behavior is not the thing being demonstrated.
- When you add public API surface, update the relevant reference page and any affected tutorials or how-to guides.

## Tests and fixtures

- Reuse existing fake providers and deterministic builtin components whenever possible.
- Add focused tests for new public behavior, then expand integration coverage only where it adds confidence.
- Keep store-backed tests explicit about which backend behavior they are exercising.

## Release expectations

- Public release text should refer to the project as `Themis`, not by an internal version nickname.
- Releases require a matching `pyproject.toml` version, a `CHANGELOG.md` entry, green quality checks, and clean built artifacts.
- Do not ship internal planning files, machine-local paths, or repo-only assets in package artifacts.

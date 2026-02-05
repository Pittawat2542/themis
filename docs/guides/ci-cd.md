# CI/CD and Release Process

This repository uses GitHub Actions with separate pipelines for continuous integration (CI), documentation deployment, and package publishing.

## Pipeline Overview

### CI (`.github/workflows/ci.yml`)

Runs on pull requests and pushes to `main`/`develop`:

- **Quality gates**
  - Ruff format check on changed Python files.
  - Ruff lint on changed Python files.
  - Repository-wide baseline safety lint (`E9`, `F63`, `F7`) for syntax/runtime issues.
- **Tests**
  - Full pytest suite on Python 3.12 and 3.13.
- **Coverage**
  - Coverage report on Python 3.12 for PRs and `main`.
  - Enforced minimum coverage: `60%`.
- **Docs validation**
  - `mkdocs build --strict` to prevent broken docs releases.
- **Package validation**
  - Build wheel/sdist and validate metadata with Twine.
- **Dependency review**
  - PR-only vulnerability and license check for dependency changes.
- **Release platform matrix**
  - On version tags (`v*.*.*`), run tests on Ubuntu, macOS, and Windows.

### Docs Deployment (`.github/workflows/docs.yml`)

- Builds docs in strict mode for pull requests and `main` pushes.
- Deploys to GitHub Pages only on non-PR events.
- Uses the official Pages artifact/deploy actions.

### PyPI Publish (`.github/workflows/pypi.yaml`)

Triggered by version tags (`v*.*.*`) or manual dispatch.

Before publishing, it validates:

- `pyproject.toml` version format (SemVer).
- `CHANGELOG.md` release section contains the same version.
- `docs/CHANGELOG.md` current version matches.
- `CITATION.cff` version matches.
- Tag version matches project version.

Validation entrypoint: `scripts/ci/validate_release.py`

### GitHub Release (`.github/workflows/release.yml`)

- Creates or updates a GitHub Release automatically when a `v*.*.*` tag is pushed.
- Pulls release body directly from the matching section in `CHANGELOG.md`.
- Builds and uploads wheel/sdist artifacts (`dist/*`) to the release.
- Supports manual dispatch with an explicit `tag` input.

## Local Preflight Commands

Run these before opening a pull request:

```bash
# Full test suite
uv run pytest -q

# Coverage (same threshold style used in CI)
uv run pytest -q --cov=themis --cov-report=xml --cov-fail-under=60

# Docs build in strict mode
uv run mkdocs build --strict

# Preview extracted release notes for a tag
uv run python scripts/ci/extract_release_notes.py --tag vX.Y.Z --changelog CHANGELOG.md --output /tmp/release_notes.md

# Baseline syntax/runtime safety lint
uv run ruff check --select E9,F63,F7 themis tests
```

For changed files, CI also enforces:

```bash
uv run ruff format --check <changed_python_files>
uv run ruff check <changed_python_files>
```

## Release Checklist

1. Ensure `pyproject.toml` has the target version.
2. Add a matching release section to root `CHANGELOG.md`.
3. Update `docs/CHANGELOG.md` current release block.
4. Update `CITATION.cff` version/date.
5. Create and push tag: `vX.Y.Z`.
6. Verify `Create GitHub Release` workflow publishes release notes + artifacts.
7. Verify `Publish to PyPI` workflow completes successfully.

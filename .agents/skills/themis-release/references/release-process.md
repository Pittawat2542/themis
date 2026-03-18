# Themis Release Process

## Scope

This reference is for preparing a release in the Themis repository itself. It
mirrors the current repository workflow in `docs/guides/releasing.md` and the
supporting scripts under `scripts/ci/`.

## Release Metadata Files

Update these files together for every release:

- `pyproject.toml`
  - Set `project.version` to the target version.
  - Stable releases must keep
    `Development Status :: 5 - Production/Stable`.
- `CHANGELOG.md`
  - Add a heading in the form `## [X.Y.Z] - YYYY-MM-DD`.
  - Summarize the release from actual commits since the previous tag.
- `CITATION.cff`
  - Set `version` to the same version as `pyproject.toml`.
  - Set `date-released` to the same date as the new changelog heading.
- `uv.lock`
  - Update the editable `[[package]] name = "themis-eval"` entry to the new
    version so local metadata stays consistent.
- `docs/changelog/index.md`
  - Leave it pointing at the canonical root changelog on GitHub.

## Commit Review

Before drafting release notes:

1. Find the last release tag with `git tag --sort=version:refname`.
2. Review the new surface with `git log --oneline <last-tag>..HEAD`.
3. For the relevant commits, inspect stats or full diffs with
   `git show --stat <sha>` or `git show <sha>`.
4. Write the changelog from the real deltas, not from branch or PR titles
   alone.

## Validation Commands

Run the release-specific checks first:

```bash
uv run python scripts/ci/validate_release.py --tag vX.Y.Z
uv run python scripts/ci/extract_release_notes.py --tag vX.Y.Z --changelog CHANGELOG.md --output /tmp/release_notes-X.Y.Z.md
uv run pytest tests/test_release_scripts.py -q
```

Then run the local equivalent of the current CI quality gates:

```bash
changed_files=( $(git diff --name-only HEAD -- '*.py') )
if (( ${#changed_files[@]} > 0 )); then
  uv run ruff format --check "${changed_files[@]}"
fi
uv run ruff check themis tests examples scripts
uv run mypy themis tests
uv run pytest -q
uv run pytest tests/storage/test_postgres_backend.py tests/storage/test_migrate.py -q
uv run pytest tests/docs/test_docs_consistency.py tests/docs/test_public_docstrings.py tests/docs/test_documented_workflows.py tests/docs/test_example_display_paths.py -q
uv run python scripts/ci/run_examples.py
uv run mkdocs build --strict
```

Finish with the package-validation path:

```bash
uv build
uvx --from twine twine check dist/*
uv run python scripts/ci/check_built_package.py
```

## What The Scripts Check

### `scripts/ci/validate_release.py`

Confirms:

- `pyproject.toml` exists and the version is valid semver
- stable releases keep the production classifier
- `CHANGELOG.md` contains the target version heading
- `docs/changelog/index.md` still references the canonical root changelog
- `CITATION.cff` version matches `pyproject.toml`
- `CITATION.cff date-released` matches the changelog date
- the passed tag resolves to the same version as `pyproject.toml`

### `scripts/ci/extract_release_notes.py`

- Accepts `--tag`, `--changelog`, and `--output`
- Extracts the exact `CHANGELOG.md` section for the requested version
- Writes the result to a markdown file suitable for release-note preview

### `scripts/ci/check_built_package.py`

- Reads the version from `pyproject.toml`
- Requires a matching wheel under `dist/`
- Creates a clean temp venv
- Installs the built wheel
- Verifies `import themis`, the benchmark-first curated root exports, and
  `themis-quickcheck --help`

## Tagging And Publish Readiness

If all checks pass, the repository is ready for:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

The documented automation then:

- waits for a successful `CI` workflow on the tagged commit SHA
- validates release metadata again
- extracts the changelog section as release notes
- builds and attaches artifacts to the GitHub Release
- rebuilds and publishes to PyPI

## Practical Notes

- `uv build` and `uvx --from twine twine check dist/*` may need escalation in a
  sandboxed environment because of the shared `uv` cache.
- `uv run ruff format --check ...` may hit the same shared-cache restriction in
  sandboxed environments.
- `dist/` may contain older release artifacts. That is not automatically a
  blocker if the new version artifacts are present and the checks pass.
- Use exact dates and explicit tags in messages to avoid ambiguity.

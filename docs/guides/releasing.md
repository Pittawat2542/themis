# Release Checklist

Use this checklist before creating a release tag. The repository already
contains the automated release path in
[`release.yml`](https://github.com/Pittawat2542/themis/blob/main/.github/workflows/release.yml)
and
[`pypi.yaml`](https://github.com/Pittawat2542/themis/blob/main/.github/workflows/pypi.yaml);
this guide makes the human preflight explicit.

## 1. Confirm release metadata

Before tagging, verify that all versioned metadata agrees on the target release:

- `pyproject.toml`
  - `project.version` matches the tag you will create
  - stable releases use `Development Status :: 5 - Production/Stable`
- `CHANGELOG.md`
  - contains a `## [x.y.z] - YYYY-MM-DD` entry for the release
- `CITATION.cff`
  - `version` matches `pyproject.toml`
  - `date-released` matches the changelog date for that version
- `docs/changelog/index.md`
  - still points readers to the canonical root changelog

Run the built-in validator:

```bash
uv run python scripts/ci/validate_release.py --tag v2.0.0
```

Substitute the target tag as needed.

## 2. Run local verification

Release automation assumes the same checks already pass locally.

```bash
uv run pytest -q
uv run mkdocs build --strict
uv build
uvx --from twine twine check dist/*
uv run python scripts/ci/check_built_package.py
```

If you only need a focused preflight while iterating on release tooling, start
with:

```bash
uv run pytest tests/test_release_scripts.py -q
```

## 3. Create the release tag

Tag the exact commit whose CI results you want to publish. Both release
workflows wait for a successful `CI` run for that commit SHA before continuing.

```bash
git tag v2.0.0
git push origin v2.0.0
```

You can also trigger the workflows manually with the same tag value through
GitHub Actions if you need to republish metadata for an existing tag.

## 4. Watch the automated publication flow

Pushing the tag triggers two workflows:

- `Create GitHub Release`
  - validates release metadata
  - extracts release notes from `CHANGELOG.md`
  - builds artifacts with `uv build`
  - creates or updates the GitHub Release with attached `dist/*` artifacts
- `Publish to PyPI`
  - revalidates metadata
  - rebuilds and verifies package metadata
  - uploads distribution artifacts
  - publishes to PyPI using trusted publishing

If either workflow fails, fix the repository state first and rerun with the
same tag only when the failure cause is understood.

## 5. Post-release checks

After both workflows succeed:

- confirm the GitHub Release body matches the intended changelog section
- confirm the uploaded wheel and sdist are present on the GitHub Release page
- confirm [PyPI](https://pypi.org/project/themis-eval/) shows the new version
- confirm the docs and installation snippets still reference the intended major
  release surface

# Release Procedure

This document outlines the standard operating procedure for releasing a new version of Themis.

## 1. Pre-Release Verification

Before creating a release commit, ensure the codebase is completely stable:
- [ ] Run the full test suite locally: `uv run pytest`
- [ ] Verify documentation builds cleanly: `uv run mkdocs build --strict`
- [ ] Ensure `main` branch is up to date and all PRs for the release are merged.
- [ ] Check GitHub Actions `ci.yml` run for the latest commit on `main` to ensure the `release-compatibility` matrix (Windows, macOS, Ubuntu) has passed.

## 2. Version Bumps (The "Release Commit")

Several files maintain the version string and MUST be updated in sync:
- [ ] `pyproject.toml`: Update `version = "X.Y.Z"` in the `[project]` block.
- [ ] `CITATION.cff`: Update `version: X.Y.Z` and `date-released: YYYY-MM-DD`.
- [ ] `CHANGELOG.md`: 
  - Add a new `## [X.Y.Z] - YYYY-MM-DD` header.
  - Summarize `Added`, `Changed`, `Fixed`, `Removed` sections.

## 3. Documentation Updates

The MkDocs documentation is version-aware and needs updates out-of-band of the core source code:
- [ ] Create `docs/releases/X.Y.Z.md`: Write the detailed release notes.
- [ ] Update `docs/CHANGELOG.md`:
  - Move the previous `Current Release` to `Previous Releases`.
  - Set the new `Current Release` to `X.Y.Z` and link to `releases/X.Y.Z.md`.
- [ ] Update `mkdocs.yml`: Add `- X.Y.Z: releases/X.Y.Z.md` under the `Releases` navigation section.
- [ ] Update `docs/index.md`: Update the `Release Spotlight` success banner to point to the new version's highlights.

## 4. Commit and Tag

Once all files are updated:
- [ ] Commit the changes: `git commit -am "chore: bump version to X.Y.Z"`
- [ ] **WAIT** for the CI to pass on this specific commit before tagging (if pushing directly to main), OR ensure the PR containing these bumps passes CI. The release workflows *will* check for a successful `ci.yml` run for this specific SHA.
- [ ] Create an annotated git tag: `git tag -a vX.Y.Z -m "Version X.Y.Z"`
- [ ] Push commits: `git push origin main`
- [ ] Push tag: `git push origin vX.Y.Z` (or `git push --tags`)

## 5. Automated Pipeline Verification

Pushing the `vX.Y.Z` tag triggers the automated release pipelines:
- [ ] Monitor the `release.yml` workflow: This extracts the changelog, builds the distribution, and publishes the GitHub Release.
- [ ] Monitor the `pypi.yaml` workflow: This builds the distribution, verifies provenance, and publishes to PyPI via Trusted Publishing.
- [ ] Validate both workflows complete successfully. They will wait up to 45 minutes for the upstream `ci.yml` job to finish if it is still running for the tagged commit.

## 6. Post-Release (Optional)

- [ ] Verify PyPI page updates correctly (https://pypi.org/project/themis-eval/).
- [ ] Verify GitHub pages documentation updates correctly.
- [ ] Validate `uv pip install themis-eval==X.Y.Z` installs the new version cleanly in an empty environment.

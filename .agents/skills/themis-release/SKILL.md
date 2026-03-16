---
name: themis-release
description: Use when preparing, validating, or reviewing a Themis project release, including version bumps, changelog drafting from recent commits, release metadata updates, local release preflight, release note extraction, or tag/publish readiness checks. Trigger this skill when the user asks to prep a release, cut a version, verify release artifacts, or follow the Themis release guide.
---

# themis-release

Prepare Themis releases by updating the versioned metadata, summarizing the
new release surface from recent commits, and running the same local preflight
checks that the release guide expects before tagging.

## Read The Right Reference

- Read `references/release-process.md` before editing release metadata or
  running release validation. It captures the repo-specific files, commands,
  and expectations from `docs/guides/releasing.md` plus the supporting CI
  scripts.

## Working Rules

- Start from the latest released tag, then inspect `git log <tag>..HEAD` and
  the underlying commits before drafting changelog text.
- Update release metadata together: `pyproject.toml`, `CHANGELOG.md`,
  `CITATION.cff`, and the editable-package entry in `uv.lock`.
- Keep `docs/changelog/index.md` pointing at the canonical root changelog; do
  not move the release history into the docs tree.
- For stable releases, keep the production classifier in `pyproject.toml`.
- Use the exact release date in both `CHANGELOG.md` and `CITATION.cff`.
- Prefer the built-in scripts over ad hoc checks:
  `scripts/ci/validate_release.py`,
  `scripts/ci/extract_release_notes.py`, and
  `scripts/ci/check_built_package.py`.
- Treat old `dist/` artifacts as noise when validating a new release. Validate
  that the target version artifacts exist and pass checks; do not assume the
  directory is clean unless the user asked for cleanup.
- If a build or `uvx` command fails on sandbox access to the shared `uv` cache,
  rerun with escalation rather than changing the workflow.

## Default Workflow

1. Identify the target version and the latest release tag.
2. Review commits since that tag and extract the user-visible changes.
3. Update `pyproject.toml`, `CHANGELOG.md`, `CITATION.cff`, and `uv.lock`.
4. Run `uv run python scripts/ci/validate_release.py --tag vX.Y.Z`.
5. Preview release notes with
   `uv run python scripts/ci/extract_release_notes.py --tag vX.Y.Z --changelog CHANGELOG.md --output /tmp/release_notes-X.Y.Z.md`.
6. Run the broader preflight from the release guide:
   `uv run pytest -q`,
   `uv run mkdocs build --strict`,
   `uv build`,
   `uvx --from twine twine check dist/*`,
   `uv run python scripts/ci/check_built_package.py`.
7. Report any blockers before tagging. If everything passes, the repo is ready
   for `git tag vX.Y.Z` and `git push origin vX.Y.Z`.

## Output Expectations

When using this skill, produce:

- a concise summary of the release surface based on the commits since the last
  tag
- exact file edits for the version bump and release date
- the validation commands you ran and whether they passed
- any residual release risks, such as stale `dist/` artifacts or skipped checks

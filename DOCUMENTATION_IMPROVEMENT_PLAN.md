# Documentation Improvement Plan

Generated: 2026-02-06

Last Updated: 2026-02-06

## Current Status Snapshot
- Priority 0: Completed
- Priority 1: Completed
- Priority 2: Completed
- Priority 3: Completed

## Goal
Make Themis documentation reliably actionable for researchers by removing runtime blockers, aligning docs with actual behavior, and improving tutorial usability/reproducibility.

## Success Criteria
- A new user can complete a local quickstart (no API key) in under 10 minutes.
- All documented CLI commands in Getting Started and CLI Guide run successfully.
- Tutorial examples use consistent run IDs and pass/fail expectations.
- API/reference docs match implementation signatures and behavior.
- Docs CI catches command/example drift, not just formatting consistency.

## Priority 0 (Critical Blockers)

### 1) Fix `themis compare` CLI crash
- Status: Completed (2026-02-06)
- Severity: Critical
- Affects: onboarding, comparison workflow, countdown tutorial part 4+
- Code target: `/Users/pittawat/projects/themis/themis/cli/main.py`
- Actions:
  - Rename `list` command function to avoid shadowing built-in `list` in type hints.
  - Re-test `uv run python -m themis.cli compare --help` and actual compare execution.
- Acceptance:
  - `compare --help` exits 0.
  - `compare run-a run-b` runs and prints a report when runs exist.

### 2) Fix broken comparison example script
- Status: Completed (2026-02-06)
- Severity: High
- Code target: `/Users/pittawat/projects/themis/examples-simple/04_comparison.py`
- Actions:
  - Ensure compared runs persist evaluation artifacts (avoid `cache=False` or explicitly export/load).
  - Add deterministic run cleanup/setup.
- Acceptance:
  - `uv run python examples-simple/04_comparison.py` exits 0 in a clean repo state.

## Priority 1 (High Impact Doc Accuracy)

### 3) Align `evaluate()` API docs with implementation
- Status: Completed (2026-02-06)
- Severity: High
- Doc target: `/Users/pittawat/projects/themis/docs/api/evaluate.md`
- Actions:
  - Add missing `reference_field` parameter in signature and parameter docs.
  - Correct backend wiring statements to match current behavior.
  - Validate examples with fake-provider/local-safe variants.
- Acceptance:
  - Signature and behavior match `/Users/pittawat/projects/themis/themis/api.py`.

### 4) Fix countdown tutorial run-ID inconsistencies
- Status: Completed (2026-02-06)
- Severity: High
- Doc target: `/Users/pittawat/projects/themis/docs/guides/countdown-tutorial.md`
- Actions:
  - Normalize Part 2 run IDs and all downstream references.
  - Ensure API and CLI snippets reference existing run IDs from earlier steps.
- Acceptance:
  - A reader can follow parts sequentially without manual ID translation.

### 5) Add explicit prerequisites and environment gates per tutorial part
- Status: Completed (2026-02-06)
- Severity: High
- Doc target: `/Users/pittawat/projects/themis/docs/guides/countdown-tutorial.md`
- Actions:
  - For each part, mark requirements:
    - internet required/not required
    - local endpoint required/not required
    - optional extras
    - expected runtime
  - Add a “preflight checks” block (endpoint health, dataset access, installed extras).
- Acceptance:
  - Users can determine readiness before running long workflows.

## Priority 2 (Usability and Onboarding)

### 6) Make quickstart local-first by default
- Status: Completed (2026-02-06)
- Severity: Medium
- Doc targets:
  - `/Users/pittawat/projects/themis/docs/getting-started/quickstart.md`
  - `/Users/pittawat/projects/themis/README.md`
- Actions:
  - Put `demo` + fake model path first.
  - Move hosted model examples to “next step” section.
- Acceptance:
  - First code path succeeds offline (except package install).

### 7) Standardize command invocation guidance
- Status: Completed (2026-02-06)
- Severity: Medium
- Doc targets:
  - `/Users/pittawat/projects/themis/docs/getting-started/installation.md`
  - `/Users/pittawat/projects/themis/docs/guides/cli.md`
  - `/Users/pittawat/projects/themis/docs/reference/api-server.md`
- Actions:
  - Clarify when to use `themis ...` vs `python -m themis.cli ...`.
  - Ensure package naming is consistent (`themis-eval`).
- Acceptance:
  - No conflicting command/package instructions across docs.

### 8) Correct stale API server response examples
- Status: Completed (2026-02-06)
- Severity: Medium
- Doc target: `/Users/pittawat/projects/themis/docs/reference/api-server.md`
- Actions:
  - Update version example and endpoint payloads to current behavior.
- Acceptance:
  - Sample responses match `TestClient` output.

## Priority 3 (Documentation Quality System)

### 9) Strengthen docs CI to catch runnable example drift
- Status: Completed (2026-02-06)
- Severity: High (process quality)
- Targets:
  - `/Users/pittawat/projects/themis/tests/docs/`
  - docs validation scripts
- Actions:
  - Add smoke tests for critical commands:
    - `demo`
    - `list benchmarks`
    - `compare --help`
  - Add script-based example smoke test coverage for `examples-simple`.
  - Add lint checks for run ID consistency in long tutorials.
- Acceptance:
  - CI fails when docs claim workflows that no longer run.

## Proposed Milestones

### Milestone A (1-2 days)
- Completed (2026-02-06)

### Milestone B (2-4 days)
- Completed (2026-02-06)

### Milestone C (2-3 days)
- Completed (2026-02-06)

## Verification Checklist (Post-Update)
- `uv run python -m themis.cli demo --limit 3`
- `uv run python -m themis.cli list benchmarks`
- `uv run python -m themis.cli compare --help`
- `uv run python examples-simple/01_quickstart.py`
- `uv run python examples-simple/04_comparison.py`
- `uv run mkdocs build --strict`
- `uv run pytest tests/docs -q`

## Notes
- Countdown tutorial parts that rely on `http://localhost:1234/api/v1/chat` should explicitly remain optional unless preflight checks pass.
- Reproducibility-gate scripts that intentionally fail under forbidden drift should label failure as expected behavior.

## Validation Evidence (2026-02-06)
- `uv run python -m themis.cli compare --help` (pass)
- `uv run python examples-simple/04_comparison.py` (pass)
- `uv run python -m themis.cli demo --limit 3` (pass)
- `uv run python -m themis.cli list benchmarks` (pass)
- `uv run pytest tests/cli/test_main_compare.py tests/docs -q` (pass)
- `uv run mkdocs build --strict` (pass)

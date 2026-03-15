from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_runs_compatibility_on_push_and_reuses_tag_results() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text()

    assert "on:\n  push:\n  pull_request:" in workflow
    assert "compatibility-precheck:" in workflow
    assert "needs.compatibility-precheck.outputs.skip != 'true'" in workflow
    assert "name: Compatibility Matrix" in workflow
    assert (
        "(github.event_name == 'push' || github.event_name == 'workflow_dispatch')"
        in workflow
    )
    assert (
        "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
        in workflow
    )


def test_docs_workflows_are_limited_to_docs_related_changes() -> None:
    ci_workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text()
    docs_workflow = (PROJECT_ROOT / ".github/workflows/docs.yml").read_text()

    assert "docs-changes:" in ci_workflow
    assert "needs.docs-changes.outputs.changed == 'true'" in ci_workflow
    assert (
        "git ls-files -- docs mkdocs.yml pyproject.toml uv.lock .github/workflows/docs.yml"
        in ci_workflow
    )
    assert "git diff --name-only --diff-filter=ACMRT" in ci_workflow

    assert "paths:" in docs_workflow
    assert "docs/**" in docs_workflow
    assert "mkdocs.yml" in docs_workflow

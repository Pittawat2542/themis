from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tomllib

import themis
from themis.catalog import list_benchmarks


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def test_mkdocs_configuration_exposes_diataxis_navigation() -> None:
    config_path = REPO_ROOT / "mkdocs.yml"

    assert config_path.is_file()
    contents = config_path.read_text(encoding="utf-8")

    for required_entry in (
        "site_name:",
        "nav:",
        "- Start Here:",
        "- Tutorials:",
        "- How-To:",
        "- Reference:",
        "- Explanation:",
        "- Glossary:",
        "- FAQ:",
        "- Project:",
    ):
        assert required_entry in contents

    assert "exclude_docs:" in contents
    assert "_snippets/" in contents


def test_mkdocs_configuration_supports_dark_mode_and_mermaid_diagrams() -> None:
    config_path = REPO_ROOT / "mkdocs.yml"

    contents = config_path.read_text(encoding="utf-8")

    for required_entry in (
        "palette:",
        "scheme: default",
        "scheme: slate",
        "toggle:",
        "icon: material/brightness-7",
        "icon: material/brightness-4",
        "custom_fences:",
        "name: mermaid",
        "class: mermaid",
        "extra_javascript:",
        "https://unpkg.com/mermaid@11/dist/mermaid.min.js",
        "javascripts/mermaid.mjs",
    ):
        assert required_entry in contents

    assert (DOCS_ROOT / "javascripts" / "mermaid.mjs").is_file()


def test_docs_tree_contains_required_diataxis_entrypoints() -> None:
    required_paths = (
        DOCS_ROOT / "index.md",
        DOCS_ROOT / "start-here" / "index.md",
        DOCS_ROOT / "start-here" / "installation.md",
        DOCS_ROOT / "start-here" / "choose-your-api-layer.md",
        DOCS_ROOT / "start-here" / "choose-your-storage-backend.md",
        DOCS_ROOT / "start-here" / "route-by-goal.md",
        DOCS_ROOT / "start-here" / "troubleshooting.md",
        DOCS_ROOT / "glossary.md",
        DOCS_ROOT / "faq.md",
        DOCS_ROOT / "tutorials" / "first-evaluate.md",
        DOCS_ROOT / "tutorials" / "first-experiment.md",
        DOCS_ROOT / "tutorials" / "first-persisted-run.md",
        DOCS_ROOT / "tutorials" / "first-llm-judged-evaluation.md",
        DOCS_ROOT / "tutorials" / "first-advanced-run.md",
        DOCS_ROOT / "tutorials" / "first-custom-component.md",
        DOCS_ROOT / "tutorials" / "first-external-execution.md",
        DOCS_ROOT / "how-to" / "choose-the-right-api-layer.md",
        DOCS_ROOT / "how-to" / "choose-the-right-store-backend.md",
        DOCS_ROOT / "how-to" / "install-extras-and-configure-providers.md",
        DOCS_ROOT / "how-to" / "run-from-python-vs-config-and-cli.md",
        DOCS_ROOT / "how-to" / "configure-generators.md",
        DOCS_ROOT / "how-to" / "author-custom-components.md",
        DOCS_ROOT / "how-to" / "observe-runs-and-instrumentation.md",
        DOCS_ROOT / "how-to" / "capture-traces-and-conversations.md",
        DOCS_ROOT / "how-to" / "use-reduction-strategies.md",
        DOCS_ROOT / "how-to" / "use-pure-metrics.md",
        DOCS_ROOT / "how-to" / "use-workflow-backed-metrics.md",
        DOCS_ROOT / "how-to" / "resume-and-inspect-runs.md",
        DOCS_ROOT / "how-to" / "reproduce-and-rejudge-runs.md",
        DOCS_ROOT / "how-to" / "compare-export-and-report.md",
        DOCS_ROOT / "how-to" / "tune-runtime-controls.md",
        DOCS_ROOT / "how-to" / "use-submit-worker-and-batch.md",
        DOCS_ROOT / "how-to" / "run-benchmarks.md",
        DOCS_ROOT / "reference" / "index.md",
        DOCS_ROOT / "reference" / "python-api.md",
        DOCS_ROOT / "reference" / "root-package-api.md",
        DOCS_ROOT / "reference" / "experiment-lifecycle.md",
        DOCS_ROOT / "reference" / "stores-and-inspection.md",
        DOCS_ROOT / "reference" / "adapters.md",
        DOCS_ROOT / "reference" / "cli.md",
        DOCS_ROOT / "reference" / "config-schema.md",
        DOCS_ROOT / "reference" / "protocols.md",
        DOCS_ROOT / "reference" / "data-models.md",
        DOCS_ROOT / "reference" / "builtins-and-adapters.md",
        DOCS_ROOT / "reference" / "benchmark-catalog.md",
        DOCS_ROOT / "explanation" / "index.md",
        DOCS_ROOT / "explanation" / "case-lifecycle.md",
        DOCS_ROOT / "explanation" / "run-snapshot.md",
        DOCS_ROOT / "explanation" / "identity-vs-provenance.md",
        DOCS_ROOT / "explanation" / "compile-vs-run.md",
        DOCS_ROOT / "explanation" / "generation-vs-evaluation.md",
        DOCS_ROOT / "explanation" / "fanout-reduction-parsing-scoring.md",
        DOCS_ROOT / "explanation" / "reducer-parser-metric-boundaries.md",
        DOCS_ROOT / "explanation" / "events-stores-and-read-models.md",
        DOCS_ROOT / "explanation" / "artifacts-and-inspection.md",
        DOCS_ROOT / "explanation" / "reproducibility-and-rejudge.md",
        DOCS_ROOT / "explanation" / "failure-retry-and-resume.md",
        DOCS_ROOT / "explanation" / "extension-boundaries.md",
        DOCS_ROOT / "explanation" / "api-layer-model.md",
        DOCS_ROOT / "explanation" / "metric-families-and-subjects.md",
        DOCS_ROOT / "explanation" / "store-backend-model.md",
        DOCS_ROOT / "explanation" / "benchmark-adapters.md",
        DOCS_ROOT / "explanation" / "reporting-and-read-models.md",
        DOCS_ROOT / "explanation" / "determinism-and-reproducibility.md",
        DOCS_ROOT / "project" / "index.md",
        DOCS_ROOT / "project" / "docs-architecture.md",
        DOCS_ROOT / "project" / "writing-guide.md",
        DOCS_ROOT / "project" / "example-authoring.md",
        DOCS_ROOT / "project" / "adding-new-docs-coverage.md",
        DOCS_ROOT / "project" / "release-and-versioning.md",
        REPO_ROOT / "CONTRIBUTING.md",
    )

    for required_path in required_paths:
        assert required_path.is_file(), (
            f"missing docs page: {required_path.relative_to(REPO_ROOT)}"
        )


def test_docs_examples_are_file_backed_and_embedded() -> None:
    examples_root = REPO_ROOT / "examples" / "docs"
    expected_examples = (
        examples_root / "first_evaluate.py",
        examples_root / "first_experiment.py",
        examples_root / "persisted_run.py",
        examples_root / "llm_judged_evaluation.py",
        examples_root / "advanced_run.py",
        examples_root / "custom_generator.py",
        examples_root / "custom_metric.py",
        examples_root / "custom_parser.py",
        examples_root / "custom_reducer.py",
        examples_root / "provider_openai.py",
        examples_root / "provider_vllm.py",
        examples_root / "provider_langgraph.py",
        examples_root / "observability.py",
        examples_root / "pure_metrics.py",
        examples_root / "workflow_metrics.py",
        examples_root / "trace_capture.py",
        examples_root / "rejudge_bundle.py",
        examples_root / "external_execution.py",
    )

    for example_path in expected_examples:
        assert example_path.is_file(), (
            f"missing example file: {example_path.relative_to(REPO_ROOT)}"
        )

    snippets_root = DOCS_ROOT / "_snippets"
    embedded_docs = (
        DOCS_ROOT / "tutorials" / "first-evaluate.md",
        DOCS_ROOT / "tutorials" / "first-experiment.md",
        DOCS_ROOT / "tutorials" / "first-persisted-run.md",
        DOCS_ROOT / "tutorials" / "first-llm-judged-evaluation.md",
        DOCS_ROOT / "tutorials" / "first-advanced-run.md",
        DOCS_ROOT / "tutorials" / "first-custom-component.md",
        DOCS_ROOT / "tutorials" / "first-external-execution.md",
        DOCS_ROOT / "how-to" / "run-from-python-vs-config-and-cli.md",
        DOCS_ROOT / "how-to" / "configure-generators.md",
        DOCS_ROOT / "how-to" / "author-custom-components.md",
        DOCS_ROOT / "how-to" / "observe-runs-and-instrumentation.md",
        DOCS_ROOT / "how-to" / "capture-traces-and-conversations.md",
        DOCS_ROOT / "how-to" / "reproduce-and-rejudge-runs.md",
        DOCS_ROOT / "how-to" / "use-reduction-strategies.md",
        DOCS_ROOT / "how-to" / "use-pure-metrics.md",
        DOCS_ROOT / "how-to" / "use-workflow-backed-metrics.md",
    )

    for doc_path in embedded_docs:
        contents = doc_path.read_text(encoding="utf-8")
        assert "--8<--" in contents
        assert (
            str(snippets_root.relative_to(DOCS_ROOT)) in contents
            or "examples/docs/" in contents
        )


def test_reference_docs_cover_cli_public_api_and_catalogs() -> None:
    cli_reference = (DOCS_ROOT / "reference" / "cli.md").read_text(encoding="utf-8")
    for command in (
        "quick-eval",
        "run",
        "submit",
        "resume",
        "estimate",
        "report",
        "quickcheck",
        "compare",
        "export",
        "init",
        "worker",
        "batch",
    ):
        assert f"`{command}`" in cli_reference

    python_reference = (DOCS_ROOT / "reference" / "python-api.md").read_text(
        encoding="utf-8"
    )
    assert "::: themis" in python_reference
    for symbol in themis.__all__:
        assert symbol in python_reference, (
            f"missing public symbol in reference docs: {symbol}"
        )

    builtins_reference = (
        DOCS_ROOT / "reference" / "builtins-and-adapters.md"
    ).read_text(encoding="utf-8")
    components_manifest = tomllib.loads(
        (REPO_ROOT / "themis" / "catalog" / "manifests" / "components.toml").read_text(
            encoding="utf-8"
        )
    )
    for component_id in components_manifest["components"]:
        assert f"`{component_id}`" in builtins_reference

    benchmark_reference = (DOCS_ROOT / "reference" / "benchmark-catalog.md").read_text(
        encoding="utf-8"
    )
    benchmark_manifest = tomllib.loads(
        (
            REPO_ROOT
            / "themis"
            / "catalog"
            / "benchmarks"
            / "manifests"
            / "benchmarks.toml"
        ).read_text(encoding="utf-8")
    )
    for benchmark_name in benchmark_manifest["benchmarks"]:
        assert f"`{benchmark_name}`" in benchmark_reference

    assert "themis.catalog.load(...)" in benchmark_reference
    assert "themis.catalog.run(...)" in benchmark_reference


def test_benchmark_reference_tracks_catalog_metadata() -> None:
    benchmark_reference = (DOCS_ROOT / "reference" / "benchmark-catalog.md").read_text(
        encoding="utf-8"
    )

    assert "themis.catalog.list_benchmark_ids(...)" in benchmark_reference
    assert "themis.catalog.list_benchmarks(...)" in benchmark_reference
    assert "Support tier" in benchmark_reference

    for entry in list_benchmarks():
        assert f"`{entry.benchmark_id}`" in benchmark_reference
        assert f"| `{entry.benchmark_id}` |" in benchmark_reference
        assert entry.support_tier in benchmark_reference
        if entry.declared_variants:
            for variant in entry.declared_variants:
                assert variant in benchmark_reference
        if entry.version_notes is not None:
            assert entry.version_notes in benchmark_reference


def test_docs_cover_required_topics_and_optional_extras() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/docs/build_inventory.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    inventory = json.loads(result.stdout)
    docs_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if "_snippets" not in path.parts
    )

    for markers in inventory["required_topics"].values():
        for marker in markers:
            assert marker in docs_text

    install_guide = (
        DOCS_ROOT / "how-to" / "install-extras-and-configure-providers.md"
    ).read_text(encoding="utf-8")
    for extra in (
        "themis-eval[openai]",
        "themis-eval[vllm]",
        "themis-eval[langgraph]",
        "themis-eval[datasets]",
        "themis-eval[mongodb]",
        "themis-eval[postgres]",
    ):
        assert extra in install_guide


def test_substantive_docs_pages_have_metadata_contract() -> None:
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        if "_snippets" in path.parts:
            continue
        contents = path.read_text(encoding="utf-8")
        assert contents.startswith("---\n"), (
            f"missing front matter: {path.relative_to(REPO_ROOT)}"
        )
        assert "\ndiataxis:" in contents, (
            f"missing diataxis metadata: {path.relative_to(REPO_ROOT)}"
        )
        assert "\naudience:" in contents, (
            f"missing audience metadata: {path.relative_to(REPO_ROOT)}"
        )
        assert "\ngoal:" in contents, (
            f"missing goal metadata: {path.relative_to(REPO_ROOT)}"
        )


def test_tutorials_and_guides_follow_expected_structure() -> None:
    tutorial_required_sections = (
        "## What you will build",
        "## Prerequisites",
        "## Steps",
        "## Expected results",
        "## Common failure points",
        "## Next steps",
    )
    for path in sorted((DOCS_ROOT / "tutorials").glob("*.md")):
        contents = path.read_text(encoding="utf-8")
        for heading in tutorial_required_sections:
            assert heading in contents, (
                f"missing tutorial section {heading}: {path.relative_to(REPO_ROOT)}"
            )

    guide_required_sections = (
        "Goal:",
        "When to use this",
        "## Procedure",
        "## Variants",
        "## Expected result",
        "## Troubleshooting",
    )
    for path in sorted((DOCS_ROOT / "how-to").glob("*.md")):
        contents = path.read_text(encoding="utf-8")
        for heading in guide_required_sections:
            assert heading in contents, (
                f"missing guide section {heading}: {path.relative_to(REPO_ROOT)}"
            )


def test_readme_is_docs_landing_page_not_full_manual() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Documentation" in readme
    assert "## Quick Start" in readme
    assert "## Examples" not in readme
    assert "## Runtime controls" not in readme
    assert "## Resume and bundle behavior" not in readme


def test_contributing_guide_uses_repository_relative_links() -> None:
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "/Users/" not in contributing
    assert "(docs/project/index.md)" in contributing

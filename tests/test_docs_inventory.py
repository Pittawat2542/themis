from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

TABLE_REQUIRED_SECTIONS = (
    (
        "reference/benchmark-catalog.md",
        "Catalog surfaces",
        ("Surface", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/benchmark-catalog.md",
        "Python entry points",
        ("Entry point", "Kind", "Use when", "Notes"),
    ),
    (
        "reference/benchmark-catalog.md",
        "Reusable component ids",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/benchmark-catalog.md",
        "Named benchmark entries",
        ("Benchmark", "Shape", "Parser / Metric", "Variants", "Extra setup"),
    ),
    (
        "reference/builtins-and-adapters.md",
        "Builtin component ids",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/builtins-and-adapters.md",
        "Adapter families",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/cli.md",
        "Command groups",
        ("Command", "What it does", "When to use it", "Key inputs / constraints"),
    ),
    (
        "reference/cli.md",
        "Command behavior",
        ("Command", "What it does", "When to use it", "Key inputs / constraints"),
    ),
    (
        "reference/cli.md",
        "Subcommands",
        ("Command", "What it does", "When to use it", "Key inputs / constraints"),
    ),
    (
        "reference/cli.md",
        "Current CLI boundary",
        ("Surface", "Current behavior", "Use instead when", "Notes"),
    ),
    (
        "reference/config-schema.md",
        "Config file support",
        ("Field", "Required", "Purpose", "Affects run_id", "Notes"),
    ),
    (
        "reference/config-schema.md",
        "Component target syntax",
        ("Field", "Required", "Purpose", "Affects run_id", "Notes"),
    ),
    (
        "reference/config-schema.md",
        "`GenerationConfig`",
        ("Field", "Required", "Purpose", "Affects run_id", "Notes"),
    ),
    (
        "reference/config-schema.md",
        "`EvaluationConfig`",
        ("Field", "Required", "Purpose", "Affects run_id", "Notes"),
    ),
    (
        "reference/config-schema.md",
        "`StorageConfig`",
        ("Field", "Required", "Purpose", "Affects run_id", "Notes"),
    ),
    (
        "reference/config-schema.md",
        "`RuntimeConfig`",
        ("Field", "Required", "Purpose", "Affects run_id", "Notes"),
    ),
    (
        "reference/protocols.md",
        "Important runtime instrumentation contracts",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/protocols.md",
        "Important config/runtime contracts",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/adapters.md",
        "Available adapters",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/adapters.md",
        "Provider notes",
        ("Option", "Best for", "Persistence / runtime behavior", "Caveats"),
    ),
    (
        "reference/experiment-lifecycle.md",
        "Primary entry points",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/experiment-lifecycle.md",
        "Lookup notes",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/stores-and-inspection.md",
        "Store-related symbols",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/stores-and-inspection.md",
        "Inspection and export helpers",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/stores-and-inspection.md",
        "Persistence boundaries",
        ("Option", "Best for", "Persistence / runtime behavior", "Caveats"),
    ),
    (
        "reference/python-api.md",
        "Root exports",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "reference/data-models.md",
        "Important payloads to inspect directly",
        ("Name", "Kind", "Use when", "Key constraints / notes"),
    ),
    (
        "start-here/installation.md",
        "Optional extras",
        ("Option", "Best for", "Persistence / runtime behavior", "Caveats"),
    ),
)

VARIANT_SECTIONS = (
    "how-to/author-custom-components.md",
    "how-to/capture-traces-and-conversations.md",
    "how-to/choose-the-right-api-layer.md",
    "how-to/choose-the-right-store-backend.md",
    "how-to/compare-export-and-report.md",
    "how-to/configure-generators.md",
    "how-to/install-extras-and-configure-providers.md",
    "how-to/observe-runs-and-instrumentation.md",
    "how-to/reproduce-and-rejudge-runs.md",
    "how-to/resume-and-inspect-runs.md",
    "how-to/run-benchmarks.md",
    "how-to/run-from-python-vs-config-and-cli.md",
    "how-to/tune-runtime-controls.md",
    "how-to/use-pure-metrics.md",
    "how-to/use-reduction-strategies.md",
    "how-to/use-submit-worker-and-batch.md",
    "how-to/use-workflow-backed-metrics.md",
)


def test_docs_inventory_script_reports_public_surface() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/docs/build_inventory.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert sorted(payload) == [
        "benchmarks",
        "builtin_components",
        "catalog_exports",
        "cli_commands",
        "docs_destinations",
        "public_exports",
        "required_topics",
    ]
    assert "Experiment" in payload["public_exports"]
    assert "load" in payload["catalog_exports"]
    assert "quick-eval benchmark" in payload["cli_commands"]
    assert "builtin/exact_match" in payload["builtin_components"]
    assert "mmlu_pro" in payload["benchmarks"]
    assert payload["docs_destinations"]["glossary"] == "docs/glossary.md"
    assert "LifecycleSubscriber" in payload["required_topics"]["observability"]


def _section_text(doc_path: Path, heading: str) -> str:
    heading_line = f"## {heading}"
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == heading_line:
            start_index = index + 1
            break
    assert start_index is not None, f"missing section {heading!r} in {doc_path}"

    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if lines[index].startswith("## "):
            end_index = index
            break
    return "\n".join(lines[start_index:end_index]).strip()


def _assert_table_section(
    doc_path: Path, heading: str, required_columns: tuple[str, ...]
) -> None:
    section = _section_text(doc_path, heading)
    table_lines = [line for line in section.splitlines() if line.startswith("|")]
    assert table_lines, f"expected markdown table in {doc_path} section {heading!r}"

    header_line = table_lines[0]
    for column in required_columns:
        assert column in header_line, (
            f"missing column {column!r} in {doc_path} section {heading!r}"
        )

    assert not any(line.startswith("- ") for line in section.splitlines()), (
        f"inventory section {heading!r} in {doc_path} should use a table, not bullets"
    )


def test_docs_inventory_sections_use_decision_tables() -> None:
    for relative_path, heading, required_columns in TABLE_REQUIRED_SECTIONS:
        _assert_table_section(DOCS_ROOT / relative_path, heading, required_columns)

    variant_columns = (
        "Variant",
        "Best when",
        "Tradeoff",
        "Related APIs / commands",
    )
    for relative_path in VARIANT_SECTIONS:
        _assert_table_section(DOCS_ROOT / relative_path, "Variants", variant_columns)

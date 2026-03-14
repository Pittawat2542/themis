from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_quick_start_embeds_the_runnable_hello_world_example() -> None:
    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()

    assert '--8<-- "examples/01_hello_world.py"' in quick_start


def test_readme_and_quick_start_use_curated_top_level_imports() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()

    assert "examples/01_hello_world.py" in readme
    assert "from themis.specs import DatasetSpec" not in readme
    assert "from themis.specs import DatasetSpec" not in quick_start
    assert "from themis.registry.plugin_registry import PluginRegistry" not in readme
    assert (
        "from themis.registry.plugin_registry import PluginRegistry" not in quick_start
    )


def test_hello_world_example_uses_documented_root_entry_points() -> None:
    hello_world = (PROJECT_ROOT / "examples/01_hello_world.py").read_text()

    assert "from themis import (" in hello_world
    assert "    Orchestrator," in hello_world
    assert "    PluginRegistry," in hello_world
    assert (
        "from themis.orchestration.orchestrator import Orchestrator" not in hello_world
    )
    assert (
        "from themis.registry.plugin_registry import PluginRegistry" not in hello_world
    )


def test_readme_and_architecture_docs_explain_lazy_public_namespaces() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    architecture = (PROJECT_ROOT / "docs/concepts/architecture.md").read_text()

    assert "`themis.records`" in readme
    assert "`themis.types`" in readme
    assert "`themis.stats`" in readme
    assert "`stats` extra" in readme
    assert "`themis.records`" in architecture
    assert "`themis.types`" in architecture


def test_examples_and_tutorials_use_records_namespace_in_teaching_paths() -> None:
    hello_world = (PROJECT_ROOT / "examples/01_hello_world.py").read_text()
    tutorial = (PROJECT_ROOT / "docs/tutorials/hello-world.md").read_text()
    custom_extractor = (
        PROJECT_ROOT / "examples/03_custom_extractor_metric.py"
    ).read_text()

    assert "from themis.records import InferenceRecord, MetricScore" in hello_world
    assert "from themis.records import InferenceRecord, MetricScore" in tutorial
    assert (
        "from themis.records import ExtractionRecord, InferenceRecord, MetricScore"
        in custom_extractor
    )


def test_plugin_docs_describe_supported_extractor_signature_and_hash_identity() -> None:
    plugins = (PROJECT_ROOT / "docs/guides/plugins.md").read_text()
    specs_and_records = (
        PROJECT_ROOT / "docs/concepts/specs-and-records.md"
    ).read_text()

    assert "def extract(self, trial, candidate, config):" in plugins
    assert (
        "Custom extractors use the signature `(trial, candidate, config)`." in plugins
    )
    assert "12-character public aliases" in specs_and_records
    assert "full canonical hashes" in specs_and_records


def test_installation_docs_cover_all_optional_extras() -> None:
    installation = (PROJECT_ROOT / "docs/installation-setup/index.md").read_text()
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    extras = sorted(pyproject["project"]["optional-dependencies"])

    for extra in extras:
        assert f"### `{extra}`" in installation


def test_guides_index_lists_advanced_workflow_pages() -> None:
    guides_index = (PROJECT_ROOT / "docs/guides/index.md").read_text()

    assert "Advanced Workflows" in guides_index
    assert "external-stage-handoffs.md" in guides_index
    assert "evolve-an-experiment.md" in guides_index
    assert "scaling-execution.md" in guides_index


def test_examples_readme_lists_advanced_workflow_examples() -> None:
    examples_readme = (PROJECT_ROOT / "examples/README.md").read_text()

    assert "08_external_stage_handoff.py" in examples_readme
    assert "09_experiment_evolution.py" in examples_readme

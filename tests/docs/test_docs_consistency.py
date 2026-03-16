from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_quick_start_embeds_the_runnable_hello_world_example() -> None:
    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()

    assert '--8<-- "examples/01_hello_world.py"' in quick_start
    assert "uv run python examples/01_hello_world.py" in quick_start


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
    assert "`uv` is required" in readme
    assert (
        "- `uv` required"
        in (PROJECT_ROOT / "docs/installation-setup/index.md").read_text()
    )


def test_hello_world_example_uses_documented_root_entry_points() -> None:
    hello_world = (PROJECT_ROOT / "examples/01_hello_world.py").read_text()
    tutorial = (PROJECT_ROOT / "docs/tutorials/hello-world.md").read_text()

    assert "from themis import (" in hello_world
    assert "    Orchestrator," in hello_world
    assert "    PluginRegistry," in hello_world
    assert "    SqliteBlobStorageSpec," in hello_world
    assert "    StorageSpec," not in hello_world
    assert "SqliteBlobStorageSpec(" in tutorial
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


def test_guides_index_lists_research_workflow_pages() -> None:
    guides_index = (PROJECT_ROOT / "docs/guides/index.md").read_text()

    assert "Example Catalog" in guides_index
    assert "external-stage-handoffs.md" in guides_index
    assert "evolve-an-experiment.md" in guides_index
    assert "scaling-execution.md" in guides_index


def test_examples_readme_lists_advanced_workflow_examples() -> None:
    examples_readme = (PROJECT_ROOT / "examples/README.md").read_text()

    assert "08_external_stage_handoff.py" in examples_readme
    assert "09_experiment_evolution.py" in examples_readme


def test_readme_and_examples_index_list_all_shipped_examples() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    examples_readme = (PROJECT_ROOT / "examples/README.md").read_text()

    for example_name in [
        "01_hello_world.py",
        "02_project_file.py",
        "03_custom_extractor_metric.py",
        "04_compare_models.py",
        "05_resume_run.py",
        "06_hooks_and_timeline.py",
        "07_judge_metric.py",
        "08_external_stage_handoff.py",
        "09_experiment_evolution.py",
    ]:
        assert example_name in readme
        assert example_name in examples_readme


def test_quick_start_and_guides_use_verified_example_storage_roots() -> None:
    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()
    quickcheck = (PROJECT_ROOT / "docs/guides/quickcheck.md").read_text()
    compare = (PROJECT_ROOT / "docs/guides/compare-and-export.md").read_text()
    analyze = (PROJECT_ROOT / "docs/guides/analyze-results.md").read_text()

    assert ".cache/themis-examples/01-hello-world/themis.sqlite3" in quick_start
    assert ".cache/themis-examples/01-hello-world/themis.sqlite3" in quickcheck
    assert ".cache/themis-examples/04-compare-models/themis.sqlite3" in analyze
    assert ".cache/themis-examples/04-compare-models/report.md" in compare


def test_api_reference_navigation_lists_new_public_surfaces() -> None:
    mkdocs = (PROJECT_ROOT / "mkdocs.yml").read_text()
    api_index = (PROJECT_ROOT / "docs/api-reference/index.md").read_text()

    for nav_entry in [
        "api-reference/errors.md",
        "api-reference/progress.md",
        "api-reference/types.md",
        "api-reference/run-planning.md",
        "api-reference/postgres-storage.md",
    ]:
        assert nav_entry in mkdocs

    for label in [
        "Errors",
        "Progress",
        "Types",
        "Run Planning",
        "Postgres Storage",
    ]:
        assert label in api_index


def test_tutorials_index_keeps_only_beginner_guided_paths() -> None:
    tutorials = (PROJECT_ROOT / "docs/tutorials/index.md").read_text()

    assert "Hello World Walkthrough" in tutorials
    assert "Load a Project File" in tutorials
    assert "Analyze a Stored Run" in tutorials
    assert "Provider-backed Run" in tutorials


def test_guides_index_lists_research_lifecycle_pages() -> None:
    guides = (PROJECT_ROOT / "docs/guides/index.md").read_text()

    for label in [
        "Analyze Results",
        "Validate Dataset Loaders",
        "Reproduce and Share Runs",
        "Build a Provider Engine",
    ]:
        assert label in guides


def test_docs_include_failure_mode_examples_for_optional_deps_and_project_files() -> (
    None
):
    installation = (PROJECT_ROOT / "docs/installation-setup/index.md").read_text()
    project_files = (PROJECT_ROOT / "docs/guides/project-files.md").read_text()

    for expected in [
        "Optional dependency 'zstandard' is required for this feature.",
        "Optional dependency 'jsonschema' is required for this feature.",
        "Optional dependency 'themis.stats.stats_engine' is required for this feature.",
        "Optional dependency 'langfuse' is required for this feature.",
    ]:
        assert expected in installation

    assert "Failed to parse project config broken.json:" in project_files


def test_contributing_captures_docs_qa_and_maintenance_rules() -> None:
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text()

    for expected in [
        "Beginner flow",
        "Researcher flow",
        "Power-user flow",
        "Any new example requires a docs link, expected output, and API-reference decision before merge.",
        "Any new public namespace requires a docs link, expected output, and API-reference decision before merge.",
    ]:
        assert expected in contributing


def test_docs_workflows_run_docs_verification_suite_before_build() -> None:
    docs_workflow = (PROJECT_ROOT / ".github/workflows/docs.yml").read_text()
    ci_workflow = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text()

    verification_command = (
        "uv run pytest tests/docs/test_docs_consistency.py "
        "tests/docs/test_public_docstrings.py "
        "tests/docs/test_documented_workflows.py "
        "tests/docs/test_example_display_paths.py"
    )

    assert verification_command in docs_workflow
    assert verification_command in ci_workflow


def test_tutorials_include_explicit_run_commands() -> None:
    hello_world = (PROJECT_ROOT / "docs/tutorials/hello-world.md").read_text()
    project_files = (PROJECT_ROOT / "docs/tutorials/project-files.md").read_text()
    analyze = (PROJECT_ROOT / "docs/tutorials/analyze-results.md").read_text()
    provider = (PROJECT_ROOT / "docs/tutorials/provider-backed-run.md").read_text()

    assert "uv run python hello_world.py" in hello_world
    assert "uv run python examples/02_project_file.py" in project_files
    assert "uv run python examples/04_compare_models.py" in analyze
    assert 'uv add "themis-eval[providers-openai]"' in provider
    assert "uv run python provider_run.py" in provider


def test_api_reference_marks_internal_detail_pages_and_covers_root_namespaces() -> None:
    orchestration = (PROJECT_ROOT / "docs/api-reference/orchestration.md").read_text()
    storage = (PROJECT_ROOT / "docs/api-reference/storage.md").read_text()
    postgres = (PROJECT_ROOT / "docs/api-reference/postgres-storage.md").read_text()
    errors = (PROJECT_ROOT / "docs/api-reference/errors.md").read_text()
    registry = (PROJECT_ROOT / "docs/api-reference/registry.md").read_text()
    telemetry = (PROJECT_ROOT / "docs/api-reference/telemetry.md").read_text()
    records = (PROJECT_ROOT / "docs/api-reference/records.md").read_text()
    types = (PROJECT_ROOT / "docs/api-reference/types.md").read_text()
    reporting = (PROJECT_ROOT / "docs/api-reference/reporting-and-stats.md").read_text()

    for page in [orchestration, storage, postgres]:
        assert "implementation detail" in page
        assert "stable extension surface" in page

    assert "::: themis.errors" in errors
    assert "::: themis.registry" in registry
    assert "::: themis.telemetry" in telemetry
    assert "::: themis.records" in records
    assert "::: themis.types" in types
    assert "::: themis.stats" in reporting


def test_api_index_and_concept_indexes_cross_link_core_pages() -> None:
    api_index = (PROJECT_ROOT / "docs/api-reference/index.md").read_text()
    concepts_index = (PROJECT_ROOT / "docs/concepts/index.md").read_text()
    guides_index = (PROJECT_ROOT / "docs/guides/index.md").read_text()
    introduction = (PROJECT_ROOT / "docs/introduction/index.md").read_text()

    for link in [
        "(root.md)",
        "(errors.md)",
        "(run-planning.md)",
        "(progress.md)",
        "(registry.md)",
        "(telemetry.md)",
        "(records.md)",
        "(storage.md)",
        "(postgres-storage.md)",
        "(types.md)",
        "(cli.md)",
    ]:
        assert link in api_index

    assert "(architecture.md)" in concepts_index
    assert "(specs-and-records.md)" in concepts_index
    assert "(guides/releasing.md)" not in introduction
    assert "(releasing.md)" in guides_index


def test_cli_reference_is_manual_and_not_auto_doc_only() -> None:
    cli_reference = (PROJECT_ROOT / "docs/api-reference/cli.md").read_text()

    for heading in [
        "## `themis`",
        "## `themis report`",
        "## `themis quickcheck`",
        "## `themis-quickcheck`",
        "Representative failures",
    ]:
        assert heading in cli_reference


def test_resume_and_reproduction_docs_explain_resume_return_types() -> None:
    reproduce = (PROJECT_ROOT / "docs/guides/reproduce-runs.md").read_text()
    resume = (PROJECT_ROOT / "docs/guides/resume-and-inspect.md").read_text()

    for page in [reproduce, resume]:
        assert "RunHandle" in page
        assert "ExperimentResult" in page


def test_dataset_validation_handles_mapping_context_and_scalar_items() -> None:
    validation = (PROJECT_ROOT / "docs/guides/dataset-validation.md").read_text()
    loaders = (PROJECT_ROOT / "docs/guides/dataset-loaders.md").read_text()

    assert "Mapping" in validation
    assert "DataItemContext" in validation
    assert "scalar item" in validation
    assert "duck-typing" in validation
    assert "DataItemContext" in loaders


def test_provider_and_config_report_guides_explain_failure_modes_and_degradation() -> (
    None
):
    provider = (PROJECT_ROOT / "docs/guides/provider-engines.md").read_text()
    config_reports = (PROJECT_ROOT / "docs/guides/config-reports.md").read_text()

    assert "RetryableProviderError" in provider
    assert "InferenceError" in provider
    assert "Expected success output pattern" in provider
    assert "dynamic classes" in config_reports
    assert "third-party classes" in config_reports
    assert "compiled extensions" in config_reports

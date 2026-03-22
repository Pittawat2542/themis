from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_readme_and_intro_are_benchmark_first() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    intro = (PROJECT_ROOT / "docs/introduction/index.md").read_text()

    assert "BenchmarkSpec" in readme
    assert "BenchmarkResult" in readme
    assert "ExperimentSpec" not in readme
    assert "DatasetProvider.scan(slice_spec, query)" in intro
    assert "ExperimentSpec" not in intro


def test_quick_start_promotes_cli_onboarding_and_example_follow_up() -> None:
    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()

    assert "themis quick-eval inline" in quick_start
    assert "themis init starter-eval" in quick_start
    assert (
        ".cache/themis/quick-eval/inline-demo-model-exact-match/themis.sqlite3"
        in quick_start
    )
    assert '--8<-- "examples/01_hello_world.py"' in quick_start
    assert "uv run python examples/01_hello_world.py" in quick_start
    assert (
        ".cache/themis-examples/01-hello-world-benchmark-first/themis.sqlite3"
        in quick_start
    )
    assert "list_catalog_benchmarks()" in quick_start
    assert (
        "data/sample.jsonl"
        not in quick_start.split("That builtin scaffold includes:")[1]
    )


def test_readme_and_examples_index_list_all_numbered_examples() -> None:
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
        "10_agent_eval.py",
        "11_quick_benchmark.py",
        "12_iter_and_estimate.py",
        "13_catalog_builtin_benchmark.py",
        "14_mcp_openai.py",
    ]:
        assert example_name in readme
        assert example_name in examples_readme


def test_builtin_catalog_docs_include_discovery_and_python_entrypoint() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    quick_start = (PROJECT_ROOT / "docs/quick-start/index.md").read_text()
    benchmark_catalog = (PROJECT_ROOT / "docs/guides/builtin-benchmarks.md").read_text()
    cli_reference = (PROJECT_ROOT / "docs/api-reference/cli.md").read_text()

    assert "list_catalog_benchmarks()" in readme
    assert "build_catalog_benchmark_project(...)" in quick_start
    assert "list_catalog_benchmarks()" in benchmark_catalog
    assert "build_catalog_benchmark_project(...)" in benchmark_catalog
    assert "`mmlu_pro`" in benchmark_catalog
    assert "builtin benchmark-backed starter" in cli_reference
    assert "default local-data template" in cli_reference


def test_example_catalog_marks_medical_reasoning_eval_as_non_recommended() -> None:
    example_catalog = (PROJECT_ROOT / "docs/guides/examples.md").read_text()
    examples_readme = (PROJECT_ROOT / "examples/README.md").read_text()

    assert "medical_reasoning_eval" in example_catalog
    assert "recommended public example path" in example_catalog
    assert "medical_reasoning_eval" in examples_readme


def test_guides_and_tutorials_use_benchmark_first_terms() -> None:
    tutorials = (PROJECT_ROOT / "docs/tutorials/index.md").read_text()
    guides = (PROJECT_ROOT / "docs/guides/index.md").read_text()
    quickcheck = (PROJECT_ROOT / "docs/guides/quickcheck.md").read_text()

    assert "BenchmarkSpec" in tutorials
    assert "Builtin Benchmarks" in guides
    assert "Build a Dataset Provider" in guides
    assert "--slice qa" in quickcheck
    assert "--dimension source=synthetic" in quickcheck
    assert "--task" not in quickcheck


def test_api_navigation_lists_benchmark_first_reference_pages() -> None:
    mkdocs = (PROJECT_ROOT / "mkdocs.yml").read_text()
    api_index = (PROJECT_ROOT / "docs/api-reference/index.md").read_text()

    for nav_entry in [
        "api-reference/specs.md",
        "api-reference/orchestration.md",
        "api-reference/runtime.md",
        "api-reference/registry.md",
        "api-reference/protocols.md",
        "api-reference/cli.md",
        "guides/builtin-benchmarks.md",
    ]:
        assert nav_entry in mkdocs

    for label in [
        "Specs",
        "Orchestration",
        "Runtime",
        "Registry",
        "Protocols",
        "CLI",
    ]:
        assert label in api_index

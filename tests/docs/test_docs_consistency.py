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


def test_installation_docs_cover_all_optional_extras() -> None:
    installation = (PROJECT_ROOT / "docs/installation-setup/index.md").read_text()
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    extras = sorted(pyproject["project"]["optional-dependencies"])

    for extra in extras:
        assert f"### `{extra}`" in installation

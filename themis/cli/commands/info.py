"""System information and listing commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from themis.providers.registry import _REGISTRY


def show_info() -> int:
    """Show system information and installed components."""
    import themis
    from themis import _version

    print("Themis Information")
    print("=" * 60)
    print(f"Version: {getattr(_version, '__version__', 'unknown')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    print("\n📦 Installed Providers:")
    providers = sorted(_REGISTRY._factories.keys())
    for provider in providers:
        print(f"  ✓ {provider}")

    print("\n📊 Available Benchmarks:")
    benchmarks = [
        "demo",
        "math500",
        "aime24",
        "aime25",
        "amc23",
        "olympiadbench",
        "beyondaime",
        "supergpqa",
        "mmlu-pro",
        "inline (via config)",
    ]
    for bench in benchmarks:
        print(f"  ✓ {bench}")

    print("\n📁 Example Locations:")
    examples_dir = Path(themis.__file__).parent.parent / "examples"
    if examples_dir.exists():
        print(f"  {examples_dir}")
        example_dirs = sorted(
            [
                d.name
                for d in examples_dir.iterdir()
                if d.is_dir() and not d.name.startswith("_")
            ]
        )
        for ex in example_dirs:
            print(f"    • {ex}/")

    print("\n📚 Documentation:")
    print("  examples/README.md - Comprehensive tutorial cookbook")
    print("  COOKBOOK.md - Quick reference guide")
    print("  docs/ - Detailed documentation")

    print("\n🚀 Quick Start:")
    print("  uv run python -m themis.cli demo")
    print("  uv run python -m themis.cli list-providers")
    print("  uv run python -m themis.cli list-benchmarks")

    return 0


def new_project(
    *,
    project_name: Annotated[str, Parameter(help="The name of the new project")],
    project_path: Annotated[
        Path,
        Parameter(help="The path where the new project will be created"),
    ] = Path("."),
) -> int:
    """Create a new Themis project."""
    from themis.cli.new_project import create_project

    try:
        create_project(project_name, project_path)
        print(f"Successfully created new project '{project_name}' in {project_path}")
        return 0
    except FileExistsError as e:
        print(f"Error: {e}")
        return 1

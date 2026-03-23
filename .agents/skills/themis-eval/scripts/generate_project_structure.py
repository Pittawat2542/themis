#!/usr/bin/env python3
"""Generate an ideal Themis project file tree in a target folder."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_FILES = {
    "project.toml": 'project_name = "starter_eval"\n',
    ".env.example": "THEMIS_CATALOG_PROVIDER=demo\nTHEMIS_CATALOG_MODEL=demo-model\n",
    "README.md": "# Starter Eval\n",
    "data/sample.jsonl": "",
    "__init__.py": '"""Catalog benchmark package."""\n',
    "__main__.py": "from starter_eval.app import main\n",
    "app.py": '"""Project entrypoint."""\n',
    "settings.py": '"""Environment-backed project settings."""\n',
    "registry.py": '"""Registry builder for benchmark runs."""\n',
    "benchmarks/__init__.py": "from .default import build_benchmark\n",
    "benchmarks/default.py": '"""Benchmark authoring entrypoint."""\n',
    "datasets/__init__.py": "from .local_file import build_dataset_provider\n",
    "datasets/local_file.py": '"""Local dataset-provider entrypoint."""\n',
}

BUILTIN_FILES = {
    "project.toml": 'project_name = "starter_eval"\n',
    ".env.example": (
        "THEMIS_CATALOG_PROVIDER=demo\n"
        "THEMIS_CATALOG_MODEL=demo-model\n"
        "THEMIS_CATALOG_BENCHMARK=mmlu_pro\n"
    ),
    "README.md": "# Starter Eval\n",
    "__init__.py": '"""Catalog benchmark package."""\n',
    "__main__.py": "from starter_eval.app import main\n",
    "app.py": '"""Project entrypoint."""\n',
    "settings.py": '"""Environment-backed project settings."""\n',
    "registry.py": '"""Registry builder for benchmark runs."""\n',
    "benchmarks/__init__.py": (
        "from .default import build_benchmark, render_preview, summarize_result\n"
    ),
    "benchmarks/default.py": '"""Built-in benchmark wrapper entrypoint."""\n',
    "datasets/__init__.py": "from .builtin import build_dataset_provider\n",
    "datasets/builtin.py": '"""Built-in dataset-provider entrypoint."""\n',
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Themis starter project structure."
    )
    parser.add_argument("--target", required=True, help="Target folder to create.")
    parser.add_argument(
        "--package-name",
        required=True,
        help="Python package name to scaffold under the target folder.",
    )
    parser.add_argument(
        "--mode",
        choices=("default", "builtin"),
        default="default",
        help="Scaffold shape to generate.",
    )
    return parser.parse_args()


def _build_manifest(mode: str) -> dict[str, str]:
    if mode == "builtin":
        return BUILTIN_FILES
    return DEFAULT_FILES


def _write_structure(target: Path, package_name: str, manifest: dict[str, str]) -> None:
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Target directory must be empty: {target}")

    target.mkdir(parents=True, exist_ok=True)
    package_root = target / package_name

    for relative_path, contents in manifest.items():
        destination = (
            package_root / relative_path
            if not relative_path.startswith((".", "README", "project.toml", "data/"))
            else target / relative_path
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents.replace("starter_eval", package_name))


def main() -> int:
    args = _parse_args()
    _write_structure(
        target=Path(args.target),
        package_name=args.package_name,
        manifest=_build_manifest(args.mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

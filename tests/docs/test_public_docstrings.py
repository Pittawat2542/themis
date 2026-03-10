from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCSTRING_TARGETS = [
    PROJECT_ROOT / "themis/contracts/protocols.py",
    PROJECT_ROOT / "themis/orchestration/orchestrator.py",
    PROJECT_ROOT / "themis/orchestration/executor.py",
    PROJECT_ROOT / "themis/orchestration/trial_runner.py",
    PROJECT_ROOT / "themis/orchestration/trial_planner.py",
    PROJECT_ROOT / "themis/orchestration/projection_handler.py",
    PROJECT_ROOT / "themis/orchestration/candidate_pipeline.py",
    PROJECT_ROOT / "themis/runtime/experiment_result.py",
    PROJECT_ROOT / "themis/runtime/comparison.py",
    PROJECT_ROOT / "themis/registry/plugin_registry.py",
    PROJECT_ROOT / "themis/storage/event_repo.py",
    PROJECT_ROOT / "themis/storage/projection_repo.py",
    PROJECT_ROOT / "themis/storage/artifact_store.py",
    PROJECT_ROOT / "themis/storage/sqlite_schema.py",
    PROJECT_ROOT / "themis/report/builder.py",
    PROJECT_ROOT / "themis/report/exporters.py",
    PROJECT_ROOT / "themis/cli/quickcheck.py",
    PROJECT_ROOT / "themis/extractors/builtin.py",
    PROJECT_ROOT / "themis/stats/stats_engine.py",
    PROJECT_ROOT / "themis/specs/experiment.py",
    PROJECT_ROOT / "themis/specs/foundational.py",
]


def test_public_reference_modules_have_public_docstrings() -> None:
    missing: list[str] = []

    for path in DOCSTRING_TARGETS:
        tree = ast.parse(path.read_text())
        if ast.get_docstring(tree) is None:
            missing.append(f"{path.relative_to(PROJECT_ROOT)}:module")

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(PROJECT_ROOT)}:{node.name}")
                for child in node.body:
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and not child.name.startswith("_"):
                        if ast.get_docstring(child) is None:
                            missing.append(
                                f"{path.relative_to(PROJECT_ROOT)}:{node.name}.{child.name}"
                            )
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(PROJECT_ROOT)}:{node.name}")

    assert missing == []

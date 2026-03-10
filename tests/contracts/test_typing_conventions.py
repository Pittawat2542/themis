from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPING_TARGETS = [
    PROJECT_ROOT / "themis/contracts/protocols.py",
    PROJECT_ROOT / "themis/evaluation/judge_service.py",
    PROJECT_ROOT / "themis/orchestration/candidate_pipeline.py",
    PROJECT_ROOT / "themis/orchestration/executor.py",
    PROJECT_ROOT / "themis/orchestration/trial_runner.py",
    PROJECT_ROOT / "themis/records/base.py",
    PROJECT_ROOT / "themis/records/candidate.py",
    PROJECT_ROOT / "themis/records/conversation.py",
    PROJECT_ROOT / "themis/records/error.py",
    PROJECT_ROOT / "themis/records/evaluation.py",
    PROJECT_ROOT / "themis/records/extraction.py",
    PROJECT_ROOT / "themis/records/inference.py",
    PROJECT_ROOT / "themis/records/report.py",
    PROJECT_ROOT / "themis/records/trial.py",
    PROJECT_ROOT / "themis/registry/plugin_registry.py",
    PROJECT_ROOT / "themis/report/builder.py",
    PROJECT_ROOT / "themis/runtime/comparison.py",
    PROJECT_ROOT / "themis/runtime/experiment_result.py",
    PROJECT_ROOT / "themis/stats/stats_engine.py",
    PROJECT_ROOT / "themis/storage/artifact_store.py",
    PROJECT_ROOT / "themis/types/events.py",
]


def _annotation_uses_name(annotation: ast.AST | None, target: str) -> bool:
    if annotation is None:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == target
        for node in ast.walk(annotation)
    )


def test_v2_core_modules_avoid_any_and_optional_annotations() -> None:
    violations: list[str] = []

    for path in TYPING_TARGETS:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                annotation = node.annotation
                label = f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotation = node.returns
                label = f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
            elif isinstance(node, ast.AnnAssign):
                annotation = node.annotation
                label = f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
            else:
                continue

            if _annotation_uses_name(annotation, "Any"):
                violations.append(f"{label} uses Any")
            if _annotation_uses_name(annotation, "Optional"):
                violations.append(f"{label} uses Optional")

    assert violations == []

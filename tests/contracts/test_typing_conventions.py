from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPING_TARGETS = [
    PROJECT_ROOT / "themis/__init__.py",
    PROJECT_ROOT / "themis/contracts/protocols.py",
    PROJECT_ROOT / "themis/records/__init__.py",
    PROJECT_ROOT / "themis/records/base.py",
    PROJECT_ROOT / "themis/records/candidate.py",
    PROJECT_ROOT / "themis/records/conversation.py",
    PROJECT_ROOT / "themis/records/error.py",
    PROJECT_ROOT / "themis/records/evaluation.py",
    PROJECT_ROOT / "themis/records/extraction.py",
    PROJECT_ROOT / "themis/records/inference.py",
    PROJECT_ROOT / "themis/records/report.py",
    PROJECT_ROOT / "themis/records/trial.py",
    PROJECT_ROOT / "themis/registry/__init__.py",
    PROJECT_ROOT / "themis/registry/plugin_registry.py",
    PROJECT_ROOT / "themis/report/builder.py",
    PROJECT_ROOT / "themis/runtime/comparison.py",
    PROJECT_ROOT / "themis/runtime/experiment_result.py",
    PROJECT_ROOT / "themis/specs/__init__.py",
    PROJECT_ROOT / "themis/stats/__init__.py",
    PROJECT_ROOT / "themis/stats/stats_engine.py",
    PROJECT_ROOT / "themis/storage/artifact_store.py",
    PROJECT_ROOT / "themis/telemetry/langfuse_callback.py",
    PROJECT_ROOT / "themis/types/__init__.py",
    PROJECT_ROOT / "themis/types/events.py",
]
STRICT_NO_ANY_TARGETS = [
    PROJECT_ROOT / "themis/records/__init__.py",
    PROJECT_ROOT / "themis/report/builder.py",
    PROJECT_ROOT / "themis/stats/__init__.py",
    PROJECT_ROOT / "themis/stats/stats_engine.py",
    PROJECT_ROOT / "themis/telemetry/langfuse_callback.py",
    PROJECT_ROOT / "themis/types/__init__.py",
]
REQUIRED_RETURN_ANNOTATIONS = {
    PROJECT_ROOT / "themis/report/builder.py": {"_extract_metric_rows"},
    PROJECT_ROOT / "themis/records/__init__.py": {"__dir__", "__getattr__"},
    PROJECT_ROOT / "themis/stats/__init__.py": {"__dir__", "__getattr__"},
    PROJECT_ROOT / "themis/types/__init__.py": {"__dir__", "__getattr__"},
}


def _annotation_uses_name(annotation: ast.AST | None, target: str) -> bool:
    if annotation is None:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == target
        for node in ast.walk(annotation)
    )


def _call_is_cast_any(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "cast":
        return False
    if not node.args:
        return False
    first_arg = node.args[0]
    return isinstance(first_arg, ast.Name) and first_arg.id == "Any"


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


def test_selected_runtime_modules_avoid_any_imports_and_cast_any() -> None:
    violations: list[str] = []

    for path in STRICT_NO_ANY_TARGETS:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                if any(alias.name == "Any" for alias in node.names):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} imports Any")
            if _call_is_cast_any(node):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} uses cast(Any, ...)"
                )

    assert violations == []


def test_selected_runtime_helpers_have_explicit_return_annotations() -> None:
    violations: list[str] = []

    for path, function_names in REQUIRED_RETURN_ANNOTATIONS.items():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in function_names
                and node.returns is None
            ):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} missing return annotation"
                )

    assert violations == []

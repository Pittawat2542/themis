from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
PATH_SUFFIXES = ("_path", "_dir", "_root")
DISPLAY_FORMATTERS = {"_format_display_path"}


def _is_path_like_name(name: str) -> bool:
    return name.endswith(PATH_SUFFIXES)


def _find_raw_printed_path_vars(source: str) -> set[str]:
    tree = ast.parse(source)
    path_vars: set[str] = set()
    violations: set[str] = set()

    def is_path_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in path_vars or _is_path_like_name(node.id)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "Path":
                return True
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return is_path_expr(node.left) or is_path_expr(node.right)
        return False

    def is_safe_display_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISPLAY_FORMATTERS:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "as_posix":
                return True
        return False

    def collect_raw_path_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        if is_safe_display_expr(node):
            return names
        if isinstance(node, ast.Name) and is_path_expr(node):
            names.add(node.id)
            return names
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    names.update(collect_raw_path_names(value.value))
            return names
        for child in ast.iter_child_nodes(node):
            names.update(collect_raw_path_names(child))
        return names

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and is_path_expr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    path_vars.add(target.id)
        if isinstance(node, ast.AnnAssign) and node.value and is_path_expr(node.value):
            if isinstance(node.target, ast.Name):
                path_vars.add(node.target.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "print":
            continue
        for arg in node.args:
            violations.update(collect_raw_path_names(arg))

    return violations


def test_path_print_detector_flags_raw_path_variables() -> None:
    source = """
from pathlib import Path

report_path = Path(".cache/report.md")
print("Report written to:", report_path)
"""

    assert _find_raw_printed_path_vars(source) == {"report_path"}


def test_path_print_detector_allows_explicit_display_formatting() -> None:
    source = """
from pathlib import Path

report_path = Path(".cache/report.md")

def _format_display_path(path: Path) -> str:
    return path.as_posix()

print("Report written to:", _format_display_path(report_path))
print(f"Report written to: {report_path.as_posix()}")
"""

    assert _find_raw_printed_path_vars(source) == set()


def test_examples_do_not_print_raw_path_objects() -> None:
    violations_by_file: dict[str, set[str]] = {}

    for example_path in sorted(EXAMPLES_DIR.glob("*.py")):
        violations = _find_raw_printed_path_vars(example_path.read_text())
        if violations:
            violations_by_file[str(example_path.relative_to(PROJECT_ROOT))] = violations

    assert violations_by_file == {}

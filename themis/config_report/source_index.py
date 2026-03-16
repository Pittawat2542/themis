"""Best-effort source and comment indexing for config-report metadata."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import tokenize


@dataclass(frozen=True)
class ClassSourceInfo:
    """Source metadata for one class definition."""

    qualname: str
    source_file: str
    source_line: int | None
    docstring: str | None
    field_lines: dict[str, int] = field(default_factory=dict)
    field_comments: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceIndex:
    """Indexed class and field metadata for one Python source file."""

    classes: dict[str, ClassSourceInfo]

    def get_class_info(self, qualname: str) -> ClassSourceInfo | None:
        """Return source metadata for one qualified class name."""

        return self.classes.get(qualname)


def _extract_comment(
    lines: list[str], comment_by_line: dict[int, list[str]], line: int
) -> str | None:
    inline = comment_by_line.get(line, [])
    if inline:
        return " ".join(comment.strip().lstrip("#").strip() for comment in inline)

    collected: list[str] = []
    current = line - 1
    while current >= 1:
        raw = lines[current - 1].strip()
        if not raw:
            if collected:
                break
            current -= 1
            continue
        if raw.startswith("#"):
            collected.append(raw.lstrip("#").strip())
            current -= 1
            continue
        break
    if collected:
        return " ".join(reversed(collected))
    return None


def _class_field_lines(node: ast.ClassDef) -> dict[str, int]:
    field_lines: dict[str, int] = {}
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            field_lines[child.target.id] = child.lineno
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    field_lines[target.id] = child.lineno
    return field_lines


def _walk_classes(
    node: ast.AST,
    *,
    prefix: str,
    source_file: str,
    lines: list[str],
    comment_by_line: dict[int, list[str]],
    classes: dict[str, ClassSourceInfo],
) -> None:
    for child in getattr(node, "body", []):
        if not isinstance(child, ast.ClassDef):
            continue
        qualname = f"{prefix}.{child.name}" if prefix else child.name
        field_lines = _class_field_lines(child)
        field_comments = {
            field_name: comment
            for field_name, line in field_lines.items()
            if (comment := _extract_comment(lines, comment_by_line, line)) is not None
        }
        classes[qualname] = ClassSourceInfo(
            qualname=qualname,
            source_file=source_file,
            source_line=child.lineno,
            docstring=ast.get_docstring(child),
            field_lines=field_lines,
            field_comments=field_comments,
        )
        _walk_classes(
            child,
            prefix=qualname,
            source_file=source_file,
            lines=lines,
            comment_by_line=comment_by_line,
            classes=classes,
        )


@lru_cache(maxsize=None)
def load_source_index(path: str) -> SourceIndex:
    """Parse and index one Python source file."""

    file_path = Path(path)
    source = file_path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)

    comment_by_line: dict[int, list[str]] = {}
    for token in tokenize.generate_tokens(
        iter(source.splitlines(keepends=True)).__next__
    ):
        if token.type == tokenize.COMMENT:
            comment_by_line.setdefault(token.start[0], []).append(token.string)

    classes: dict[str, ClassSourceInfo] = {}
    _walk_classes(
        tree,
        prefix="",
        source_file=str(file_path),
        lines=lines,
        comment_by_line=comment_by_line,
        classes=classes,
    )
    return SourceIndex(classes=classes)

"""Renderers for configuration report documents."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Protocol

from themis.config_report.models import ConfigReportDocument, ConfigReportNode
from themis.config_report.types import ConfigReportFormat


class ConfigReportRenderer(Protocol):
    """Renderer interface for configuration report formats."""

    def render(self, document: ConfigReportDocument) -> str:
        """Render the document into one output string."""
        ...


def _document_payload(document: ConfigReportDocument) -> dict[str, object]:
    return document.model_dump(mode="json")


class JsonConfigReportRenderer:
    """Render configuration reports as structured JSON."""

    def render(self, document: ConfigReportDocument) -> str:
        """Return one indented JSON document."""

        return json.dumps(_document_payload(document), indent=2, sort_keys=False) + "\n"


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#[]{}&*!?|>'\"%@`\\\n"):
        return json.dumps(text)
    return text


def _render_yaml(value: object, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_render_yaml(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_render_yaml(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


class YamlConfigReportRenderer:
    """Render configuration reports as deterministic YAML."""

    def render(self, document: ConfigReportDocument) -> str:
        """Return one YAML document."""

        return "\n".join(_render_yaml(_document_payload(document))) + "\n"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _render_markdown_table(node: ConfigReportNode) -> str:
    if not node.parameters:
        return "_No parameters_\n"
    lines = [
        "| Parameter | Value | Type | Default | Declared In | Source | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for parameter in node.parameters:
        source = ""
        if parameter.source_file and parameter.source_line:
            source = f"{Path(parameter.source_file).name}:{parameter.source_line}"
        notes = " ".join(
            part for part in [parameter.doc, parameter.inline_comment] if part
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(parameter.name),
                    _markdown_cell(parameter.value),
                    _markdown_cell(parameter.type_repr),
                    _markdown_cell(parameter.default if parameter.has_default else ""),
                    _markdown_cell(parameter.declared_in or ""),
                    _markdown_cell(source),
                    _markdown_cell(notes),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown_node(node: ConfigReportNode) -> list[str]:
    lines = [
        "<details>",
        f"<summary><strong>{node.name}</strong> <code>{node.path}</code></summary>",
        "",
    ]
    if node.docstring:
        lines.extend([node.docstring, ""])
    lines.append(_render_markdown_table(node))
    for child in node.children:
        lines.extend(_render_markdown_node(child))
    lines.extend(["</details>", ""])
    return lines


class MarkdownConfigReportRenderer:
    """Render configuration reports as nested Markdown details blocks."""

    def render(self, document: ConfigReportDocument) -> str:
        """Return one Markdown document."""

        header = document.header
        lines = [
            "# Configuration Report",
            "",
            f"- Generated At: {header.generated_at}",
            f"- Git Commit: {header.git_commit or ''}",
            f"- Project Name: {header.project_name or ''}",
            f"- Entrypoint: {header.entrypoint or ''}",
            f"- Root Type: {header.root_type}",
            f"- Verbosity: {header.verbosity}",
            "",
        ]
        lines.extend(_render_markdown_node(document.root))
        return "\n".join(lines).rstrip() + "\n"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(source) for source in replacements))
    text = pattern.sub(lambda match: replacements[match.group(0)], text)
    return text.replace("\n", r"\\")


def _latex_section(depth: int) -> str:
    if depth <= 1:
        return r"\subsection*"
    if depth == 2:
        return r"\subsubsection*"
    return r"\paragraph*"


def _render_latex_table(node: ConfigReportNode) -> str:
    lines = [
        r"\begin{longtable}{p{0.14\linewidth}p{0.14\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.14\linewidth}p{0.12\linewidth}p{0.18\linewidth}}",
        r"\textbf{Parameter} & \textbf{Value} & \textbf{Type} & \textbf{Default} & \textbf{Declared In} & \textbf{Source} & \textbf{Notes} \\",
        r"\hline",
    ]
    if not node.parameters:
        lines.append(r"\multicolumn{7}{l}{\textit{No parameters}} \\")
    for parameter in node.parameters:
        source = ""
        if parameter.source_file and parameter.source_line:
            source = f"{Path(parameter.source_file).name}:{parameter.source_line}"
        notes = " ".join(
            part for part in [parameter.doc, parameter.inline_comment] if part
        )
        lines.append(
            " & ".join(
                [
                    _latex_escape(parameter.name),
                    _latex_escape(parameter.value),
                    _latex_escape(parameter.type_repr),
                    _latex_escape(parameter.default if parameter.has_default else ""),
                    _latex_escape(parameter.declared_in or ""),
                    _latex_escape(source),
                    _latex_escape(notes),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", ""])
    return "\n".join(lines)


def _render_latex_node(node: ConfigReportNode) -> list[str]:
    lines = [f"{_latex_section(node.depth)}{{{_latex_escape(node.name)}}}", ""]
    lines.append(r"\texttt{" + _latex_escape(node.path) + "}")
    lines.append("")
    if node.docstring:
        lines.extend([_latex_escape(node.docstring), ""])
    lines.append(_render_latex_table(node))
    for child in node.children:
        lines.extend(_render_latex_node(child))
    return lines


class LatexConfigReportRenderer:
    """Render configuration reports as LaTeX appendix-ready sections."""

    def render(self, document: ConfigReportDocument) -> str:
        """Return one LaTeX document fragment."""

        header = document.header
        lines = [
            r"\section*{Configuration Report}",
            "",
            r"\begin{description}",
            rf"\item[Generated At] {_latex_escape(header.generated_at)}",
            rf"\item[Git Commit] {_latex_escape(header.git_commit or '')}",
            rf"\item[Project Name] {_latex_escape(header.project_name or '')}",
            rf"\item[Entrypoint] {_latex_escape(header.entrypoint or '')}",
            rf"\item[Root Type] {_latex_escape(header.root_type)}",
            rf"\item[Verbosity] {_latex_escape(header.verbosity)}",
            r"\end{description}",
            "",
        ]
        lines.extend(_render_latex_node(document.root))
        return "\n".join(lines).rstrip() + "\n"


_RENDERERS: dict[ConfigReportFormat | str, ConfigReportRenderer] = {
    "json": JsonConfigReportRenderer(),
    "yaml": YamlConfigReportRenderer(),
    "markdown": MarkdownConfigReportRenderer(),
    "latex": LatexConfigReportRenderer(),
}


def register_config_report_renderer(
    name: str,
    renderer: ConfigReportRenderer,
    *,
    overwrite: bool = False,
) -> None:
    """Register one config-report renderer under a stable format name."""

    if not overwrite and name in _RENDERERS:
        raise ValueError(f"Config report renderer '{name}' is already registered.")
    _RENDERERS[name] = renderer


def get_config_report_renderer(name: ConfigReportFormat | str) -> ConfigReportRenderer:
    """Return the renderer registered for one format name."""

    try:
        return _RENDERERS[name]
    except KeyError as exc:
        supported_formats = ", ".join(sorted(_RENDERERS))
        raise ValueError(
            f"Unsupported config report format '{name}'. "
            f"Expected one of: {supported_formats}."
        ) from exc


def list_config_report_renderers() -> tuple[str, ...]:
    """Return the registered config-report renderer names."""

    return tuple(_RENDERERS)

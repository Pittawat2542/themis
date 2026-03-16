"""Public API for generating configuration reports."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from themis.config_report.collector import build_config_report_document
from themis.config_report.models import ConfigReportDocument
from themis.config_report.renderers import get_config_report_renderer
from themis.config_report.types import ConfigReportFormat, ConfigReportVerbosity
from themis.config_report.visibility import apply_verbosity


def render_config_report(
    document: ConfigReportDocument,
    *,
    format: ConfigReportFormat | str = "markdown",
    verbosity: ConfigReportVerbosity = "default",
) -> str:
    """Render one config report document into the requested format."""

    renderer = get_config_report_renderer(format)
    return renderer.render(apply_verbosity(document, verbosity=verbosity))


def generate_config_report(
    config: object,
    format: ConfigReportFormat | str = "markdown",
    output: str | PathLike[str] | None = None,
    *,
    entrypoint: str | None = None,
    verbosity: ConfigReportVerbosity = "default",
) -> str:
    """Collect and render one nested configuration report."""

    document = build_config_report_document(config, entrypoint=entrypoint)
    rendered = render_config_report(document, format=format, verbosity=verbosity)
    if output is not None:
        Path(output).write_text(rendered, encoding="utf-8")
    return rendered

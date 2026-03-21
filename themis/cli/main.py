"""Parent CLI for Themis commands."""

from __future__ import annotations

from cyclopts import App

from themis import __version__
from themis.cli._common import invoke_app
from themis.cli.init import build_app as build_init_app
from themis.cli.quick_eval import build_app as build_quick_eval_app
from themis.cli.quickcheck import build_app as build_quickcheck_app
from themis.cli.report import build_app as build_report_app


def build_app() -> App:
    """Build the top-level Themis Cyclopts app."""

    app = App(
        name="themis",
        help="Themis benchmark-first CLI.",
        version=__version__,
    )
    app.command(build_quick_eval_app())
    app.command(build_quickcheck_app())
    app.command(build_report_app())
    app.command(build_init_app())
    return app


def main(argv: list[str] | None = None) -> int:
    """Run the top-level Themis CLI."""

    return invoke_app(build_app(), argv)

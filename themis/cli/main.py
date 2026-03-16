"""Parent CLI for Themis commands."""

from __future__ import annotations

import argparse

from themis.cli import quickcheck as quickcheck_cli
from themis.cli import report as report_cli


def _system_exit_code(code: object) -> int:
    """Normalize ``SystemExit.code`` to a shell exit status."""

    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level Themis CLI parser."""

    parser = argparse.ArgumentParser(prog="themis")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    quickcheck_cli.add_quickcheck_subparser(subparsers)
    report_cli.add_report_subparser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the top-level Themis CLI."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return _system_exit_code(exc.code)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("Unknown command.")
    try:
        return handler(args)
    except SystemExit as exc:
        return _system_exit_code(exc.code)

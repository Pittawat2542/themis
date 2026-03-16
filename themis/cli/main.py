"""Parent CLI for Themis commands."""

from __future__ import annotations

import argparse

from themis.cli import quickcheck as quickcheck_cli
from themis.cli import report as report_cli


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
        return int(exc.code)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("Unknown command.")
        return 2
    try:
        return handler(args)
    except SystemExit as exc:
        return int(exc.code)

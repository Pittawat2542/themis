"""Themis CLI entrypoints."""

from themis.cli.app import app


def main() -> None:
    """Invoke the top-level `themis` command-line application."""

    app()


__all__ = ["app", "main"]

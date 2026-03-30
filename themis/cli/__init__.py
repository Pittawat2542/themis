"""Themis CLI entrypoints."""

from themis.cli.app import app


def main() -> None:
    app()


__all__ = ["app", "main"]

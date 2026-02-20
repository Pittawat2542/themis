"""Entry point for running themis.cli as a module."""

from .main import app

if __name__ == "__main__":
    raise SystemExit(app())

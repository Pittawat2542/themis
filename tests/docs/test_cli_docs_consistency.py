from __future__ import annotations

import inspect
from pathlib import Path

from themis.cli import main as cli_main
from themis.cli.commands.eval_commands import eval_command


def test_eval_cli_docs_match_current_signature():
    docs_path = Path("docs/guides/cli.md")
    text = docs_path.read_text(encoding="utf-8")
    signature = inspect.signature(eval_command)

    assert "distributed" not in signature.parameters
    assert "--distributed" not in text

    assert "workers" in signature.parameters
    assert "--workers" in text


def test_cli_docs_cover_source_checkout_and_server_extra_name():
    text = Path("docs/guides/cli.md").read_text(encoding="utf-8")

    assert "uv run python -m themis.cli" in text
    assert "themis-eval[server]" in text


def test_compare_help_command_is_callable():
    try:
        cli_main.app(["compare", "--help"])
    except SystemExit as e:
        assert e.code == 0

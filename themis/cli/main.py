"""Simplified CLI for Themis - seven focused commands.

This is the unified CLI that leverages the themis.evaluate() API.
It replaces 20+ commands with a smaller, task-oriented set.
"""

from __future__ import annotations

from cyclopts import App

from themis._version import __version__
from themis.cli.commands.comparison_commands import compare_command
from themis.cli.commands.data_commands import (
    clean_command,
    list_command,
    share_command,
)
from themis.cli.commands.eval_commands import demo_command, eval_command
from themis.cli.commands.server_commands import serve_command

# Import provider modules to ensure they register themselves
try:
    from themis.generation import clients  # noqa: F401 - registers fake provider
    from themis.providers import (
        litellm_provider,  # noqa: F401
        vllm_provider,  # noqa: F401
    )
except ImportError:
    pass

app = App(
    name="themis",
    help="Dead simple LLM evaluation platform",
    version=__version__,
)

app.command(demo_command, name="demo")
app.command(eval_command, name="eval")
app.command(compare_command, name="compare")
app.command(share_command, name="share")
app.command(serve_command, name="serve")
app.command(list_command, name="list")
app.command(clean_command, name="clean")

"""Themis experiment platform - Dead simple LLM evaluation.

The primary interface is the `evaluate()` function:

    import themis
    report = themis.evaluate("math500", model="gpt-4", limit=100)
"""

from themis import config, core, evaluation, experiment, generation, project
from themis._version import __version__
from themis.api import evaluate

__all__ = [
    # Main API
    "evaluate",
    # Submodules
    "config",
    "core",
    "evaluation",
    "experiment",
    "generation",
    "project",
    # Version
    "__version__",
]

"""Progress reporting utilities for Themis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from themis.utils import logging_utils
from themis.utils.progress.base import ProgressReporter
from themis.utils.progress.composite import CompositeProgressReporter
from themis.utils.progress.log import LogProgressReporter
from themis.utils.progress.rich import RichProgressReporter
from themis.utils.progress.wandb import WandBProgressReporter

if TYPE_CHECKING:
    from themis.experiment.integration_manager import IntegrationManager


def get_progress_reporter(
    integration_manager: IntegrationManager | None = None,
) -> ProgressReporter:
    """Get the appropriate progress reporter based on logging configuration.

    Args:
        integration_manager: Optional integration manager for WandB reporting
    """
    reporters: list[ProgressReporter] = []

    fmt = logging_utils.get_log_format()
    if fmt == "json":
        reporters.append(LogProgressReporter())
    else:
        reporters.append(RichProgressReporter())

    if integration_manager and integration_manager.has_wandb:
        reporters.append(WandBProgressReporter(integration_manager))

    if len(reporters) == 1:
        return reporters[0]

    return CompositeProgressReporter(reporters)


__all__ = [
    "ProgressReporter",
    "RichProgressReporter",
    "LogProgressReporter",
    "WandBProgressReporter",
    "CompositeProgressReporter",
    "get_progress_reporter",
]

"""Progress reporter that logs metrics to WandB."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from themis.utils.progress.base import ProgressReporter

if TYPE_CHECKING:
    from themis.experiment.integration_manager import IntegrationManager


class WandBProgressReporter(ProgressReporter):
    """Progress reporter that logs metrics to WandB."""

    def __init__(self, integration_manager: IntegrationManager) -> None:
        self._integrations = integration_manager
        self._tasks: dict[int, dict[str, Any]] = {}

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def add_task(self, description: str, total: int | None = None) -> int:
        # We don't track task IDs strictly here, just pass through updates
        # But we need to return a dummy ID if we don't have one
        return 0

    def update(self, task_id: int, advance: int = 1, **kwargs: Any) -> None:
        # Log progress metrics
        metrics = {}
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                metrics[f"progress/{k}"] = v

        if metrics:
            self._integrations.log_metrics(metrics)

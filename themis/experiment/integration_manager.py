"""Integration management for external services (WandB, HuggingFace Hub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from themis.config.schema import IntegrationsConfig
from themis.core.entities import ExperimentReport
from themis.integrations.huggingface import HuggingFaceHubUploader
from themis.integrations.wandb import WandbTracker


class IntegrationManager:
    """Manages external integrations (WandB, HuggingFace Hub).

    This class handles all integration-related operations including:
    - Initializing integrations based on configuration
    - Logging experiment results to WandB
    - Uploading results to HuggingFace Hub
    - Finalizing integrations on completion

    Single Responsibility: External integration management
    """

    def __init__(self, config: IntegrationsConfig | None = None) -> None:
        """Initialize integration manager.

        Args:
            config: Integration configuration (None disables all integrations)
        """
        self._config = config or IntegrationsConfig()

        # Initialize WandB tracker if enabled
        self._wandb_tracker = (
            WandbTracker(self._config.wandb) if self._config.wandb.enable else None
        )

        # Initialize HuggingFace Hub uploader if enabled
        self._hf_uploader = (
            HuggingFaceHubUploader(self._config.huggingface_hub)
            if self._config.huggingface_hub.enable
            else None
        )

    @property
    def has_wandb(self) -> bool:
        """Check if WandB integration is enabled."""
        return self._wandb_tracker is not None

    @property
    def has_huggingface(self) -> bool:
        """Check if HuggingFace Hub integration is enabled."""
        return self._hf_uploader is not None

    def initialize_run(self, run_config: dict[str, Any]) -> None:
        """Initialize integrations for a new run.

        Args:
            run_config: Configuration dictionary for the run
                Common keys: max_samples, run_id, resume
        """
        if self._wandb_tracker:
            self._wandb_tracker.init(run_config)

    def log_results(self, report: ExperimentReport) -> None:
        """Log experiment results to integrations.

        Args:
            report: Completed experiment report with all results
        """
        if self._wandb_tracker:
            self._wandb_tracker.log_results(report)

    def upload_results(
        self,
        report: ExperimentReport,
        run_path: str | Path | None,
    ) -> None:
        """Upload results to HuggingFace Hub.

        Args:
            report: Completed experiment report
            run_path: Path to run directory with cached results
        """
        if self._hf_uploader and run_path is not None:
            self._hf_uploader.upload_results(report, run_path)

    def finalize(self) -> None:
        """Finalize all integrations.

        This should be called after experiment completion to properly
        close connections and clean up resources.
        """
        if self._wandb_tracker:
            # WandB tracker handles finalization in log_results
            pass

        if self._hf_uploader:
            # HuggingFace uploader is stateless, no finalization needed
            pass


__all__ = ["IntegrationManager"]

"""Langfuse integration for experiment tracing and score logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langfuse import Langfuse
else:
    try:
        from langfuse import Langfuse
    except ImportError:
        Langfuse = None  # type: ignore

from themis.config.schema import LangfuseConfig
from themis.core.entities import ExperimentReport

logger = logging.getLogger(__name__)


class LangfuseTracker:
    """Tracks experiment runs in Langfuse.

    Provides two layers of integration:
    1. Automatic LLM call tracing via LiteLLM's ``langfuse_otel`` callback.
    2. Evaluation score logging via the Langfuse Python SDK.
    """

    def __init__(self, config: LangfuseConfig) -> None:
        if Langfuse is None:
            raise ImportError(
                "langfuse is not installed. Install with: pip install langfuse"
            )
        self.config = config
        self._client: Langfuse | None = None
        self._trace_id: str | None = None

    def init(self, experiment_config: dict) -> None:
        """Initialize Langfuse client and optionally enable LiteLLM tracing.

        Args:
            experiment_config: Dictionary with run configuration metadata.
        """
        if not self.config.enable:
            return

        # Build kwargs for Langfuse client — only pass non-None values so the
        # SDK falls back to environment variables for anything unset.
        client_kwargs: dict[str, Any] = {}
        if self.config.public_key is not None:
            client_kwargs["public_key"] = self.config.public_key
        if self.config.secret_key is not None:
            client_kwargs["secret_key"] = self.config.secret_key
        if self.config.base_url is not None:
            client_kwargs["base_url"] = self.config.base_url

        self._client = Langfuse(**client_kwargs)

        # Create a root trace for this experiment run
        trace_name = self.config.trace_name or experiment_config.get(
            "run_id", "themis-experiment"
        )
        trace = self._client.trace(
            name=trace_name,
            metadata=experiment_config,
            tags=self.config.tags or [],
        )
        self._trace_id = trace.id

        # Enable automatic LLM call tracing via LiteLLM callback
        if self.config.enable_tracing:
            try:
                import litellm

                if "langfuse" not in litellm.callbacks:
                    litellm.callbacks.append("langfuse")  # type: ignore[arg-type]
                    logger.info("LangfuseTracker: Enabled LiteLLM langfuse callback")
            except Exception:
                logger.warning(
                    "LangfuseTracker: Could not enable LiteLLM langfuse callback",
                    exc_info=True,
                )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to Langfuse as scores on the root trace.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number (unused, kept for API consistency).
        """
        if not self.config.enable or self._client is None or self._trace_id is None:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._client.score(
                    trace_id=self._trace_id,
                    name=name,
                    value=value,
                )

    def log_results(self, report: ExperimentReport) -> None:
        """Log final experiment results to Langfuse.

        Logs aggregate metric scores and summary metadata as scores/spans
        on the root trace.

        Args:
            report: Completed experiment report.
        """
        if not self.config.enable or self._client is None or self._trace_id is None:
            return

        # Log summary metadata
        summary: dict[str, Any] = {
            "total_samples": report.metadata.get("total_samples"),
            "successful_generations": report.metadata.get("successful_generations"),
            "failed_generations": report.metadata.get("failed_generations"),
            "evaluation_failures": report.metadata.get("evaluation_failures"),
        }

        # Log aggregate metric scores
        for name, aggregate in report.evaluation_report.metrics.items():
            score_name = f"{name}_mean"
            summary[score_name] = aggregate.mean
            self._client.score(
                trace_id=self._trace_id,
                name=score_name,
                value=aggregate.mean,
                comment=f"count={aggregate.count}",
            )

        # Update the root trace with final metadata
        self._client.trace(
            id=self._trace_id,
            output=summary,
        )

    def finalize(self) -> None:
        """Flush all pending events to Langfuse."""
        if not self.config.enable or self._client is None:
            return

        # Remove litellm callback to avoid leaking into other runs
        if self.config.enable_tracing:
            try:
                import litellm

                if "langfuse" in litellm.callbacks:
                    litellm.callbacks.remove("langfuse")  # type: ignore[arg-type]
            except Exception:
                pass

        self._client.flush()

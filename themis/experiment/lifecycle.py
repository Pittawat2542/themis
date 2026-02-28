from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from themis.core.entities import ExperimentReport, ExperimentFailure
from themis.evaluation import pipeline as evaluation_pipeline
from themis.experiment.cache_manager import CacheManager
from themis.experiment.integration_manager import IntegrationManager
from themis.experiment.manifest import (
    build_reproducibility_manifest,
    manifest_hash,
    validate_reproducibility_manifest,
)
from themis.experiment.context import _RetentionBuffer
from themis.experiment.pricing import get_pricing_metadata
from themis.generation import plan as generation_plan
from themis.experiment.cost import CostBreakdown

logger = logging.getLogger(__name__)


class RunLifecycle:
    """Manages experiment initialization, manifest creation, and final report assembly."""

    def __init__(
        self,
        plan: generation_plan.GenerationPlan,
        evaluation: evaluation_pipeline.EvaluationPipeline,
        cache: CacheManager,
        integrations: IntegrationManager,
    ) -> None:
        self._plan = plan
        self._evaluation = evaluation
        self._cache = cache
        self._integrations = integrations

    def default_run_manifest(self) -> dict[str, Any]:
        """Build a default reproducibility manifest based on current plan and eval config."""
        model = self._plan.models[0] if self._plan.models else None
        sampling = (
            self._plan.sampling_parameters[0]
            if self._plan.sampling_parameters
            else None
        )
        sampling_config = {
            "temperature": sampling.temperature if sampling else 0.0,
            "top_p": sampling.top_p if sampling else 0.95,
            "max_tokens": sampling.max_tokens if sampling else 512,
        }
        evaluation_config = dict(self._evaluation.evaluation_fingerprint())
        if "metrics" not in evaluation_config:
            evaluation_config["metrics"] = []
        if "extractor" not in evaluation_config:
            evaluation_config["extractor"] = "unknown"

        return build_reproducibility_manifest(
            model=model.identifier if model else "unknown",
            provider=model.provider if model else "unknown",
            provider_options={},
            sampling=sampling_config,
            num_samples=1,
            evaluation_config=evaluation_config,
        )

    def default_run_id(self) -> str:
        """Generate a default run ID based on current timestamp."""
        return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    def initialize_run_resources(
        self,
        *,
        dataset: Sequence[dict[str, object]],
        run_id: str | None,
        max_samples: int | None,
        resume: bool,
        run_manifest: dict[str, object] | None,
    ) -> tuple[list[dict[str, object]], str, dict[str, object], str | None]:
        """Initialize all resources needed for the run."""
        logger.info("RunLifecycle: Loading dataset...")
        dataset_list = list(dataset)
        logger.info(
            "RunLifecycle: Dataset loaded (%s total samples)",
            len(dataset_list),
            extra={"dataset_size": len(dataset_list)},
        )

        selected_dataset = (
            dataset_list[:max_samples] if max_samples is not None else dataset_list
        )
        run_identifier = run_id or self.default_run_id()
        manifest_payload = dict(run_manifest or {})
        run_manifest_hash: str | None = None

        if self._cache.has_storage:
            if not manifest_payload:
                manifest_payload = self.default_run_manifest()
            validate_reproducibility_manifest(manifest_payload)
            run_manifest_hash = manifest_hash(manifest_payload)

        logger.info(
            "RunLifecycle: Processing %s samples",
            len(selected_dataset),
            extra={"samples": len(selected_dataset)},
        )
        logger.info(
            "RunLifecycle: Run ID = %s",
            run_identifier,
            extra={"run_id": run_identifier},
        )

        if self._cache.has_storage:
            if not resume or not self._cache.run_metadata_exists(run_identifier):
                self._cache.start_run(
                    run_identifier,
                    experiment_id="default",
                    config={
                        "reproducibility_manifest": manifest_payload,
                        "manifest_hash": run_manifest_hash,
                    },
                )

        if dataset_list:
            self._cache.cache_dataset(run_identifier, dataset_list)

        return selected_dataset, run_identifier, manifest_payload, run_manifest_hash

    def finalize_experiment_run(
        self,
        *,
        run_identifier: str,
        selected_dataset: Any,
        manifest_payload: dict,
        run_manifest_hash: str | None,
        generation_results: _RetentionBuffer,
        evaluation_report: evaluation_pipeline.EvaluationReport,
        failures: list[ExperimentFailure],
        cached_eval_records: _RetentionBuffer,
        new_eval_records: _RetentionBuffer,
        successful_generations_total: int,
        failed_generations_total: int,
        evaluation_record_failures_total: int,
        cost_breakdown: CostBreakdown,
        cache_results: bool,
    ) -> ExperimentReport:
        """Finalize experiment run, including reporting and cleanup."""
        pricing_metadata = get_pricing_metadata()
        if cost_breakdown.total_cost > 0:
            logger.info(
                "Orchestrator: Total cost: $%s",
                f"{cost_breakdown.total_cost:.4f}",
                extra={"cost": cost_breakdown.total_cost},
            )

        evaluation_truncation = evaluation_report.metadata.get(
            "per_sample_metrics_truncated", False
        )
        metadata = {
            "total_samples": len(selected_dataset),
            "successful_generations": successful_generations_total,
            "failed_generations": failed_generations_total,
            "generation_records_retained": len(generation_results.to_list()),
            "generation_records_dropped": generation_results.dropped,
            "evaluation_records_retained": len(evaluation_report.records),
            "evaluation_records_dropped": (
                cached_eval_records.dropped + new_eval_records.dropped
            ),
            "per_sample_metrics_truncated": bool(evaluation_truncation),
            "run_id": run_identifier,
            "evaluation_failures": (
                evaluation_record_failures_total + len(evaluation_report.failures)
            ),
            "manifest_hash": run_manifest_hash,
            "reproducibility_manifest": manifest_payload,
            "cost": {
                "total_cost": cost_breakdown.total_cost,
                "generation_cost": cost_breakdown.generation_cost,
                "evaluation_cost": cost_breakdown.evaluation_cost,
                "currency": cost_breakdown.currency,
                "token_counts": cost_breakdown.token_counts,
                "api_calls": cost_breakdown.api_calls,
                "per_model_costs": cost_breakdown.per_model_costs,
                "pricing_version": pricing_metadata["version"],
                "pricing_updated_at": pricing_metadata["updated_at"],
            },
        }

        report = ExperimentReport(
            generation_results=generation_results.to_list(),
            evaluation_report=evaluation_report,
            failures=failures,
            metadata=metadata,
        )

        self._integrations.log_results(report)

        run_path = self._cache.get_run_path(run_identifier)
        self._integrations.upload_results(report, run_path)

        if cache_results:
            self._save_report_json(report, run_identifier)

        self._cache.complete_run(run_identifier)

        return report

    def _save_report_json(self, report: ExperimentReport, run_id: str) -> None:
        """Save experiment report as JSON for multi-experiment comparison."""
        from pathlib import Path
        from themis.experiment.export import build_json_report
        import json

        run_path_str = self._cache.get_run_path(run_id)
        if run_path_str is None:
            return

        run_path = Path(run_path_str)
        report_path = run_path / "report.json"

        json_data = build_json_report(report, title=f"Experiment {run_id}")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

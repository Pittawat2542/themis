"""Experiment orchestrator primitives."""

from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from themis.config.schema import IntegrationsConfig
from themis.core.entities import (
    EvaluationRecord,
    ExperimentFailure,
    ExperimentReport,
    GenerationRecord,
    GenerationTask,
    MetricScore,
)
from themis.evaluation import pipeline as evaluation_pipeline
from themis.evaluation.reports import EvaluationFailure
from themis.experiment.manifest import (
    build_reproducibility_manifest,
    manifest_hash,
    validate_reproducibility_manifest,
)
from themis.experiment import storage as experiment_storage
from themis.experiment.cache_manager import CacheManager
from themis.experiment.cost import CostTracker
from themis.experiment.integration_manager import IntegrationManager
from themis.experiment.pricing import calculate_cost, get_pricing_metadata
from themis.generation import plan as generation_plan
from themis.generation import runner as generation_runner

logger = logging.getLogger(__name__)


def _stable_metric_id(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", normalized).lower()


class _RetentionBuffer:
    """Bounded buffer that tracks dropped items."""

    def __init__(self, max_items: int | None = None) -> None:
        self.max_items = max_items
        self.dropped = 0
        if max_items is None:
            self._items: list[Any] | deque[Any] = []
        else:
            self._items = deque()

    def append(self, item: Any) -> None:
        if self.max_items is None:
            assert isinstance(self._items, list)
            self._items.append(item)
            return

        assert isinstance(self._items, deque)
        if len(self._items) >= self.max_items:
            self._items.popleft()
            self.dropped += 1
        self._items.append(item)

    def to_list(self) -> list[Any]:
        return list(self._items)


class ExperimentOrchestrator:
    """Orchestrates experiment execution: generation → evaluation → reporting.

    This class coordinates the experiment workflow using focused managers:
    - CacheManager: Handles storage and resumability
    - IntegrationManager: Handles WandB and HuggingFace Hub

    Single Responsibility: Orchestration of experiment flow
    """

    def __init__(
        self,
        *,
        generation_plan: generation_plan.GenerationPlan,
        generation_runner: generation_runner.GenerationRunner,
        evaluation_pipeline: evaluation_pipeline.EvaluationPipeline,
        cache_manager: CacheManager | None = None,
        integration_manager: IntegrationManager | None = None,
    ) -> None:
        """Initialize experiment orchestrator.

        Args:
            generation_plan: Plan for expanding dataset into tasks
            generation_runner: Runner for executing generation tasks
            evaluation_pipeline: Pipeline for evaluating outputs
            cache_manager: Manager for caching and resumability
            integration_manager: Manager for external integrations
        """
        self._plan = generation_plan
        self._runner = generation_runner
        self._evaluation = evaluation_pipeline

        self._cache = cache_manager or CacheManager(
            storage=None,
            enable_resume=False,
            enable_cache=False,
        )
        self._integrations = integration_manager or IntegrationManager(
            config=IntegrationsConfig()
        )

        # Initialize cost tracker
        self._cost_tracker = CostTracker()

    def run(
        self,
        dataset: Sequence[dict[str, object]] | None = None,
        *,
        dataset_loader: Callable[[], Sequence[dict[str, object]]] | None = None,
        max_samples: int | None = None,
        run_id: str | None = None,
        resume: bool = True,
        cache_results: bool = True,
        on_result: Callable[[GenerationRecord], None] | None = None,
        run_manifest: dict[str, object] | None = None,
        evaluation_batch_size: int = 100,
        max_records_in_memory: int | None = None,
    ) -> ExperimentReport:
        """Run experiment: generate responses, evaluate, and report results.

        Args:
            dataset: Optional dataset samples to use
            dataset_loader: Optional callable to load dataset
            max_samples: Optional limit on number of samples
            run_id: Optional run identifier for caching
            resume: Whether to resume from cached results
            cache_results: Whether to cache new results
            on_result: Optional callback for each generation result
            run_manifest: Required reproducibility manifest for this run
            evaluation_batch_size: Number of records per evaluation batch
            max_records_in_memory: Maximum generation/evaluation records to keep
                in the final report. None keeps all records.

        Returns:
            ExperimentReport with generation results, evaluation, and metadata
        """
        logger.info("Orchestrator: Initializing experiment run")
        if evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be >= 1")
        if max_records_in_memory is not None and max_records_in_memory < 1:
            raise ValueError("max_records_in_memory must be >= 1")

        # Initialize integrations
        self._integrations.initialize_run(
            {
                "max_samples": max_samples,
                "run_id": run_id,
                "resume": resume,
            }
        )

        # Prepare dataset
        logger.info("Orchestrator: Loading dataset...")
        try:
            dataset_list = self._resolve_dataset(
                dataset=dataset, dataset_loader=dataset_loader, run_id=run_id
            )
            logger.info(
                f"Orchestrator: Dataset loaded ({len(dataset_list)} total samples)"
            )
        except Exception as e:
            logger.error(f"Orchestrator: ❌ Failed to load dataset: {e}")
            raise

        selected_dataset = (
            dataset_list[:max_samples] if max_samples is not None else dataset_list
        )
        run_identifier = run_id or self._default_run_id()
        manifest_payload = dict(run_manifest or {})
        run_manifest_hash: str | None = None
        if self._cache.has_storage:
            if not manifest_payload:
                manifest_payload = self._default_run_manifest()
            validate_reproducibility_manifest(manifest_payload)
            run_manifest_hash = manifest_hash(manifest_payload)

        logger.info(f"Orchestrator: Processing {len(selected_dataset)} samples")
        logger.info(f"Orchestrator: Run ID = {run_identifier}")

        # Initialize run in storage (if storage exists and run doesn't exist)
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

        # Cache dataset for resumability
        if dataset_list:
            self._cache.cache_dataset(run_identifier, dataset_list)

        # Expand dataset into generation tasks
        logger.info("Orchestrator: Expanding dataset into generation tasks...")
        try:
            task_iterator = iter(self._plan.expand(selected_dataset))
        except Exception as e:
            self._cache.fail_run(run_identifier, str(e))
            logger.error(f"Orchestrator: ❌ Failed to expand dataset: {e}")
            raise

        # Build evaluation configuration for cache invalidation
        evaluation_config = self._build_evaluation_config()

        # Load cached results if resuming
        if resume:
            logger.info("Orchestrator: Loading cached results...")
        cached_records = (
            self._cache.load_cached_records(run_identifier) if resume else {}
        )
        cached_evaluations = (
            self._cache.load_cached_evaluations(run_identifier, evaluation_config)
            if resume
            else {}
        )
        if resume and cached_records:
            logger.info(
                f"Orchestrator: Found {len(cached_records)} cached generation records"
            )
        if resume and cached_evaluations:
            logger.info(
                f"Orchestrator: Found {len(cached_evaluations)} cached evaluation records"
            )

        # Process tasks: use cached or run new generations
        generation_results = _RetentionBuffer(max_records_in_memory)
        cached_eval_records = _RetentionBuffer(max_records_in_memory)
        new_eval_records = _RetentionBuffer(max_records_in_memory)
        failures: list[ExperimentFailure] = []
        eval_batch: list[GenerationRecord] = []
        new_eval_failures: list[EvaluationFailure] = []
        cached_eval_failures: list[EvaluationFailure] = []
        expected_metric_names: set[str] = set()
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        metric_samples: dict[str, _RetentionBuffer] = {}
        successful_generations_total = 0
        failed_generations_total = 0
        evaluation_record_failures_total = 0
        discovered_tasks_total = 0
        pending_tasks_total = 0

        if hasattr(self._evaluation, "_metrics"):
            expected_metric_names.update(
                metric.name for metric in self._evaluation._metrics
            )

        def _accumulate_evaluation_record(record: EvaluationRecord) -> None:
            nonlocal evaluation_record_failures_total
            evaluation_record_failures_total += len(record.failures)
            for score in record.scores:
                metric_name = score.metric_name
                metric_sums[metric_name] = (
                    metric_sums.get(metric_name, 0.0) + score.value
                )
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
                buffer = metric_samples.setdefault(
                    metric_name, _RetentionBuffer(max_records_in_memory)
                )
                buffer.append(score)

        def _flush_eval_batch() -> None:
            nonlocal eval_batch
            if not eval_batch:
                return
            logger.info(
                "Orchestrator: Evaluating batch of %s records...",
                len(eval_batch),
            )
            batch_report = self._evaluation.evaluate(eval_batch)
            expected_metric_names.update(batch_report.metrics.keys())
            for record, evaluation in zip(eval_batch, batch_report.records):
                if cache_results:
                    self._cache.save_evaluation_record(
                        run_identifier, record, evaluation, evaluation_config
                    )
                _accumulate_evaluation_record(evaluation)
                new_eval_records.append(evaluation)
            new_eval_failures.extend(batch_report.failures)
            eval_batch = []

        def _iter_pending_tasks() -> Iterable[GenerationTask]:
            nonlocal discovered_tasks_total
            nonlocal pending_tasks_total
            nonlocal successful_generations_total
            nonlocal failed_generations_total
            for task in task_iterator:
                discovered_tasks_total += 1
                if run_manifest_hash is not None:
                    task.metadata["manifest_hash"] = run_manifest_hash
                task_cache_key = experiment_storage.task_cache_key(task)
                cached = cached_records.get(task_cache_key)
                if cached is not None:
                    generation_results.append(cached)
                    if cached.error:
                        failed_generations_total += 1
                    else:
                        successful_generations_total += 1
                    if cached.error:
                        failures.append(
                            ExperimentFailure(
                                sample_id=cached.task.metadata.get("dataset_id"),
                                message=cached.error.message,
                            )
                        )
                    eval_cache_key = experiment_storage.evaluation_cache_key(
                        task, evaluation_config
                    )
                    evaluation = cached_evaluations.get(eval_cache_key)
                    if evaluation is not None:
                        cached_eval_records.append(evaluation)
                        _accumulate_evaluation_record(evaluation)
                        for message in evaluation.failures:
                            cached_eval_failures.append(
                                EvaluationFailure(
                                    sample_id=evaluation.sample_id, message=message
                                )
                            )
                    else:
                        eval_batch.append(cached)
                        if len(eval_batch) >= evaluation_batch_size:
                            _flush_eval_batch()
                    if on_result:
                        on_result(cached)
                    continue

                pending_tasks_total += 1
                yield task

        # Run pending generation tasks without pre-materializing task lists.
        completed = 0
        try:
            for record in self._runner.run(_iter_pending_tasks()):
                logger.debug("Orchestrator: Received generation record")
                generation_results.append(record)
                if record.error:
                    failed_generations_total += 1
                else:
                    successful_generations_total += 1
                completed += 1

                # Log progress every 10 samples or at key milestones
                if pending_tasks_total and (
                    completed % 10 == 0 or completed == pending_tasks_total
                ):
                    logger.info(
                        "Orchestrator: Generation progress: %s/%s (%s%%)",
                        completed,
                        pending_tasks_total,
                        (100 * completed // pending_tasks_total),
                    )

                logger.debug("Orchestrator: Processing record (cost tracking...)")
                # Track cost for successful generations
                if record.output and record.output.usage:
                    usage = record.output.usage
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    model = record.task.model.identifier

                    # Calculate cost using pricing database
                    cost = calculate_cost(model, prompt_tokens, completion_tokens)
                    self._cost_tracker.record_generation(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost=cost,
                    )

                logger.debug("Orchestrator: Processing record (error handling...)")
                if record.error:
                    failures.append(
                        ExperimentFailure(
                            sample_id=record.task.metadata.get("dataset_id"),
                            message=record.error.message,
                        )
                    )

                logger.debug("Orchestrator: Processing record (caching...)")
                cache_key = experiment_storage.task_cache_key(record.task)
                if cache_results:
                    self._cache.save_generation_record(
                        run_identifier, record, cache_key
                    )

                logger.debug("Orchestrator: Processing record (queueing evaluation...)")
                eval_batch.append(record)
                if len(eval_batch) >= evaluation_batch_size:
                    _flush_eval_batch()

                logger.debug("Orchestrator: Processing record (callback...)")
                if on_result:
                    on_result(record)
                logger.debug("Orchestrator: Record processing complete")
        except Exception as exc:
            self._cache.fail_run(run_identifier, str(exc))
            raise

        logger.info(
            "Orchestrator: Task scan complete (%s total, %s pending, %s cached)",
            discovered_tasks_total,
            pending_tasks_total,
            discovered_tasks_total - pending_tasks_total,
        )

        # Evaluate remaining queued records
        logger.info(
            "Orchestrator: Flushing final evaluation batch (%s queued)...",
            len(eval_batch),
        )
        try:
            _flush_eval_batch()
        except Exception as exc:
            self._cache.fail_run(run_identifier, str(exc))
            raise

        # Combine cached and new evaluations
        logger.info("Orchestrator: Combining cached and new evaluations...")
        try:
            evaluation_report = self._combine_evaluations(
                cached_eval_records.to_list(),
                new_eval_records.to_list(),
                new_eval_failures,
                expected_metric_names=expected_metric_names,
                cached_failures=cached_eval_failures,
                metric_sums=metric_sums,
                metric_counts=metric_counts,
                metric_samples={
                    name: buffer.to_list() for name, buffer in metric_samples.items()
                },
                max_records_in_memory=max_records_in_memory,
            )
        except Exception as exc:
            self._cache.fail_run(run_identifier, str(exc))
            raise
        logger.info(
            f"Orchestrator: Total evaluation records: {len(evaluation_report.records)}"
        )

        # Get cost breakdown
        cost_breakdown = self._cost_tracker.get_breakdown()
        pricing_metadata = get_pricing_metadata()
        if cost_breakdown.total_cost > 0:
            logger.info(f"Orchestrator: Total cost: ${cost_breakdown.total_cost:.4f}")

        # Build metadata
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
            "evaluation_records_dropped": cached_eval_records.dropped
            + new_eval_records.dropped,
            "per_sample_metrics_truncated": bool(evaluation_truncation),
            "run_id": run_identifier,
            "evaluation_failures": evaluation_record_failures_total
            + len(evaluation_report.failures),
            "manifest_hash": run_manifest_hash,
            "reproducibility_manifest": manifest_payload,
            # Cost tracking
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

        # Create final report
        report = ExperimentReport(
            generation_results=generation_results.to_list(),
            evaluation_report=evaluation_report,
            failures=failures,
            metadata=metadata,
        )

        # Log to integrations
        self._integrations.log_results(report)

        # Upload to HuggingFace Hub if enabled
        run_path = self._cache.get_run_path(run_identifier)
        self._integrations.upload_results(report, run_path)

        # Save report.json for multi-experiment comparison
        if cache_results:
            self._save_report_json(report, run_identifier)

        self._cache.complete_run(run_identifier)

        return report

    def _default_run_manifest(self) -> dict[str, object]:
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
        evaluation_config = self._build_evaluation_config()
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

    def _default_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    def _build_evaluation_config(self) -> dict:
        """Build evaluation configuration for cache key generation.

        This configuration includes all evaluation settings that affect results,
        so changing metrics or extractors will invalidate the cache.

        Returns:
            Dictionary with evaluation configuration
        """
        if hasattr(self._evaluation, "evaluation_fingerprint"):
            try:
                return dict(self._evaluation.evaluation_fingerprint())
            except Exception:
                pass

        config = {}

        # Add metric names/types
        if hasattr(self._evaluation, "_metrics"):
            config["metrics"] = sorted(
                [
                    f"{metric.__class__.__module__}.{metric.__class__.__name__}:{metric.name}"
                    for metric in self._evaluation._metrics
                ]
            )

        # Add extractor type
        if hasattr(self._evaluation, "_extractor"):
            extractor = self._evaluation._extractor
            extractor_type = (
                f"{extractor.__class__.__module__}.{extractor.__class__.__name__}"
            )
            config["extractor"] = extractor_type

            # Include extractor-specific configuration if available
            if hasattr(extractor, "field_name"):
                config["extractor_field"] = extractor.field_name

        return config

    def _resolve_dataset(
        self,
        *,
        dataset: Sequence[dict[str, object]] | None,
        dataset_loader: Callable[[], Sequence[dict[str, object]]] | None,
        run_id: str | None,
    ) -> list[dict[str, object]]:
        """Resolve dataset from various sources.

        Args:
            dataset: Direct dataset samples
            dataset_loader: Callable to load dataset
            run_id: Run ID to load cached dataset

        Returns:
            List of dataset samples

        Raises:
            ValueError: If no dataset source is available
        """
        if dataset is not None:
            return list(dataset)
        if dataset_loader is not None:
            return list(dataset_loader())
        if run_id is not None:
            cached_dataset = self._cache.load_cached_dataset(run_id)
            if cached_dataset is not None:
                return cached_dataset
        raise ValueError(
            "No dataset provided. Supply `dataset=` rows, a `dataset_loader`, "
            "or set `run_id` with storage configured so cached data can be reloaded."
        )

    def _combine_evaluations(
        self,
        cached_records: list[EvaluationRecord],
        new_records: list[EvaluationRecord],
        new_failures: list[EvaluationFailure],
        *,
        expected_metric_names: set[str] | None = None,
        cached_failures: list[EvaluationFailure] | None = None,
        metric_sums: dict[str, float] | None = None,
        metric_counts: dict[str, int] | None = None,
        metric_samples: dict[str, list[MetricScore]] | None = None,
        max_records_in_memory: int | None = None,
    ) -> evaluation_pipeline.EvaluationReport:
        record_buffer = _RetentionBuffer(max_records_in_memory)
        for record in cached_records:
            record_buffer.append(record)
        for record in new_records:
            record_buffer.append(record)
        all_records = record_buffer.to_list()

        if metric_sums is None or metric_counts is None or metric_samples is None:
            per_metric: dict[str, list[MetricScore]] = {}
            computed_sums: dict[str, float] = {}
            computed_counts: dict[str, int] = {}
            for record in all_records:
                for score in record.scores:
                    per_metric.setdefault(score.metric_name, []).append(score)
                    computed_sums[score.metric_name] = (
                        computed_sums.get(score.metric_name, 0.0) + score.value
                    )
                    computed_counts[score.metric_name] = (
                        computed_counts.get(score.metric_name, 0) + 1
                    )
            metric_sums = computed_sums
            metric_counts = computed_counts
            metric_samples = per_metric

        aggregates: dict[str, evaluation_pipeline.MetricAggregate] = {}
        metric_names = set(metric_counts.keys()) | set(expected_metric_names or set())
        metrics_truncated = False
        per_metric_truncation: dict[str, dict[str, int | bool]] = {}
        for name in metric_names:
            scores = metric_samples.get(name, [])
            count = metric_counts.get(name, 0)
            mean = (metric_sums.get(name, 0.0) / count) if count else 0.0
            truncated_count = max(0, count - len(scores))
            per_sample_complete = truncated_count == 0
            metrics_truncated = metrics_truncated or (not per_sample_complete)
            per_metric_truncation[name] = {
                "per_sample_complete": per_sample_complete,
                "truncated_count": truncated_count,
            }
            aggregates[name] = evaluation_pipeline.MetricAggregate(
                name=name,
                count=count,
                mean=mean,
                per_sample=list(scores),
                per_sample_complete=per_sample_complete,
                truncated_count=truncated_count,
            )

        failures = list(new_failures)
        if cached_failures is not None:
            failures.extend(cached_failures)
        else:
            for record in cached_records:
                for message in record.failures:
                    failures.append(
                        EvaluationFailure(sample_id=record.sample_id, message=message)
                    )

        return evaluation_pipeline.EvaluationReport(
            metrics=aggregates,
            failures=failures,
            records=all_records,
            metadata={
                "max_records_in_memory": max_records_in_memory,
                "records_retained": len(all_records),
                "records_dropped": record_buffer.dropped,
                "per_sample_metrics_truncated": metrics_truncated,
                "per_metric": per_metric_truncation,
                "metric_ids": {
                    metric_name: _stable_metric_id(metric_name)
                    for metric_name in metric_names
                },
            },
        )

    def _save_report_json(self, report: ExperimentReport, run_id: str) -> None:
        """Save experiment report as JSON for multi-experiment comparison.

        Args:
            report: Experiment report to save
            run_id: Run identifier
        """
        from pathlib import Path

        from themis.experiment.export import build_json_report

        # Get run path from cache manager
        run_path_str = self._cache.get_run_path(run_id)
        if run_path_str is None:
            # No storage configured, skip saving report.json
            return

        run_path = Path(run_path_str)
        report_path = run_path / "report.json"

        # Build JSON report
        json_data = build_json_report(report, title=f"Experiment {run_id}")

        # Save to file
        import json

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

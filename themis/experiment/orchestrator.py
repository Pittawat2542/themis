"""Experiment orchestrator primitives."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Sequence

from themis.config.schema import IntegrationsConfig

logger = logging.getLogger(__name__)
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
from themis.experiment import storage as experiment_storage
from themis.experiment.cache_manager import CacheManager
from themis.experiment.cost import CostTracker
from themis.experiment.integration_manager import IntegrationManager
from themis.experiment.pricing import calculate_cost, get_provider_pricing
from themis.generation import plan as generation_plan
from themis.generation import runner as generation_runner


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
        storage: experiment_storage.ExperimentStorage | None = None,
        integrations_config: IntegrationsConfig | None = None,
        cache_manager: CacheManager | None = None,
        integration_manager: IntegrationManager | None = None,
    ) -> None:
        """Initialize experiment orchestrator.

        Args:
            generation_plan: Plan for expanding dataset into tasks
            generation_runner: Runner for executing generation tasks
            evaluation_pipeline: Pipeline for evaluating outputs
            storage: Optional storage backend (deprecated, use cache_manager)
            integrations_config: Integration config (deprecated, use integration_manager)
            cache_manager: Manager for caching and resumability
            integration_manager: Manager for external integrations
        """
        self._plan = generation_plan
        self._runner = generation_runner
        self._evaluation = evaluation_pipeline

        # Support both new managers and legacy direct parameters for backward compatibility
        self._cache = cache_manager or CacheManager(
            storage=storage,
            enable_resume=True,
            enable_cache=True,
        )
        self._integrations = integration_manager or IntegrationManager(
            config=integrations_config or IntegrationsConfig()
        )

        # Initialize cost tracker
        self._cost_tracker = CostTracker()

        # Keep legacy references for backward compatibility
        self._storage = storage

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

        Returns:
            ExperimentReport with generation results, evaluation, and metadata
        """
        logger.info("Orchestrator: Initializing experiment run")
        
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
            logger.info(f"Orchestrator: Dataset loaded ({len(dataset_list)} total samples)")
        except Exception as e:
            logger.error(f"Orchestrator: ❌ Failed to load dataset: {e}")
            raise
        
        selected_dataset = (
            dataset_list[:max_samples] if max_samples is not None else dataset_list
        )
        run_identifier = run_id or self._default_run_id()
        
        logger.info(f"Orchestrator: Processing {len(selected_dataset)} samples")
        logger.info(f"Orchestrator: Run ID = {run_identifier}")

        # Initialize run in storage (if storage exists and run doesn't exist)
        if self._cache.has_storage:
            if not resume or not self._cache._storage._run_metadata_exists(run_identifier):
                self._cache._storage.start_run(run_identifier, experiment_id="default")

        # Cache dataset for resumability
        if dataset_list:
            self._cache.cache_dataset(run_identifier, dataset_list)

        # Expand dataset into generation tasks
        logger.info("Orchestrator: Expanding dataset into generation tasks...")
        try:
            tasks = list(self._plan.expand(selected_dataset))
            logger.info(f"Orchestrator: Created {len(tasks)} generation tasks")
        except Exception as e:
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
            self._cache.load_cached_evaluations(run_identifier, evaluation_config) if resume else {}
        )
        if resume and cached_records:
            logger.info(f"Orchestrator: Found {len(cached_records)} cached generation records")
        if resume and cached_evaluations:
            logger.info(f"Orchestrator: Found {len(cached_evaluations)} cached evaluation records")

        # Process tasks: use cached or run new generations
        generation_results: list[GenerationRecord] = []
        failures: list[ExperimentFailure] = []
        pending_tasks: list[GenerationTask] = []
        pending_records: list[GenerationRecord] = []
        pending_keys: list[str] = []
        cached_eval_records: list[EvaluationRecord] = []

        for task in tasks:
            task_cache_key = experiment_storage.task_cache_key(task)
            cached = cached_records.get(task_cache_key)
            if cached is not None:
                generation_results.append(cached)
                if cached.error:
                    failures.append(
                        ExperimentFailure(
                            sample_id=cached.task.metadata.get("dataset_id"),
                            message=cached.error.message,
                        )
                    )
                # Use evaluation_cache_key that includes evaluation config
                eval_cache_key = experiment_storage.evaluation_cache_key(task, evaluation_config)
                evaluation = cached_evaluations.get(eval_cache_key)
                if evaluation is not None:
                    cached_eval_records.append(evaluation)
                else:
                    pending_records.append(cached)
                    pending_keys.append(eval_cache_key)
                if on_result:
                    on_result(cached)
            else:
                pending_tasks.append(task)

        # Run pending generation tasks
        if pending_tasks:
            logger.info(f"Orchestrator: Running {len(pending_tasks)} generation tasks...")
            completed = 0
            for record in self._runner.run(pending_tasks):
                logger.debug(f"Orchestrator: Received generation record")
                generation_results.append(record)
                completed += 1
                
                # Log progress every 10 samples or at key milestones
                if completed % 10 == 0 or completed == len(pending_tasks):
                    logger.info(f"Orchestrator: Generation progress: {completed}/{len(pending_tasks)} ({100*completed//len(pending_tasks)}%)")

                logger.debug(f"Orchestrator: Processing record (cost tracking...)")
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

                logger.debug(f"Orchestrator: Processing record (error handling...)")
                if record.error:
                    failures.append(
                        ExperimentFailure(
                            sample_id=record.task.metadata.get("dataset_id"),
                            message=record.error.message,
                        )
                    )
                    
                logger.debug(f"Orchestrator: Processing record (caching...)")
                cache_key = experiment_storage.task_cache_key(record.task)
                if cache_results:
                    self._cache.save_generation_record(
                        run_identifier, record, cache_key
                    )
                    
                logger.debug(f"Orchestrator: Processing record (adding to pending...)")
                pending_records.append(record)
                pending_keys.append(cache_key)
                
                logger.debug(f"Orchestrator: Processing record (callback...)")
                if on_result:
                    on_result(record)
                logger.debug(f"Orchestrator: Record processing complete")

        # Evaluate pending records
        logger.info(f"Orchestrator: Preparing to evaluate {len(pending_records)} pending records...")
        if pending_records:
            logger.info(f"Orchestrator: Starting evaluation of {len(pending_records)} records...")
            try:
                new_evaluation_report = self._evaluation.evaluate(pending_records)
                logger.info(f"Orchestrator: ✅ Evaluation complete - got {len(new_evaluation_report.records)} results")
            except Exception as e:
                logger.error(f"Orchestrator: ❌ Evaluation failed: {e}")
                raise
        else:
            logger.info("Orchestrator: No new records to evaluate (all cached)")
            new_evaluation_report = evaluation_pipeline.EvaluationReport(
                metrics={}, failures=[], records=[]
            )

        # Cache evaluation results
        for record, evaluation in zip(pending_records, new_evaluation_report.records):
            self._cache.save_evaluation_record(
                run_identifier, record, evaluation, evaluation_config
            )

        # Combine cached and new evaluations
        logger.info("Orchestrator: Combining cached and new evaluations...")
        evaluation_report = self._combine_evaluations(
            cached_eval_records, new_evaluation_report
        )
        logger.info(f"Orchestrator: Total evaluation records: {len(evaluation_report.records)}")

        # Get cost breakdown
        cost_breakdown = self._cost_tracker.get_breakdown()
        if cost_breakdown.total_cost > 0:
            logger.info(f"Orchestrator: Total cost: ${cost_breakdown.total_cost:.4f}")

        # Build metadata
        metadata = {
            "total_samples": len(selected_dataset),
            "successful_generations": sum(
                1 for result in generation_results if not result.error
            ),
            "failed_generations": sum(
                1 for result in generation_results if result.error
            ),
            "run_id": run_identifier,
            "evaluation_failures": sum(
                1 for record in evaluation_report.records if record.failures
            )
            + len(evaluation_report.failures),
            # Cost tracking
            "cost": {
                "total_cost": cost_breakdown.total_cost,
                "generation_cost": cost_breakdown.generation_cost,
                "evaluation_cost": cost_breakdown.evaluation_cost,
                "currency": cost_breakdown.currency,
                "token_counts": cost_breakdown.token_counts,
                "api_calls": cost_breakdown.api_calls,
                "per_model_costs": cost_breakdown.per_model_costs,
            },
        }

        # Create final report
        report = ExperimentReport(
            generation_results=generation_results,
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

        return report

    def _default_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

    def _build_evaluation_config(self) -> dict:
        """Build evaluation configuration for cache key generation.
        
        This configuration includes all evaluation settings that affect results,
        so changing metrics or extractors will invalidate the cache.
        
        Returns:
            Dictionary with evaluation configuration
        """
        config = {}
        
        # Add metric names/types
        if hasattr(self._evaluation, "_metrics"):
            config["metrics"] = sorted([
                f"{metric.__class__.__module__}.{metric.__class__.__name__}:{metric.name}"
                for metric in self._evaluation._metrics
            ])
        
        # Add extractor type
        if hasattr(self._evaluation, "_extractor"):
            extractor = self._evaluation._extractor
            extractor_type = f"{extractor.__class__.__module__}.{extractor.__class__.__name__}"
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
        # Try to load from cache (for backward compatibility, still use _storage directly)
        if self._storage is not None and run_id is not None:
            return self._storage.load_dataset(run_id)
        raise ValueError(
            "No dataset provided. Supply `dataset=` rows, a `dataset_loader`, "
            "or set `run_id` with storage configured so cached data can be reloaded."
        )

    def _combine_evaluations(
        self,
        cached_records: list[EvaluationRecord],
        new_report: evaluation_pipeline.EvaluationReport,
    ) -> evaluation_pipeline.EvaluationReport:
        all_records = list(cached_records) + list(new_report.records)
        per_metric: dict[str, list[MetricScore]] = {}
        for record in all_records:
            for score in record.scores:
                per_metric.setdefault(score.metric_name, []).append(score)

        aggregates: dict[str, evaluation_pipeline.MetricAggregate] = {}
        metric_names = set(per_metric.keys()) | set(new_report.metrics.keys())
        for name in metric_names:
            scores = per_metric.get(name, [])
            mean = sum(score.value for score in scores) / len(scores) if scores else 0.0
            aggregates[name] = evaluation_pipeline.MetricAggregate(
                name=name,
                count=len(scores),
                mean=mean,
                per_sample=scores,
            )

        failures = list(new_report.failures)
        for record in cached_records:
            for message in record.failures:
                failures.append(
                    EvaluationFailure(sample_id=record.sample_id, message=message)
                )

        return evaluation_pipeline.EvaluationReport(
            metrics=aggregates,
            failures=failures,
            records=all_records,
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

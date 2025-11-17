"""Experiment orchestrator primitives."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Sequence

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
from themis.experiment import storage as experiment_storage
from themis.experiment.cache_manager import CacheManager
from themis.experiment.integration_manager import IntegrationManager
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
        # Initialize integrations
        self._integrations.initialize_run(
            {
                "max_samples": max_samples,
                "run_id": run_id,
                "resume": resume,
            }
        )

        # Prepare dataset
        dataset_list = self._resolve_dataset(
            dataset=dataset, dataset_loader=dataset_loader, run_id=run_id
        )
        selected_dataset = (
            dataset_list[:max_samples] if max_samples is not None else dataset_list
        )
        run_identifier = run_id or self._default_run_id()

        # Cache dataset for resumability
        if dataset_list:
            self._cache.cache_dataset(run_identifier, dataset_list)

        # Expand dataset into generation tasks
        tasks = list(self._plan.expand(selected_dataset))

        # Load cached results if resuming
        cached_records = (
            self._cache.load_cached_records(run_identifier) if resume else {}
        )
        cached_evaluations = (
            self._cache.load_cached_evaluations(run_identifier) if resume else {}
        )

        # Process tasks: use cached or run new generations
        generation_results: list[GenerationRecord] = []
        failures: list[ExperimentFailure] = []
        pending_tasks: list[GenerationTask] = []
        pending_records: list[GenerationRecord] = []
        pending_keys: list[str] = []
        cached_eval_records: list[EvaluationRecord] = []

        for task in tasks:
            cache_key = experiment_storage.task_cache_key(task)
            cached = cached_records.get(cache_key)
            if cached is not None:
                generation_results.append(cached)
                if cached.error:
                    failures.append(
                        ExperimentFailure(
                            sample_id=cached.task.metadata.get("dataset_id"),
                            message=cached.error.message,
                        )
                    )
                evaluation = cached_evaluations.get(cache_key)
                if evaluation is not None:
                    cached_eval_records.append(evaluation)
                else:
                    pending_records.append(cached)
                    pending_keys.append(cache_key)
                if on_result:
                    on_result(cached)
            else:
                pending_tasks.append(task)

        # Run pending generation tasks
        if pending_tasks:
            for record in self._runner.run(pending_tasks):
                generation_results.append(record)
                if record.error:
                    failures.append(
                        ExperimentFailure(
                            sample_id=record.task.metadata.get("dataset_id"),
                            message=record.error.message,
                        )
                    )
                cache_key = experiment_storage.task_cache_key(record.task)
                if cache_results:
                    self._cache.save_generation_record(
                        run_identifier, record, cache_key
                    )
                pending_records.append(record)
                pending_keys.append(cache_key)
                if on_result:
                    on_result(record)

        # Evaluate pending records
        if pending_records:
            new_evaluation_report = self._evaluation.evaluate(pending_records)
        else:
            new_evaluation_report = evaluation_pipeline.EvaluationReport(
                metrics={}, failures=[], records=[]
            )

        # Cache evaluation results
        for record, evaluation in zip(pending_records, new_evaluation_report.records):
            self._cache.save_evaluation_record(run_identifier, record, evaluation)

        # Combine cached and new evaluations
        evaluation_report = self._combine_evaluations(
            cached_eval_records, new_evaluation_report
        )

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

        return report

    def _default_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")

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
        for name, scores in per_metric.items():
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

"""Experiment orchestrator primitives."""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from typing import Any

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
from themis import storage as experiment_storage
from themis.exceptions import ConfigurationError
from themis.experiment.cache_manager import CacheManager
from themis.experiment.cost import CostTracker
from themis.experiment.integration_manager import IntegrationManager
from themis.experiment.pricing import calculate_cost, get_pricing_metadata
from themis.generation import plan as generation_plan
from themis.generation import runner as generation_runner
from themis.utils.progress import get_progress_reporter

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


class _ExperimentContext:
    """Encapsulates state for a single experiment run."""

    def __init__(self, max_records_in_memory: int | None) -> None:
        self.generation_results = _RetentionBuffer(max_records_in_memory)
        self.cached_eval_records = _RetentionBuffer(max_records_in_memory)
        self.new_eval_records = _RetentionBuffer(max_records_in_memory)
        self.failures: list[ExperimentFailure] = []
        self.eval_batch: list[GenerationRecord] = []
        self.new_eval_failures: list[EvaluationFailure] = []
        self.cached_eval_failures: list[EvaluationFailure] = []
        self.expected_metric_names: set[str] = set()
        self.metric_sums: dict[str, float] = {}
        self.metric_counts: dict[str, int] = {}
        self.metric_samples: dict[str, _RetentionBuffer] = {}

        # Counters
        self.successful_generations_total = 0
        self.failed_generations_total = 0
        self.evaluation_record_failures_total = 0
        self.discovered_tasks_total = 0
        self.pending_tasks_total = 0


class ExperimentOrchestrator:
    """Orchestrates experiment execution: generation → evaluation → reporting.

    The Orchestrator is the engine powering `themis.evaluate()` and the `ExperimentSession` API.
    It coordinates the entire lifecycle of an experiment run, expanding datasets into tasks,
    piping them through GenerationRunners, feeding them into EvaluationPipelines,
    and managing caching so interrupted runs can be resumed.

    Example:
        ```python
        import themis
        from themis.experiment.orchestrator import ExperimentOrchestrator
        from themis.generation.plan import GenerationPlan
        # ... assuming plan, runner, pipeline, and cache are instantiated

        orchestrator = ExperimentOrchestrator(
            generation_plan=plan,
            generation_runner=runner,
            evaluation_pipeline=pipeline,
            cache_manager=cache,
        )

        report = orchestrator.run(
            dataset=[{"question": "2+2", "expected": "4"}],
            run_id="my-experiment-run",
            resume=True
        )
        ```
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
        """Initialize the experiment orchestrator.

        Args:
            generation_plan: The plan dictating how dataset rows expand into concrete tasks (prompts/models).
            generation_runner: The async/threaded engine that physically executes LLM requests.
            evaluation_pipeline: The pipeline containing metrics to score outputs against references.
            cache_manager: Manager for writing/reading `.cache/experiments/{run_id}` to allow resumability.
                If None, memory-only execution is used.
            integration_manager: Manager that pushes real-time results to Weights & Biases or HF Hub.
                If None, a default no-op manager is used.
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
        dataset: Sequence[dict[str, object]],
        *,
        max_samples: int | None = None,
        run_id: str | None = None,
        resume: bool = True,
        cache_results: bool = True,
        on_result: Callable[[GenerationRecord], None] | None = None,
        run_manifest: dict[str, object] | None = None,
        evaluation_batch_size: int = 100,
        max_records_in_memory: int | None = None,
    ) -> ExperimentReport:
        """Execute the experiment run pipeline: fetch, generate, evaluate, and assemble report.

        This method expands the dataset into tasks, checks the cache for previously
        completed work (if `resume=True`), queues up missing work to the generation runner,
        batches the generations into the evaluation pipeline, and produces a final report.

        Args:
            dataset: A resolved list of dataset samples (dictionaries) to evaluate.
            max_samples: Optional limit on the number of samples to process. Used for quick tests.
            run_id: A unique identifier used to look up cached results. If None, a timestamp ID is used.
            resume: Whether to skip generating/evaluating samples that were already completed
                in a previous run with this same `run_id`.
            cache_results: Whether to save freshly generated results to disk for future resumption.
            on_result: An optional callback function triggered immediately after a generation
                record is evaluated. Useful for updating real-time UIs.
            run_manifest: A reproducibility dictionary mapping model configurations and prompt hashes.
                Used to ensure runs with the exact same variables can be reproduced precisely.
            evaluation_batch_size: How many records to evaluate before flushing stats. Default: 100.
            max_records_in_memory: Memory optimization. If set, limits the number of historical
                generation/evaluation records saved in RAM (and thus in the returned report). `None` keeps all.

        Returns:
            An `ExperimentReport` containing standard aggregated metrics, failure logs, and individual records.

        Raises:
            ConfigurationError: If `evaluation_batch_size` or `max_records_in_memory` are < 1.
        """
        logger.info("Orchestrator: Initializing experiment run")
        if evaluation_batch_size < 1:
            raise ConfigurationError("evaluation_batch_size must be >= 1")
        if max_records_in_memory is not None and max_records_in_memory < 1:
            raise ConfigurationError("max_records_in_memory must be >= 1")

        # Initialize integrations
        self._integrations.initialize_run(
            {
                "max_samples": max_samples,
                "run_id": run_id,
                "resume": resume,
            }
        )

        # Initialize run resources (dataset, manifest, storage)
        (
            selected_dataset,
            run_identifier,
            manifest_payload,
            run_manifest_hash,
        ) = self._initialize_run_resources(
            dataset=dataset,
            run_id=run_id,
            max_samples=max_samples,
            resume=resume,
            run_manifest=run_manifest,
        )

        # Expand dataset into generation tasks
        logger.info("Orchestrator: Expanding dataset into generation tasks...")
        try:
            task_iterator = iter(self._plan.expand(selected_dataset))
        except Exception as e:
            self._cache.fail_run(run_identifier, str(e))
            logger.error("Orchestrator: Failed to expand dataset", exc_info=True)
            raise

        # Build evaluation configuration and load cache
        evaluation_config = self._build_evaluation_config()
        cached_records, cached_evaluations = self._load_cache_state(
            run_identifier, evaluation_config, resume
        )

        # Process tasks: use cached or run new generations
        ctx = _ExperimentContext(max_records_in_memory)

        if hasattr(self._evaluation, "_metrics"):
            ctx.expected_metric_names.update(
                metric.name for metric in self._evaluation._metrics
            )

        # Execute generation loop
        self._execute_run_loop(
            ctx=ctx,
            task_iterator=task_iterator,
            run_identifier=run_identifier,
            run_manifest_hash=run_manifest_hash,
            evaluation_config=evaluation_config,
            cached_records=cached_records,
            cached_evaluations=cached_evaluations,
            evaluation_batch_size=evaluation_batch_size,
            on_result=on_result,
            cache_results=cache_results,
        )

        # Combine cached and new evaluations
        logger.info("Orchestrator: Combining cached and new evaluations...")
        try:
            evaluation_report = self._combine_evaluations(
                ctx.cached_eval_records.to_list(),
                ctx.new_eval_records.to_list(),
                ctx.new_eval_failures,
                expected_metric_names=ctx.expected_metric_names,
                cached_failures=ctx.cached_eval_failures,
                metric_sums=ctx.metric_sums,
                metric_counts=ctx.metric_counts,
                metric_samples={
                    name: buffer.to_list()
                    for name, buffer in ctx.metric_samples.items()
                },
                max_records_in_memory=max_records_in_memory,
            )
        except Exception as exc:
            self._cache.fail_run(run_identifier, str(exc))
            raise
        logger.info(
            "Orchestrator: Total evaluation records: %s",
            len(evaluation_report.records),
            extra={"total_records": len(evaluation_report.records)},
        )

        return self._finalize_experiment_run(
            run_identifier=run_identifier,
            selected_dataset=selected_dataset,
            manifest_payload=manifest_payload,
            run_manifest_hash=run_manifest_hash,
            generation_results=ctx.generation_results,
            evaluation_report=evaluation_report,
            failures=ctx.failures,
            cached_eval_records=ctx.cached_eval_records,
            new_eval_records=ctx.new_eval_records,
            successful_generations_total=ctx.successful_generations_total,
            failed_generations_total=ctx.failed_generations_total,
            evaluation_record_failures_total=ctx.evaluation_record_failures_total,
            cache_results=cache_results,
        )

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

    def _initialize_run_resources(
        self,
        *,
        dataset: Sequence[dict[str, object]],
        run_id: str | None,
        max_samples: int | None,
        resume: bool,
        run_manifest: dict[str, object] | None,
    ) -> tuple[list[dict[str, object]], str, dict[str, object], str | None]:
        """Initialize all resources needed for the run."""
        # Prepare dataset
        logger.info("Orchestrator: Loading dataset...")
        dataset_list = list(dataset)
        logger.info(
            "Orchestrator: Dataset loaded (%s total samples)",
            len(dataset_list),
            extra={"dataset_size": len(dataset_list)},
        )

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

        logger.info(
            "Orchestrator: Processing %s samples",
            len(selected_dataset),
            extra={"samples": len(selected_dataset)},
        )
        logger.info(
            "Orchestrator: Run ID = %s",
            run_identifier,
            extra={"run_id": run_identifier},
        )

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

        return selected_dataset, run_identifier, manifest_payload, run_manifest_hash

    def _load_cache_state(
        self, run_identifier: str, evaluation_config: dict, resume: bool
    ) -> tuple[dict[str, GenerationRecord], dict[str, EvaluationRecord]]:
        """Load cached generation and evaluation records."""
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
                "Orchestrator: Found %s cached generation records",
                len(cached_records),
                extra={"cached_records": len(cached_records)},
            )
        if resume and cached_evaluations:
            logger.info(
                "Orchestrator: Found %s cached evaluation records",
                len(cached_evaluations),
                extra={"cached_evaluations": len(cached_evaluations)},
            )
        return cached_records, cached_evaluations

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

    def _update_metric_stats(
        self, ctx: _ExperimentContext, record: EvaluationRecord
    ) -> None:
        """Update metric statistics from a single evaluation record."""
        ctx.evaluation_record_failures_total += len(record.failures)
        for score in record.scores:
            metric_name = score.metric_name
            ctx.metric_sums[metric_name] = (
                ctx.metric_sums.get(metric_name, 0.0) + score.value
            )
            ctx.metric_counts[metric_name] = ctx.metric_counts.get(metric_name, 0) + 1
            buffer = ctx.metric_samples.setdefault(
                metric_name, _RetentionBuffer(ctx.generation_results.max_items)
            )
            buffer.append(score)

    def _evaluate_batch(
        self,
        ctx: _ExperimentContext,
        run_identifier: str,
        evaluation_config: dict,
        cache_results: bool,
    ) -> None:
        """Evaluate keyed batch of records and update context."""
        if not ctx.eval_batch:
            return

        logger.info(
            "Orchestrator: Evaluating batch of %s records...",
            len(ctx.eval_batch),
            extra={"batch_size": len(ctx.eval_batch)},
        )

        batch_report = self._evaluation.evaluate(ctx.eval_batch)
        ctx.expected_metric_names.update(batch_report.metrics.keys())

        for record, evaluation in zip(ctx.eval_batch, batch_report.records):
            if cache_results:
                self._cache.save_evaluation_record(
                    run_identifier, record, evaluation, evaluation_config
                )
            self._update_metric_stats(ctx, evaluation)
            ctx.new_eval_records.append(evaluation)

        ctx.new_eval_failures.extend(batch_report.failures)
        ctx.eval_batch = []

    def _yield_pending_tasks(
        self,
        ctx: _ExperimentContext,
        task_iterator: Iterator[GenerationTask],
        run_identifier: str,
        run_manifest_hash: str | None,
        evaluation_config: dict,
        cached_records: dict[str, GenerationRecord],
        cached_evaluations: dict[str, EvaluationRecord],
        evaluation_batch_size: int,
        on_result: Callable[[GenerationRecord], None] | None,
        cache_results: bool,
    ) -> Iterator[GenerationTask]:
        """Iterate tasks, handling cached results and yielding pending ones."""
        for task in task_iterator:
            ctx.discovered_tasks_total += 1
            if run_manifest_hash is not None:
                task.metadata["manifest_hash"] = run_manifest_hash

            task_cache_key = experiment_storage.task_cache_key(task)
            cached = cached_records.get(task_cache_key)

            if cached is not None:
                ctx.generation_results.append(cached)
                if cached.error:
                    ctx.failed_generations_total += 1
                else:
                    ctx.successful_generations_total += 1

                if cached.error:
                    ctx.failures.append(
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
                    ctx.cached_eval_records.append(evaluation)
                    self._update_metric_stats(ctx, evaluation)
                    for message in evaluation.failures:
                        ctx.cached_eval_failures.append(
                            EvaluationFailure(
                                sample_id=evaluation.sample_id, message=message
                            )
                        )
                else:
                    ctx.eval_batch.append(cached)
                    if len(ctx.eval_batch) >= evaluation_batch_size:
                        self._evaluate_batch(
                            ctx, run_identifier, evaluation_config, cache_results
                        )

                if on_result:
                    on_result(cached)
                continue

            ctx.pending_tasks_total += 1
            yield task

    def _execute_run_loop(
        self,
        *,
        ctx: _ExperimentContext,
        task_iterator: Iterator[GenerationTask],
        run_identifier: str,
        run_manifest_hash: str | None,
        evaluation_config: dict,
        cached_records: dict[str, GenerationRecord],
        cached_evaluations: dict[str, EvaluationRecord],
        evaluation_batch_size: int,
        on_result: Callable[[GenerationRecord], None] | None,
        cache_results: bool,
    ) -> None:
        """Execute the main generation and evaluation loop."""

        def _flush_eval_batch() -> None:
            self._evaluate_batch(ctx, run_identifier, evaluation_config, cache_results)

        # Run pending generation tasks without pre-materializing task lists.
        completed = 0
        progress = get_progress_reporter(self._integrations)
        progress.__enter__()
        progress_task_id = progress.add_task("Running experiment...", total=None)

        pending_tasks = self._yield_pending_tasks(
            ctx,
            task_iterator,
            run_identifier,
            run_manifest_hash,
            evaluation_config,
            cached_records,
            cached_evaluations,
            evaluation_batch_size,
            on_result,
            cache_results,
        )

        try:
            for record in self._runner.run(pending_tasks):
                logger.debug("Orchestrator: Received generation record")
                ctx.generation_results.append(record)
                if record.error:
                    ctx.failed_generations_total += 1
                else:
                    ctx.successful_generations_total += 1
                completed += 1

                # Update progress
                progress.update(
                    progress_task_id,
                    advance=1,
                    completed=completed,
                    total=None,
                    pending=ctx.pending_tasks_total,
                    successful=ctx.successful_generations_total,
                    failed=ctx.failed_generations_total,
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
                    ctx.failures.append(
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
                ctx.eval_batch.append(record)
                if len(ctx.eval_batch) >= evaluation_batch_size:
                    _flush_eval_batch()

                logger.debug("Orchestrator: Processing record (callback...)")
                if on_result:
                    on_result(record)
                logger.debug("Orchestrator: Record processing complete")
        except Exception as exc:
            self._cache.fail_run(run_identifier, str(exc))
            raise
        finally:
            progress.__exit__(None, None, None)

        logger.info(
            "Orchestrator: Task scan complete (%s total, %s pending, %s cached)",
            ctx.discovered_tasks_total,
            ctx.pending_tasks_total,
            ctx.discovered_tasks_total - ctx.pending_tasks_total,
            extra={
                "total": ctx.discovered_tasks_total,
                "pending": ctx.pending_tasks_total,
                "cached": ctx.discovered_tasks_total - ctx.pending_tasks_total,
            },
        )

        # Evaluate remaining queued records
        logger.info(
            "Orchestrator: Flushing final evaluation batch (%s queued)...",
            len(ctx.eval_batch),
            extra={"queued": len(ctx.eval_batch)},
        )
        try:
            _flush_eval_batch()
        except Exception as exc:
            self._cache.fail_run(run_identifier, str(exc))
            raise

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

    def _finalize_experiment_run(
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
        cache_results: bool,
    ) -> ExperimentReport:
        """Finalize experiment run, including reporting and cleanup."""
        # Get cost breakdown
        cost_breakdown = self._cost_tracker.get_breakdown()
        pricing_metadata = get_pricing_metadata()
        if cost_breakdown.total_cost > 0:
            logger.info(
                "Orchestrator: Total cost: $%s",
                f"{cost_breakdown.total_cost:.4f}",
                extra={"cost": cost_breakdown.total_cost},
            )

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

"""Experiment orchestrator primitives."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from themis.config.schema import IntegrationsConfig
from themis.core.entities import (
    ExperimentFailure,
    ExperimentReport,
    GenerationRecord,
    GenerationTask,
)
from themis.evaluation import pipeline as evaluation_pipeline
from themis import storage as experiment_storage
from themis.exceptions import ConfigurationError
from themis.experiment.cache_manager import CacheManager
from themis.experiment.cost import CostTracker
from themis.experiment.integration_manager import IntegrationManager
from themis.experiment.pricing import calculate_cost

from themis.experiment.context import _ExperimentContext
from themis.experiment.batch_evaluator import BatchEvaluator
from themis.experiment.task_expander import TaskExpander
from themis.experiment.lifecycle import RunLifecycle
from themis.generation import plan as generation_plan
from themis.generation import runner as generation_runner
from themis.utils.progress import get_progress_reporter

logger = logging.getLogger(__name__)


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

        # Initialize sub-coordinators
        self._batch_evaluator = BatchEvaluator(
            evaluation_pipeline=self._evaluation,
            cache_manager=self._cache,
        )
        self._task_expander = TaskExpander(batch_evaluator=self._batch_evaluator)
        self._lifecycle = RunLifecycle(
            plan=self._plan,
            evaluation=self._evaluation,
            cache=self._cache,
            integrations=self._integrations,
        )

    # =========================================================================
    # PRIMARY API
    # =========================================================================

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
        ) = self._lifecycle.initialize_run_resources(
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
        evaluation_config = dict(self._evaluation.evaluation_fingerprint())
        cached_records = (
            self._cache.load_cached_records(run_identifier) if resume else {}
        )
        cached_evaluations = (
            self._cache.load_cached_evaluations(run_identifier, evaluation_config)
            if resume
            else {}
        )

        # Process tasks: use cached or run new generations
        ctx = _ExperimentContext(
            max_records_in_memory=max_records_in_memory,
            run_identifier=run_identifier,
            evaluation_config=evaluation_config,
            cache_results=cache_results,
        )

        ctx.expected_metric_names.update(self._evaluation.metric_names)

        # Execute generation loop
        self._execute_run_loop(
            ctx=ctx,
            task_iterator=task_iterator,
            run_manifest_hash=run_manifest_hash,
            cached_records=cached_records,
            cached_evaluations=cached_evaluations,
            evaluation_batch_size=evaluation_batch_size,
            on_result=on_result,
        )

        # Combine cached and new evaluations
        logger.info("Orchestrator: Combining cached and new evaluations...")
        try:
            evaluation_report = self._batch_evaluator.combine_evaluations(
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

        return self._lifecycle.finalize_experiment_run(
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
            cost_breakdown=self._cost_tracker.get_breakdown(),
            cache_results=cache_results,
        )

    # =========================================================================
    # =========================================================================
    # GENERATION & EVALUATION LOOP
    # =========================================================================

    def _execute_run_loop(
        self,
        *,
        ctx: _ExperimentContext,
        task_iterator: Iterator[GenerationTask],
        run_manifest_hash: str | None,
        cached_records: dict[str, GenerationRecord],
        cached_evaluations: dict[str, Any],
        evaluation_batch_size: int,
        on_result: Callable[[GenerationRecord], None] | None,
    ) -> None:
        """Execute the main generation and evaluation loop."""

        def _flush_eval_batch() -> None:
            self._batch_evaluator.evaluate_batch(ctx)

        # Run pending generation tasks without pre-materializing task lists.
        completed = 0
        progress = get_progress_reporter(self._integrations)
        progress.__enter__()
        progress_task_id = progress.add_task("Running experiment...", total=None)

        pending_tasks = self._task_expander.yield_pending_tasks(
            ctx,
            task_iterator,
            run_manifest_hash,
            cached_records,
            cached_evaluations,
            evaluation_batch_size,
            on_result,
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
                if ctx.cache_results:
                    self._cache.save_generation_record(
                        ctx.run_identifier, record, cache_key
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
            self._cache.fail_run(ctx.run_identifier, str(exc))
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
            self._cache.fail_run(ctx.run_identifier, str(exc))
            raise

    # =========================================================================

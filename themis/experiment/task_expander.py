from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from themis.core.entities import (
    EvaluationRecord,
    ExperimentFailure,
    GenerationRecord,
    GenerationTask,
)
from themis.evaluation.reports import EvaluationFailure
from themis.experiment.batch_evaluator import BatchEvaluator
from themis.experiment.context import _ExperimentContext
from themis import storage as experiment_storage

logger = logging.getLogger(__name__)


class TaskExpander:
    """Iterates tasks, handles cached results and yields pending ones."""

    def __init__(self, batch_evaluator: BatchEvaluator):
        self._batch_evaluator = batch_evaluator

    def yield_pending_tasks(
        self,
        ctx: _ExperimentContext,
        task_iterator: Iterator[GenerationTask],
        run_manifest_hash: str | None,
        cached_records: dict[str, GenerationRecord],
        cached_evaluations: dict[str, EvaluationRecord],
        evaluation_batch_size: int,
        on_result: Callable[[GenerationRecord], None] | None,
    ) -> Iterator[GenerationTask]:
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
                    task, ctx.evaluation_config
                )
                evaluation = cached_evaluations.get(eval_cache_key)

                if evaluation is not None:
                    ctx.cached_eval_records.append(evaluation)
                    self._batch_evaluator.update_metric_stats(ctx, evaluation)
                    for message in evaluation.failures:
                        ctx.cached_eval_failures.append(
                            EvaluationFailure(
                                sample_id=evaluation.sample_id, message=message
                            )
                        )
                else:
                    ctx.eval_batch.append(cached)
                    if len(ctx.eval_batch) >= evaluation_batch_size:
                        self._batch_evaluator.evaluate_batch(ctx)

                if on_result:
                    on_result(cached)
                continue

            ctx.pending_tasks_total += 1
            yield task

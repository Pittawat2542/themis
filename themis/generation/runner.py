"""Generation runner primitives."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, List

from themis.core import entities as core_entities
from themis.generation import strategies
from themis.interfaces import ModelProvider
from themis.utils import tracing

logger = logging.getLogger(__name__)


class GenerationRunner:
    """Delegates generation tasks to an injected provider with strategy support."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        strategy_resolver: Callable[
            [core_entities.GenerationTask], strategies.GenerationStrategy
        ]
        | None = None,
        max_parallel: int = 1,
        max_retries: int = 3,
        retry_initial_delay: float = 0.5,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay: float | None = 2.0,
    ) -> None:
        self._provider = provider
        self._strategy_resolver = strategy_resolver or (
            lambda task: strategies.SingleAttemptStrategy()
        )
        self._max_parallel = max(1, max_parallel)
        self._max_retries = max(1, int(max_retries))
        self._retry_initial_delay = max(0.0, retry_initial_delay)
        self._retry_backoff_multiplier = max(1.0, retry_backoff_multiplier)
        self._retry_max_delay = (
            retry_max_delay if retry_max_delay is None else max(0.0, retry_max_delay)
        )

    def run(
        self, tasks: Iterable[core_entities.GenerationTask]
    ) -> Iterator[core_entities.GenerationRecord]:
        task_list = list(tasks)
        if not task_list:
            logger.info("Runner: No tasks to execute")
            return
        
        logger.info(f"Runner: Starting execution of {len(task_list)} tasks with {self._max_parallel} workers")
        
        if self._max_parallel <= 1:
            logger.info("Runner: Using sequential execution (1 worker)")
            for i, task in enumerate(task_list, 1):
                logger.debug(f"Runner: Processing task {i}/{len(task_list)}")
                yield self._execute_task(task)
            return

        logger.info(f"Runner: Using parallel execution ({self._max_parallel} workers)")
        with ThreadPoolExecutor(max_workers=self._max_parallel) as executor:
            futures = [executor.submit(self._execute_task, task) for task in task_list]
            completed = 0
            for future in futures:
                try:
                    result = future.result()
                    completed += 1
                    if completed % max(1, len(task_list) // 10) == 0 or completed == len(task_list):
                        logger.debug(f"Runner: Completed {completed}/{len(task_list)} tasks")
                    yield result
                except Exception as e:
                    logger.error(f"Runner: Task execution failed: {e}")
                    raise

    def _run_single_attempt(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        attempt_errors: List[dict[str, object]] = []
        last_error: Exception | None = None
        delay = self._retry_initial_delay
        task_label = task.metadata.get("dataset_id") or task.prompt.template_name
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "Runner: Starting generation for %s (attempt %s/%s)",
                    task_label,
                    attempt,
                    self._max_retries,
                )
                record = self._invoke_provider(task)
                record.metrics["generation_attempts"] = attempt
                if attempt_errors:
                    record.metrics.setdefault("retry_errors", attempt_errors)
                logger.debug("Runner: ✅ Completed %s in %s attempt(s)", task_label, attempt)
                return record
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = exc
                logger.warning(
                    "Runner: ⚠️  Attempt %s/%s for %s failed: %s",
                    attempt,
                    self._max_retries,
                    task_label,
                    str(exc)[:100],  # Truncate long error messages
                )
                attempt_errors.append(
                    {
                        "attempt": attempt,
                        "error": str(exc),
                        "exception_type": exc.__class__.__name__,
                    }
                )
                if attempt >= self._max_retries:
                    break
                if delay > 0:
                    time.sleep(delay)
                delay = self._next_delay(delay)

        return self._build_failure_record(task, attempt_errors, last_error)

    def _invoke_provider(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        start = time.perf_counter()

        with tracing.span("provider_generate", model=task.model.identifier):
            record = self._provider.generate(task)

        elapsed_ms = (time.perf_counter() - start) * 1000
        record.metrics.setdefault("generation_time_ms", elapsed_ms)
        record.metrics.setdefault("prompt_chars", len(task.prompt.text))
        prompt_tokens = record.metrics.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = self._count_tokens(task.prompt.text)
            if prompt_tokens is None:
                prompt_tokens = len(task.prompt.text.split())
            record.metrics["prompt_tokens"] = prompt_tokens
        if record.output:
            record.metrics.setdefault("response_chars", len(record.output.text))
            response_tokens = record.metrics.get("response_tokens")
            if response_tokens is None:
                response_tokens = self._count_tokens(record.output.text)
                if response_tokens is None:
                    response_tokens = len(record.output.text.split())
                record.metrics["response_tokens"] = response_tokens
        return record

    def _next_delay(self, previous_delay: float) -> float:
        if previous_delay <= 0:
            next_delay = self._retry_initial_delay
        else:
            next_delay = previous_delay * self._retry_backoff_multiplier
        if self._retry_max_delay is not None:
            next_delay = min(next_delay, self._retry_max_delay)
        return next_delay

    def _build_failure_record(
        self,
        task: core_entities.GenerationTask,
        attempt_errors: List[dict[str, object]],
        last_error: Exception | None,
    ) -> core_entities.GenerationRecord:
        attempts = len(attempt_errors) or 1
        cause = str(last_error) if last_error else "unknown error"
        message = (
            f"Generation failed for model '{task.model.identifier}' "
            f"after {attempts} attempt(s): {cause}"
        )
        logger.error(
            "All attempts failed for %s after %s tries",
            task.metadata.get("dataset_id") or task.prompt.template_name,
            attempts,
            exc_info=last_error,
        )
        return core_entities.GenerationRecord(
            task=task,
            output=None,
            error=core_entities.ModelError(
                message=message,
                kind="provider_error",
                details={
                    "attempts": attempt_errors,
                    "model": task.model.identifier,
                    "provider": task.model.provider,
                },
            ),
            metrics={"generation_attempts": attempts, "retry_errors": attempt_errors},
        )

    def _execute_task(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        task_id = task.metadata.get("dataset_id", "unknown")
        model_id = task.model.identifier

        with tracing.span("execute_task", task_id=task_id, model=model_id):
            strategy = self._strategy_resolver(task)
            attempt_records: List[core_entities.GenerationRecord] = []

            with tracing.span("expand_strategy"):
                expansion = list(strategy.expand(task))

            for attempt_task in expansion:
                with tracing.span("run_attempt"):
                    attempt_records.append(self._run_single_attempt(attempt_task))

            with tracing.span("aggregate_strategy"):
                aggregated = strategy.aggregate(task, attempt_records)

            aggregated.attempts = attempt_records
            return aggregated

    def _count_tokens(self, text: str) -> int | None:
        counter = getattr(self._provider, "count_tokens", None)
        if callable(counter):
            try:
                return int(counter(text))
            except Exception:  # pragma: no cover - tokenization failure
                return None
        return None

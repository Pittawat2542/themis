"""Generation runner primitives."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from themis.core import entities as core_entities
from themis.exceptions import ConfigurationError
from themis.generation import strategies
from themis.interfaces import StatelessTaskExecutor
from themis.utils import tracing

logger = logging.getLogger(__name__)


class GenerationRunner:
    """Delegates generation tasks to an injected executor with strategy support."""

    _NON_RETRYABLE_ERROR_MARKERS = (
        "authentication",
        "api_key",
        "invalid_request",
        "badrequest",
        "permission",
        "unauthorized",
        "forbidden",
        "not_found",
    )

    def __init__(
        self,
        *,
        executor: StatelessTaskExecutor,
        strategy_resolver: Callable[
            [core_entities.GenerationTask], strategies.GenerationStrategy
        ]
        | None = None,
        execution_backend: object | None = None,
        max_parallel: int = 1,
        max_retries: int = 3,
        retry_initial_delay: float = 0.5,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay: float | None = 2.0,
        max_in_flight_tasks: int | None = None,
    ) -> None:
        self._executor = executor
        self._strategy_resolver = strategy_resolver or (
            lambda task: strategies.SingleAttemptStrategy()
        )
        self._execution_backend = execution_backend
        self._max_parallel = max(1, max_parallel)
        self._max_retries = max(1, int(max_retries))
        self._retry_initial_delay = max(0.0, retry_initial_delay)
        self._retry_backoff_multiplier = max(1.0, retry_backoff_multiplier)
        self._retry_max_delay = (
            retry_max_delay if retry_max_delay is None else max(0.0, retry_max_delay)
        )
        if max_in_flight_tasks is not None and max_in_flight_tasks < 1:
            raise ConfigurationError("max_in_flight_tasks must be >= 1 when provided.")
        default_in_flight = self._max_parallel * 4
        self._max_in_flight_tasks = max_in_flight_tasks or default_in_flight

    def run(
        self, tasks: Iterable[core_entities.GenerationTask]
    ) -> Iterator[core_entities.GenerationRecord]:
        task_iter = iter(tasks)
        try:
            first_task = next(task_iter)
        except StopIteration:
            logger.info("Runner: No tasks to execute")
            return

        task_stream = itertools.chain([first_task], task_iter)
        logger.info(
            "Runner: Starting execution with %s workers (max_in_flight=%s)",
            self._max_parallel,
            self._max_in_flight_tasks,
            extra={
                "workers": self._max_parallel,
                "max_in_flight": self._max_in_flight_tasks,
            },
        )

        if self._execution_backend is not None:
            logger.info("Runner: Using custom execution backend")
            backend = self._execution_backend
            try:
                for result in backend.map(
                    self._execute_task, task_stream, max_workers=self._max_parallel
                ):
                    yield result
            except Exception:
                logger.error("Runner: Execution backend failed", exc_info=True)
                raise
            return

        if self._max_parallel <= 1:
            logger.info("Runner: Using sequential execution (1 worker)")
            for i, task in enumerate(task_stream, 1):
                logger.debug("Runner: Processing task", extra={"task_index": i})
                yield self._execute_task(task)
            return

        logger.info(
            "Runner: Using parallel execution (%s workers)",
            self._max_parallel,
            extra={"workers": self._max_parallel},
        )
        yield from self._run_parallel_streaming(task_stream)

    def _run_parallel_streaming(
        self, tasks: Iterable[core_entities.GenerationTask]
    ) -> Iterator[core_entities.GenerationRecord]:
        task_iter = iter(tasks)
        max_in_flight = max(self._max_parallel, self._max_in_flight_tasks)

        with ThreadPoolExecutor(max_workers=self._max_parallel) as executor:
            futures = set()
            completed = 0

            def _submit_until_full() -> None:
                while len(futures) < max_in_flight:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        return
                    futures.add(executor.submit(self._execute_task, task))

            _submit_until_full()

            while futures:
                for future in as_completed(tuple(futures)):
                    futures.remove(future)
                    try:
                        result = future.result()
                    except Exception:
                        logger.error("Runner: Task execution failed", exc_info=True)
                        raise
                    completed += 1
                    if completed % max(1, self._max_parallel) == 0:
                        logger.debug(
                            "Runner: Completed %s task(s), %s in-flight",
                            completed,
                            len(futures),
                            extra={"completed": completed, "in_flight": len(futures)},
                        )
                    yield result
                    _submit_until_full()
                    break

    def _run_single_attempt(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        attempt_errors: list[dict[str, object]] = []
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
                    extra={
                        "task": task_label,
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                    },
                )
                record = self._invoke_executor(task)
                if record.error is not None:
                    error_message = record.error.message
                    retryable = self._is_retryable_record_error(record)
                    if not retryable:
                        record.metrics["generation_attempts"] = attempt
                        if attempt_errors:
                            record.metrics.setdefault("retry_errors", attempt_errors)
                        return record
                    logger.warning(
                        "Runner: Retryable executor error (attempt %s/%s) for %s: %s",
                        attempt,
                        self._max_retries,
                        task_label,
                        error_message[:100],
                        extra={
                            "task": task_label,
                            "attempt": attempt,
                            "max_retries": self._max_retries,
                            "error": error_message,
                        },
                    )
                    attempt_errors.append(
                        {
                            "attempt": attempt,
                            "error": error_message,
                            "exception_type": record.error.kind,
                        }
                    )
                    last_error = RuntimeError(error_message)
                    if attempt >= self._max_retries:
                        break
                    if delay > 0:
                        time.sleep(delay)
                    delay = self._next_delay(delay)
                    continue

                record.metrics["generation_attempts"] = attempt
                if attempt_errors:
                    record.metrics.setdefault("retry_errors", attempt_errors)
                logger.debug(
                    "Runner: Completed %s in %s attempt(s)",
                    task_label,
                    attempt,
                    extra={"task": task_label, "attempts": attempt},
                )
                return record
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = exc
                logger.warning(
                    "Runner: Attempt %s/%s for %s failed",
                    attempt,
                    self._max_retries,
                    task_label,
                    extra={
                        "task": task_label,
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                        "error": str(exc),
                    },
                    exc_info=True,
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

    def _is_retryable_record_error(
        self, record: core_entities.GenerationRecord
    ) -> bool:
        error = record.error
        if error is None:
            return False
        haystack = f"{error.kind} {error.message}".lower()
        return not any(
            marker in haystack for marker in self._NON_RETRYABLE_ERROR_MARKERS
        )

    def _invoke_executor(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        start = time.perf_counter()

        with tracing.span("executor_execute", model=task.model.identifier):
            record = self._executor.execute(task)

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
        attempt_errors: list[dict[str, object]],
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
            extra={
                "task": task.metadata.get("dataset_id") or task.prompt.template_name,
                "attempts": attempts,
            },
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
            attempt_records: list[core_entities.GenerationRecord] = []

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
        counter = getattr(self._executor, "count_tokens", None)
        if callable(counter):
            try:
                return int(counter(text))
            except Exception:  # pragma: no cover - tokenization failure
                return None
        return None

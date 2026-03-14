"""Bounded async scheduler for streaming candidate work items."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from themis.orchestration._async import run_coroutine_sync

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class WorkSchedulerStats:
    """Execution counters captured for one scheduler run."""

    max_seen_in_flight: int = 0
    max_buffered_work_items: int = 0


@dataclass(frozen=True, slots=True)
class ScheduledResult(Generic[T, R]):
    """One completed work item paired with the originating input."""

    work_item: T
    result: R


class WorkScheduler:
    """Run streamed work items with one bounded global concurrency limit."""

    def __init__(self, max_in_flight_work_items: int = 32) -> None:
        if max_in_flight_work_items < 1:
            raise ValueError("max_in_flight_work_items must be >= 1.")
        self.max_in_flight_work_items = max_in_flight_work_items
        self.last_stats = WorkSchedulerStats()

    def run_generation(
        self,
        work_items: Iterable[T],
        worker: Callable[[T], R | Awaitable[R]],
    ) -> list[ScheduledResult[T, R]]:
        """Run streamed generation work items under the bounded scheduler."""
        return run_coroutine_sync(lambda: self._run_bounded(work_items, worker))

    def run_transforms(
        self,
        work_items: Iterable[T],
        worker: Callable[[T], R | Awaitable[R]],
    ) -> list[ScheduledResult[T, R]]:
        """Run streamed transform work items under the bounded scheduler."""
        return run_coroutine_sync(lambda: self._run_bounded(work_items, worker))

    def run_evaluations(
        self,
        work_items: Iterable[T],
        worker: Callable[[T], R | Awaitable[R]],
    ) -> list[ScheduledResult[T, R]]:
        """Run streamed evaluation work items under the bounded scheduler."""
        return run_coroutine_sync(lambda: self._run_bounded(work_items, worker))

    async def _run_bounded(
        self,
        work_items: Iterable[T],
        worker: Callable[[T], R | Awaitable[R]],
    ) -> list[ScheduledResult[T, R]]:
        queue: asyncio.Queue[tuple[int, T] | None] = asyncio.Queue(
            maxsize=self.max_in_flight_work_items * 2
        )
        results: dict[int, ScheduledResult[T, R]] = {}
        errors: list[BaseException] = []
        in_flight = 0
        max_seen_in_flight = 0
        max_buffered_work_items = 0
        result_lock = asyncio.Lock()

        async def producer() -> None:
            nonlocal max_buffered_work_items
            for index, work_item in enumerate(work_items):
                await queue.put((index, work_item))
                max_buffered_work_items = max(max_buffered_work_items, queue.qsize())
            for _ in range(self.max_in_flight_work_items):
                await queue.put(None)

        async def worker_loop() -> None:
            nonlocal in_flight, max_seen_in_flight
            while True:
                queued = await queue.get()
                if queued is None:
                    queue.task_done()
                    break
                index, work_item = queued
                async with result_lock:
                    in_flight += 1
                    max_seen_in_flight = max(max_seen_in_flight, in_flight)
                try:
                    if inspect.iscoroutinefunction(worker):
                        value = await worker(work_item)
                    else:
                        value = await asyncio.to_thread(worker, work_item)
                        if inspect.isawaitable(value):
                            value = await value
                    async with result_lock:
                        results[index] = ScheduledResult(
                            work_item=work_item,
                            result=value,
                        )
                except BaseException as exc:
                    async with result_lock:
                        errors.append(exc)
                finally:
                    async with result_lock:
                        in_flight -= 1
                    queue.task_done()

        producer_task = asyncio.create_task(producer())
        workers = [
            asyncio.create_task(worker_loop())
            for _ in range(self.max_in_flight_work_items)
        ]

        try:
            await producer_task
            await queue.join()
            await asyncio.gather(*workers)
        finally:
            self.last_stats = WorkSchedulerStats(
                max_seen_in_flight=max_seen_in_flight,
                max_buffered_work_items=max_buffered_work_items,
            )

        if errors:
            raise errors[0]

        return [results[index] for index in sorted(results)]

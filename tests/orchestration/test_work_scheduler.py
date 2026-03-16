from __future__ import annotations

import asyncio

import pytest

from themis.orchestration.work_scheduler import WorkScheduler


@pytest.mark.asyncio
async def test_run_bounded_offloads_sync_hooks_with_coroutine_workers(
    monkeypatch,
) -> None:
    scheduler = WorkScheduler(max_in_flight_work_items=1)
    to_thread_calls: list[str] = []
    original_to_thread = asyncio.to_thread

    async def recording_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append(getattr(func, "__name__", func.__class__.__name__))
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording_to_thread)

    async def worker(item: int) -> int:
        return item + 1

    def on_started(item: int) -> None:
        del item

    def on_finished(item: int, value: int | None, error: BaseException | None) -> None:
        del item, value, error

    results = await scheduler._run_bounded(
        [1],
        worker,
        on_work_item_started=on_started,
        on_work_item_finished=on_finished,
    )

    assert [result.result for result in results] == [2]
    assert to_thread_calls == ["on_started", "on_finished"]

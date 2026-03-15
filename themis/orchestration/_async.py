"""Helpers for running async orchestration code from synchronous APIs."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coroutine_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine from sync code, falling back to a worker thread in async hosts.

    The coroutine is created lazily inside the execution context so callers do not
    leak un-awaited coroutine warnings when a loop is already running.
    """

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is None or not running_loop.is_running():
        return asyncio.run(coroutine_factory())

    result_container: list[T] = []
    error_container: list[BaseException] = []

    def thread_target() -> None:
        try:
            result_container.append(asyncio.run(coroutine_factory()))
        except BaseException as exc:  # pragma: no cover - exercised via caller tests.
            error_container.append(exc)

    worker = threading.Thread(target=thread_target)
    worker.start()
    worker.join()

    if error_container:
        raise error_container[0]
    return result_container[0]

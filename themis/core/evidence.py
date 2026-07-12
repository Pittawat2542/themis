"""Bounded async adapter for synchronous evidence stores and subscribers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


class EvidenceWriter:
    """Run synchronous evidence calls off-loop with bounded backpressure."""

    def __init__(self, capacity: int) -> None:
        self._capacity = asyncio.Semaphore(capacity)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="themis-evidence"
        )
        self._closed = False

    async def call(
        self,
        function: Callable[..., T],
        *args: object,
        timeout: float,
    ) -> T:
        if self._closed:
            raise RuntimeError("Evidence writer is closed")
        async with self._capacity:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._executor, function, *args)
            return await asyncio.wait_for(future, timeout=timeout)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True)

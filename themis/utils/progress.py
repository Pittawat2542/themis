"""Simple CLI-friendly progress reporter."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable

from tqdm import tqdm


class ProgressReporter(AbstractContextManager["ProgressReporter"]):
    def __init__(
        self,
        *,
        total: int | None,
        description: str = "Processing",
        unit: str = "sample",
        leave: bool = False,
    ) -> None:
        self._total = total
        self._description = description
        self._unit = unit
        self._leave = leave
        self._pbar: tqdm | None = None

    def __enter__(self) -> "ProgressReporter":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def start(self) -> None:
        if self._pbar is None:
            self._pbar = tqdm(
                total=self._total,
                desc=self._description,
                unit=self._unit,
                leave=self._leave,
            )

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def increment(self, step: int = 1) -> None:
        if self._pbar is not None:
            self._pbar.update(step)

    def on_result(self, _record: Any) -> None:
        self.increment()

    def as_callback(self) -> Callable[[Any], None]:
        return self.on_result


__all__ = ["ProgressReporter"]

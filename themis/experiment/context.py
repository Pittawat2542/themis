from __future__ import annotations

from collections import deque
from typing import Any

from themis.core.entities import ExperimentFailure, GenerationRecord
from themis.evaluation.reports import EvaluationFailure


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

    def __init__(
        self,
        max_records_in_memory: int | None,
        run_identifier: str,
        evaluation_config: dict,
        cache_results: bool,
    ) -> None:
        self.run_identifier = run_identifier
        self.evaluation_config = evaluation_config
        self.cache_results = cache_results

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

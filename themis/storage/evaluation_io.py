"""Evaluation record I/O: append and load cached evaluations."""

from __future__ import annotations

import json

from themis.core import entities as core_entities
from themis.core import serialization as core_serialization
from themis.storage.cache_keys import evaluation_cache_key
from themis.storage.context import StorageContext


class EvaluationIO:
    """Handles reading and writing evaluation records."""

    def __init__(self, ctx: StorageContext) -> None:
        self._ctx = ctx

    def append_evaluation(
        self,
        run_id: str,
        record: core_entities.GenerationRecord,
        evaluation: core_entities.EvaluationRecord,
        *,
        eval_id: str = "default",
        evaluation_config: dict | None = None,
    ) -> None:
        """Append evaluation result."""
        with self._ctx.acquire_lock(run_id):
            eval_dir = self._ctx.get_evaluation_dir(run_id, eval_id)
            eval_dir.mkdir(parents=True, exist_ok=True)

            path = eval_dir / "evaluation.jsonl"

            if not self._ctx.fs.file_exists_any_compression(path):
                self._ctx.fs.write_jsonl_with_header(path, [], file_type="evaluation")

            cache_key = evaluation_cache_key(record.task, evaluation_config)

            payload = {
                "cache_key": cache_key,
                "evaluation": core_serialization.serialize_evaluation_record(
                    evaluation
                ),
            }
            self._ctx.fs.atomic_append(path, payload)

    def load_cached_evaluations(
        self,
        run_id: str,
        eval_id: str = "default",
        evaluation_config: dict | None = None,
    ) -> dict[str, core_entities.EvaluationRecord]:
        """Load cached evaluation records."""
        eval_dir = self._ctx.get_evaluation_dir(run_id, eval_id)
        path = eval_dir / "evaluation.jsonl"

        try:
            handle = self._ctx.fs.open_for_read(path)
        except FileNotFoundError:
            return {}

        evaluations: dict[str, core_entities.EvaluationRecord] = {}

        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue

                key = data.get("cache_key")
                if not key:
                    continue

                evaluations[key] = core_serialization.deserialize_evaluation_record(
                    data["evaluation"]
                )

        return evaluations

"""Generation record I/O: append, load cached records, dataset caching."""

from __future__ import annotations

import json
from collections.abc import Iterable

from themis.core import entities as core_entities
from themis.storage.cache_keys import task_cache_key
from themis.storage.context import StorageContext
from themis.storage.serialization import RecordSerializer


class GenerationIO:
    """Handles reading and writing generation records and datasets."""

    def __init__(self, ctx: StorageContext, serializer: RecordSerializer) -> None:
        self._ctx = ctx
        self._serializer = serializer

    def append_record(
        self,
        run_id: str,
        record: core_entities.GenerationRecord,
        *,
        cache_key: str | None = None,
    ) -> None:
        """Append record with atomic write and locking."""
        with self._ctx.acquire_lock(run_id):
            gen_dir = self._ctx.get_generation_dir(run_id)
            gen_dir.mkdir(parents=True, exist_ok=True)

            path = gen_dir / "records.jsonl"

            if not self._ctx.fs.file_exists_any_compression(path):
                self._ctx.fs.write_jsonl_with_header(path, [], file_type="records")

            payload = self._serializer.serialize_record(run_id, record)
            payload["cache_key"] = cache_key or task_cache_key(record.task)

            self._ctx.fs.atomic_append(path, payload)

    def load_cached_records(
        self, run_id: str
    ) -> dict[str, core_entities.GenerationRecord]:
        """Load cached generation records."""
        gen_dir = self._ctx.get_generation_dir(run_id)
        path = gen_dir / "records.jsonl"

        try:
            handle = self._ctx.fs.open_for_read(path)
        except FileNotFoundError:
            return {}

        tasks = self._serializer.load_tasks(run_id)
        records: dict[str, core_entities.GenerationRecord] = {}

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

                record = self._serializer.deserialize_record(data, tasks)
                records[key] = record

        return records

    def cache_dataset(self, run_id: str, dataset: Iterable[dict[str, object]]) -> None:
        """Cache dataset samples to storage."""
        if not self._ctx.config.save_dataset:
            return

        with self._ctx.acquire_lock(run_id):
            gen_dir = self._ctx.get_generation_dir(run_id)
            gen_dir.mkdir(parents=True, exist_ok=True)
            path = gen_dir / "dataset.jsonl"
            self._ctx.fs.write_jsonl_with_header(path, dataset, file_type="dataset")

    def load_dataset(self, run_id: str) -> list[dict[str, object]]:
        """Load cached dataset."""
        gen_dir = self._ctx.get_generation_dir(run_id)
        path = gen_dir / "dataset.jsonl"

        rows: list[dict[str, object]] = []
        with self._ctx.fs.open_for_read(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue
                rows.append(data)
        return rows

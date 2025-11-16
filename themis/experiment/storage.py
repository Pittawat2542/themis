"""Local storage helpers for experiment datasets and cached records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

from themis.core import entities as core_entities
from themis.core import serialization as core_serialization


def task_cache_key(task: core_entities.GenerationTask) -> str:
    """Derive a stable cache key for a generation task."""

    dataset_raw = task.metadata.get("dataset_id") or task.metadata.get("sample_id")
    dataset_id = str(dataset_raw) if dataset_raw is not None else ""
    prompt_hash = hashlib.sha256(task.prompt.text.encode("utf-8")).hexdigest()[:12]
    sampling = task.sampling
    sampling_key = (
        f"{sampling.temperature:.3f}-{sampling.top_p:.3f}-{sampling.max_tokens}"
    )
    template = task.prompt.spec.name
    model = task.model.identifier
    return "::".join(
        filter(None, [dataset_id, template, model, sampling_key, prompt_hash])
    )


class ExperimentStorage:
    """Persists datasets and generation records for resumability/caching."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._task_index: dict[str, set[str]] = {}

    def cache_dataset(self, run_id: str, dataset: Iterable[dict[str, object]]) -> None:
        path = self._dataset_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(json.dumps(row) + "\n")

    def load_dataset(self, run_id: str) -> List[dict[str, object]]:
        path = self._dataset_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Dataset cache not found for run '{run_id}'")
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
        return rows

    def append_record(
        self,
        run_id: str,
        record: core_entities.GenerationRecord,
        *,
        cache_key: str | None = None,
    ) -> None:
        path = self._records_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize_record(run_id, record)
        payload["cache_key"] = cache_key or task_cache_key(record.task)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def load_cached_records(
        self, run_id: str
    ) -> Dict[str, core_entities.GenerationRecord]:
        path = self._records_path(run_id)
        if not path.exists():
            return {}
        tasks = self._load_tasks(run_id)
        records: dict[str, core_entities.GenerationRecord] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                key = data.get("cache_key")
                if not key:
                    continue
                record = self._deserialize_record(data, tasks)
                records[key] = record
        return records

    def append_evaluation(
        self,
        run_id: str,
        record: core_entities.GenerationRecord,
        evaluation: core_entities.EvaluationRecord,
    ) -> None:
        path = self._evaluation_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": task_cache_key(record.task),
            "evaluation": core_serialization.serialize_evaluation_record(evaluation),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def load_cached_evaluations(
        self, run_id: str
    ) -> Dict[str, core_entities.EvaluationRecord]:
        path = self._evaluation_path(run_id)
        if not path.exists():
            return {}
        evaluations: dict[str, core_entities.EvaluationRecord] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                key = data.get("cache_key")
                if not key:
                    continue
                evaluations[key] = core_serialization.deserialize_evaluation_record(
                    data["evaluation"]
                )
        return evaluations

    def get_run_path(self, run_id: str) -> Path:
        """Get the filesystem path for a run's storage directory.

        Args:
            run_id: Unique run identifier

        Returns:
            Path to the run's storage directory
        """
        return self._run_dir(run_id)

    def _dataset_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "dataset.jsonl"

    def _records_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "records.jsonl"

    def _tasks_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "tasks.jsonl"

    def _evaluation_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "evaluation.jsonl"

    def _run_dir(self, run_id: str) -> Path:
        return self._root / run_id

    def _serialize_record(
        self, run_id: str, record: core_entities.GenerationRecord
    ) -> dict[str, object]:
        task_key = self._persist_task(run_id, record.task)
        payload = {
            "task_key": task_key,
            "output": {
                "text": record.output.text,
                "raw": record.output.raw,
            }
            if record.output
            else None,
            "error": {
                "message": record.error.message,
                "kind": record.error.kind,
                "details": record.error.details,
            }
            if record.error
            else None,
            "metrics": record.metrics,
            "attempts": [
                self._serialize_record(run_id, attempt) for attempt in record.attempts
            ],
        }
        return payload

    def _deserialize_record(
        self, payload: dict[str, object], tasks: dict[str, core_entities.GenerationTask]
    ) -> core_entities.GenerationRecord:
        task_key = payload["task_key"]
        task = tasks[task_key]
        output_data = payload.get("output")
        error_data = payload.get("error")
        record = core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(
                text=output_data["text"], raw=output_data.get("raw")
            )
            if output_data
            else None,
            error=core_entities.ModelError(
                message=error_data["message"],
                kind=error_data.get("kind", "model_error"),
                details=error_data.get("details", {}),
            )
            if error_data
            else None,
            metrics=payload.get("metrics", {}),
        )
        record.attempts = [
            self._deserialize_record(attempt, tasks)
            for attempt in payload.get("attempts", [])
        ]
        return record

    def _persist_task(self, run_id: str, task: core_entities.GenerationTask) -> str:
        key = task_cache_key(task)
        index = self._load_task_index(run_id)
        if key in index:
            return key
        path = self._tasks_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_key": key,
            "task": core_serialization.serialize_generation_task(task),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        index.add(key)
        return key

    def _load_tasks(self, run_id: str) -> dict[str, core_entities.GenerationTask]:
        path = self._tasks_path(run_id)
        tasks: dict[str, core_entities.GenerationTask] = {}
        if not path.exists():
            return tasks
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_key = data["task_key"]
                tasks[task_key] = core_serialization.deserialize_generation_task(
                    data["task"]
                )
        self._task_index[run_id] = set(tasks.keys())
        return tasks

    def _load_task_index(self, run_id: str) -> set[str]:
        if run_id in self._task_index:
            return self._task_index[run_id]
        path = self._tasks_path(run_id)
        index: set[str] = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    index.add(data["task_key"])
        self._task_index[run_id] = index
        return index


__all__ = ["ExperimentStorage", "task_cache_key"]

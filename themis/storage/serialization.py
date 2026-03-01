"""Serialization and deserialization of generation records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from themis.core import entities as core_entities
from themis.core import serialization as core_serialization
from themis.storage.cache_keys import task_cache_key
from themis.storage.context import StorageContext


class RecordSerializer:
    """Handles serialization of generation records and task persistence."""

    def __init__(self, ctx: StorageContext) -> None:
        self._ctx = ctx

    def serialize_record(
        self, run_id: str, record: core_entities.GenerationRecord
    ) -> dict:
        """Serialize generation record."""
        task_key = self.persist_task(run_id, record.task)

        output_data = None
        if record.output:
            output_data = {"text": record.output.text}
            if self._ctx.config.save_raw_responses:
                output_data["raw"] = record.output.raw

        return {
            "task_key": task_key,
            "output": output_data,
            "error": {
                "message": record.error.message,
                "kind": record.error.kind,
                "details": record.error.details,
            }
            if record.error
            else None,
            "metrics": record.metrics,
            "attempts": [
                self.serialize_record(run_id, attempt) for attempt in record.attempts
            ],
        }

    def deserialize_record(
        self, payload: dict, tasks: dict[str, core_entities.GenerationTask]
    ) -> core_entities.GenerationRecord:
        """Deserialize generation record."""
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
            self.deserialize_record(attempt, tasks)
            for attempt in payload.get("attempts", [])
        ]

        return record

    def persist_task(self, run_id: str, task: core_entities.GenerationTask) -> str:
        """Persist task and return cache key."""
        key = task_cache_key(task)
        index = self._load_task_index(run_id)

        if key in index:
            return key

        gen_dir = self._ctx.get_generation_dir(run_id)
        gen_dir.mkdir(parents=True, exist_ok=True)
        path = gen_dir / "tasks.jsonl"

        if not self._ctx.fs.file_exists_any_compression(path):
            self._ctx.fs.write_jsonl_with_header(path, [], file_type="tasks")

        if self._ctx.config.deduplicate_templates:
            template_id = self._persist_template(run_id, task.prompt.spec)
            task_data = core_serialization.serialize_generation_task(task)
            task_data["prompt"]["spec"] = {"_template_ref": template_id}
        else:
            task_data = core_serialization.serialize_generation_task(task)

        payload = {"task_key": key, "task": task_data}
        self._ctx.fs.atomic_append(path, payload)

        index.add(key)
        self._save_task_index(run_id, index)

        return key

    def _persist_template(self, run_id: str, spec: core_entities.PromptSpec) -> str:
        """Persist prompt template."""
        template_content = f"{spec.name}:{spec.template}"
        template_id = hashlib.sha256(template_content.encode("utf-8")).hexdigest()[:16]

        if run_id not in self._ctx.template_index:
            self._ctx.template_index[run_id] = {}
            self.load_templates(run_id)

        if template_id in self._ctx.template_index[run_id]:
            return template_id

        gen_dir = self._ctx.get_generation_dir(run_id)
        path = gen_dir / "templates.jsonl"

        if not self._ctx.fs.file_exists_any_compression(path):
            self._ctx.fs.write_jsonl_with_header(path, [], file_type="templates")

        payload = {
            "template_id": template_id,
            "spec": core_serialization.serialize_prompt_spec(spec),
        }
        self._ctx.fs.atomic_append(path, payload)

        self._ctx.template_index[run_id][template_id] = spec.template
        return template_id

    def _load_task_index(self, run_id: str) -> set[str]:
        """Load task index from disk cache."""
        if run_id in self._ctx.task_index:
            return self._ctx.task_index[run_id]

        index_path = self._ctx.get_run_dir(run_id) / ".index.json"
        if index_path.exists():
            index_data = json.loads(index_path.read_text())
            self._ctx.task_index[run_id] = set(index_data.get("task_keys", []))
            return self._ctx.task_index[run_id]

        self._ctx.task_index[run_id] = set()
        return self._ctx.task_index[run_id]

    def _save_task_index(self, run_id: str, index: set[str]) -> None:
        """Save task index to disk."""
        index_path = self._ctx.get_run_dir(run_id) / ".index.json"
        index_data = {
            "task_keys": list(index),
            "template_ids": self._ctx.template_index.get(run_id, {}),
            "last_updated": datetime.now().isoformat(),
        }
        index_path.write_text(json.dumps(index_data))

    def load_templates(self, run_id: str) -> dict[str, core_entities.PromptSpec]:
        """Load templates from disk."""
        gen_dir = self._ctx.get_generation_dir(run_id)
        path = gen_dir / "templates.jsonl"

        templates: dict[str, core_entities.PromptSpec] = {}
        try:
            handle = self._ctx.fs.open_for_read(path)
        except FileNotFoundError:
            return templates

        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue

                template_id = data["template_id"]
                templates[template_id] = core_serialization.deserialize_prompt_spec(
                    data["spec"]
                )

        return templates

    def load_tasks(self, run_id: str) -> dict[str, core_entities.GenerationTask]:
        """Load tasks from disk."""
        gen_dir = self._ctx.get_generation_dir(run_id)
        path = gen_dir / "tasks.jsonl"

        tasks: dict[str, core_entities.GenerationTask] = {}
        try:
            handle = self._ctx.fs.open_for_read(path)
        except FileNotFoundError:
            return tasks

        templates = (
            self.load_templates(run_id)
            if self._ctx.config.deduplicate_templates
            else {}
        )

        with handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "header":
                    continue

                task_key = data["task_key"]
                task_data = data["task"]

                if (
                    self._ctx.config.deduplicate_templates
                    and "_template_ref" in task_data.get("prompt", {}).get("spec", {})
                ):
                    template_id = task_data["prompt"]["spec"]["_template_ref"]
                    if template_id in templates:
                        task_data["prompt"]["spec"] = (
                            core_serialization.serialize_prompt_spec(
                                templates[template_id]
                            )
                        )

                tasks[task_key] = core_serialization.deserialize_generation_task(
                    task_data
                )

        self._ctx.task_index[run_id] = set(tasks.keys())
        return tasks

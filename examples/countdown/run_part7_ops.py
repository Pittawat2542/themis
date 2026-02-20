from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from themis.storage import RetentionPolicy
from themis.storage import ExperimentStorage

from common import (
    BASELINE_PROMPT,
    make_spec,
    register_countdown_extensions,
    run_spec,
)


if __name__ == "__main__":
    register_countdown_extensions()

    dataset_limit = int(os.getenv("COUNTDOWN_LIMIT", "6"))
    if dataset_limit < 1:
        dataset_limit = 1

    events_path = Path("outputs/countdown_part7/events.jsonl")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        events_path.unlink()

    def on_result(record):
        row = {
            "sample_id": record.task.metadata.get("dataset_id"),
            "has_error": record.error is not None,
            "error_kind": record.error.kind if record.error else None,
            "attempt_count": len(record.attempts),
            "output_preview": record.output.text[:120] if record.output else None,
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    spec = make_spec(
        run_id="countdown-part7-ops",
        prompt=BASELINE_PROMPT,
        dataset_limit=dataset_limit,
        max_records_in_memory=20,
    )
    spec = replace(spec, provider_options={**spec.provider_options, "timeout": 20})
    report = run_spec(
        spec,
        workers=1,
        max_retries=1,
        storage_path=".cache/experiments",
        cache=True,
        on_result=on_result,
    )

    storage = ExperimentStorage(".cache/experiments")
    integrity = storage.validate_integrity("countdown-part7-ops")
    cached_records = storage.load_cached_records("countdown-part7-ops")
    cached_evaluations = storage.load_cached_evaluations("countdown-part7-ops")

    print("retained_generation_records", len(report.generation_results))
    print("dropped_generation_records", report.metadata["generation_records_dropped"])
    print("integrity_valid", integrity["valid"])
    print("cached_generation_records", len(cached_records))
    print("cached_evaluation_records", len(cached_evaluations))
    print("events_file", events_path)
    print("dataset_limit", dataset_limit)

    policy = RetentionPolicy(
        max_runs_per_experiment=20,
        max_age_days=14,
        keep_completed_only=True,
        keep_latest_n=5,
    )
    storage.apply_retention_policy(policy)
    print("retention_applied", True)

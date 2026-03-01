"""Example: Custom storage backend interface.

This script demonstrates how to create a custom storage backend
to persist results to a different medium (e.g. an in-memory mock or simple json),
and then pass it into themis.evaluate().
"""

from typing import Any

import themis
from themis.backends.storage import StorageBackend
from themis.core.entities import GenerationRecord, EvaluationRecord, ExperimentReport


class InMemoryDictStorage(StorageBackend):
    """A simplistic in-memory storage for demonstration."""

    def __init__(self):
        self.runs: dict[str, dict[str, Any]] = {}

    def save_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        if run_id not in self.runs:
            self.runs[run_id] = {
                "metadata": {},
                "generations": [],
                "evaluations": {},
                "report": None,
            }
        self.runs[run_id]["metadata"].update(metadata)

    def load_run_metadata(self, run_id: str) -> dict[str, Any]:
        return self.runs.get(run_id, {}).get("metadata", {})

    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        self.runs[run_id]["generations"].append(record)

    def load_generation_records(self, run_id: str) -> list[GenerationRecord]:
        return self.runs.get(run_id, {}).get("generations", [])

    def save_evaluation_record(
        self, run_id: str, generation_record: GenerationRecord, record: EvaluationRecord
    ) -> None:
        if record.sample_id:
            self.runs[run_id]["evaluations"][record.sample_id] = record

    def load_evaluation_records(self, run_id: str) -> dict[str, EvaluationRecord]:
        return self.runs.get(run_id, {}).get("evaluations", {})

    def save_report(self, run_id: str, report: ExperimentReport) -> None:
        self.runs[run_id]["report"] = report

    def load_report(self, run_id: str) -> ExperimentReport:
        if "report" not in self.runs.get(run_id, {}):
            raise FileNotFoundError(f"Report for {run_id} not found")
        return self.runs[run_id]["report"]

    def list_run_ids(self) -> list[str]:
        return list(self.runs.keys())

    def run_exists(self, run_id: str) -> bool:
        return run_id in self.runs

    def delete_run(self, run_id: str) -> None:
        if run_id in self.runs:
            del self.runs[run_id]


def main():
    # Instantiate custom storage
    my_storage = InMemoryDictStorage()

    # Run evaluation using our custom storage
    report = themis.evaluate(
        "demo",
        model="fake:fake-math-llm",
        limit=3,
        run_id="custom_storage_test",
        storage=my_storage,  # Pass backend directly!
    )

    print("Run ID: custom_storage_test")
    print(f"Accuracy: {report.metric('exact_match').mean:.2%}")
    print(
        f"Successfully wrote {len(my_storage.runs['custom_storage_test']['generations'])} items to in-memory dict."
    )


if __name__ == "__main__":
    main()

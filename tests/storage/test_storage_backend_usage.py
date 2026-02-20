from __future__ import annotations

from pathlib import Path

import themis


class FakeStorage:
    def __init__(self):
        self.start_called = False
        self.cached_dataset_calls: list[dict[str, object]] = []
        self.append_record_calls: list[dict[str, object]] = []
        self.append_evaluation_calls: list[dict[str, object]] = []
        self.load_cached_records_calls: list[str] = []
        self.load_cached_evaluations_calls: list[dict[str, object]] = []

    def run_metadata_exists(self, run_id: str) -> bool:
        return False

    def start_run(
        self,
        run_id: str,
        experiment_id: str = "default",
        config: dict | None = None,
    ) -> None:
        self.start_called = True

    def cache_dataset(self, run_id: str, dataset):
        self.cached_dataset_calls.append(
            {"run_id": run_id, "sample_count": len(dataset)}
        )

    def load_cached_records(self, run_id: str):
        self.load_cached_records_calls.append(run_id)
        return {}

    def load_cached_evaluations(
        self, run_id: str, eval_id: str = "default", evaluation_config=None
    ):
        self.load_cached_evaluations_calls.append(
            {
                "run_id": run_id,
                "eval_id": eval_id,
                "evaluation_config": dict(evaluation_config or {}),
            }
        )
        return {}

    def append_record(self, run_id: str, record, cache_key: str | None = None):
        self.append_record_calls.append(
            {
                "run_id": run_id,
                "cache_key": cache_key,
                "sample_id": record.task.metadata.get("dataset_id"),
            }
        )

    def append_evaluation(
        self,
        run_id: str,
        record,
        evaluation,
        eval_id: str = "default",
        evaluation_config=None,
    ):
        self.append_evaluation_calls.append(
            {
                "run_id": run_id,
                "sample_id": record.task.metadata.get("dataset_id"),
                "eval_id": eval_id,
                "evaluation_config": dict(evaluation_config or {}),
            }
        )

    def get_run_path(self, run_id: str):
        return None


def test_evaluate_uses_storage_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    backend = FakeStorage()

    themis.evaluate(
        [{"id": "1", "question": "2+2", "answer": "4"}],
        model="fake:fake-math-llm",
        prompt="Solve: {question}",
        metrics=["response_length"],
        temperature=0.0,
        run_id="storage-test",
        storage_backend=backend,
    )

    assert backend.start_called is True
    assert backend.cached_dataset_calls == [
        {"run_id": "storage-test", "sample_count": 1}
    ]
    assert len(backend.append_record_calls) == 1
    assert backend.append_record_calls[0]["sample_id"] == "1"
    assert len(backend.append_evaluation_calls) == 1
    assert backend.append_evaluation_calls[0]["sample_id"] == "1"
    assert backend.load_cached_records_calls == ["storage-test"]
    assert len(backend.load_cached_evaluations_calls) == 1
    assert backend.load_cached_evaluations_calls[0]["run_id"] == "storage-test"
    assert Path("None").exists() is False

from __future__ import annotations

from themis.backends.storage import LocalFileStorageBackend
from tests.factories import make_evaluation_record, make_record


def test_local_storage_backend_round_trip_records(tmp_path):
    backend = LocalFileStorageBackend(tmp_path)
    run_id = "run-local-backend"

    backend.save_run_metadata(run_id, {"experiment_id": "exp-a"})
    assert backend.run_exists(run_id) is True

    metadata = backend.load_run_metadata(run_id)
    assert metadata["run_id"] == run_id
    assert metadata["experiment_id"] == "exp-a"

    generation_record = make_record(sample_id="s1")
    backend.save_generation_record(run_id, generation_record)

    loaded_generations = backend.load_generation_records(run_id)
    assert len(loaded_generations) == 1
    assert loaded_generations[0].task.metadata["dataset_id"] == "s1"

    evaluation_record = make_evaluation_record(sample_id="s1", metric_name="ExactMatch", value=1.0)
    backend.save_evaluation_record(run_id, generation_record, evaluation_record)

    loaded_evaluations = backend.load_evaluation_records(run_id)
    assert len(loaded_evaluations) == 1
    assert next(iter(loaded_evaluations.values())).sample_id == "s1"

    assert run_id in backend.list_runs()


def test_local_storage_backend_delete_run(tmp_path):
    backend = LocalFileStorageBackend(tmp_path)
    run_id = "run-delete"
    backend.save_run_metadata(run_id, {"experiment_id": "exp-a"})
    assert backend.run_exists(run_id) is True

    backend.delete_run(run_id)

    assert backend.run_exists(run_id) is False
    assert run_id not in backend.list_runs()


def test_local_storage_backend_lifecycle_methods(tmp_path):
    backend = LocalFileStorageBackend(tmp_path)
    run_id = "run-lifecycle"
    generation_record = make_record(sample_id="s1")
    evaluation_record = make_evaluation_record(
        sample_id="s1", metric_name="ExactMatch", value=1.0
    )

    backend.start_run(run_id, experiment_id="exp-lifecycle", config={"foo": "bar"})
    backend.append_generation_record(run_id, generation_record)
    backend.append_evaluation_record(run_id, generation_record, evaluation_record)
    backend.complete_run(run_id)

    metadata = backend.load_run_metadata(run_id)
    assert metadata["status"] == "completed"
    assert metadata["experiment_id"] == "exp-lifecycle"

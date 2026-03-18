from __future__ import annotations

import pytest

from themis.benchmark.compiler import compile_benchmark
from themis.benchmark.specs import BenchmarkSpec, PromptVariantSpec, SliceSpec
from themis.errors import StorageError
from themis.records.observability import ObservabilityLink
from themis.orchestration.orchestrator import Orchestrator
from themis.orchestration.run_manifest import RunManifest, StageWorkItem, WorkItemStatus
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import (
    BatchExecutionBackendSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ProjectSpec,
    PromptTemplateSpec,
    SqliteBlobStorageSpec,
    TrialSpec,
)
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.storage import build_storage_bundle
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.migrate import migrate_sqlite_store
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import RecordStatus, DatasetSource, RunStage
from themis.types.events import TrialEvent, TimelineStage


def test_migrate_sqlite_store_rebuilds_projections_and_copies_links(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manager = DatabaseManager(f"sqlite:///{source_root / 'themis.sqlite3'}")
    source_manager.initialize()
    event_repo = SqliteEventRepository(source_manager)
    observability_store = SqliteObservabilityStore(source_manager)
    source_blob_store = ArtifactStore(source_root / "artifacts", manager=source_manager)

    blob_ref = source_blob_store.put_blob(b'{"payload":"copied"}', "application/json")
    trial = TrialSpec(
        trial_id="trial_projection",
        model=ModelSpec(model_id="test", provider="fake"),
        task=TaskSpec(
            task_id="task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
        ),
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
    )
    event_repo.save_spec(trial)
    for event in [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type="item_loaded",
            stage=TimelineStage.ITEM_LOAD,
            metadata={"item_id": trial.item_id, "dataset_source": "memory"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type="prompt_rendered",
            stage=TimelineStage.PROMPT_RENDER,
            metadata={"prompt_template_id": "baseline"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type="candidate_started",
            candidate_id="candidate_1",
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=4,
            event_id="evt_4",
            event_type="inference_completed",
            candidate_id="candidate_1",
            stage=TimelineStage.INFERENCE,
            metadata={"provider": "fake", "model_id": "test"},
            payload={"spec_hash": "inf_hash", "raw_text": "42"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type="evaluation_completed",
            candidate_id="candidate_1",
            stage="evaluation",
            metadata={
                "metric_id": "exact_match",
                "score": 1.0,
                "transform_hash": None,
                "evaluation_hash": "eval_1",
            },
            payload={
                "spec_hash": "candidate_1",
                "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=6,
            event_id="evt_6",
            event_type="candidate_completed",
            candidate_id="candidate_1",
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=7,
            event_id="evt_7",
            event_type="projection_completed",
            stage=TimelineStage.PROJECTION,
            metadata={
                "transform_hash": None,
                "evaluation_hash": "eval_1",
                "projection_version": "v1",
            },
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=8,
            event_id="evt_8",
            event_type="trial_completed",
            payload={"status": "ok"},
        ),
    ]:
        event_repo.append_event(event)
    observability_store.save_link(
        trial.spec_hash,
        "candidate_1",
        "evaluation:eval_1",
        ObservabilityLink(
            provider="langfuse",
            external_id="trace-1",
            url="https://langfuse.example/trace/trace-1",
        ),
    )

    destination_bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "destination"))
    )

    migrate_sqlite_store(
        source_db_path=source_root / "themis.sqlite3",
        destination_bundle=destination_bundle,
        source_blob_root=source_root / "artifacts",
    )

    record = destination_bundle.projection_repo.get_trial_record(
        trial.spec_hash,
        evaluation_hash="eval_1",
    )
    assert record is not None
    assert record.status == RecordStatus.OK
    assert record.candidates[0].evaluation.aggregate_scores["exact_match"] == 1.0

    snapshot = destination_bundle.observability_store.get_snapshot(
        trial.spec_hash,
        "candidate_1",
        "evaluation:eval_1",
    )
    assert snapshot is not None
    assert snapshot.url_for("langfuse") == "https://langfuse.example/trace/trace-1"

    assert destination_bundle.blob_store is not None
    assert destination_bundle.blob_store.exists(blob_ref)


def test_migrate_sqlite_store_copies_run_manifests_and_stage_work_items(tmp_path):
    source_root = tmp_path / "source_manifests"
    source_root.mkdir()
    source_manager = DatabaseManager(f"sqlite:///{source_root / 'themis.sqlite3'}")
    source_manager.initialize()
    source_repo = RunManifestRepository(source_manager)
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="test", provider="fake")],
        tasks=[
            TaskSpec(
                task_id="task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
    )
    project = ProjectSpec(
        project_name="migration-project",
        researcher_id="researcher-1",
        global_seed=7,
        storage=SqliteBlobStorageSpec(root_dir=str(source_root)),
        execution_policy=ExecutionPolicySpec(),
    )
    manifest = RunManifest(
        run_id="run_manifest_copy",
        backend_kind="batch",
        project_spec=project,
        experiment_spec=experiment,
        trial_hashes=["trial_hash_1"],
        work_items=[
            StageWorkItem(
                work_item_id="work_pending",
                stage=RunStage.GENERATION,
                status=WorkItemStatus.PENDING,
                trial_hash="trial_hash_1",
                candidate_index=0,
                candidate_id="candidate_1",
                started_at="2026-03-15T10:00:00+00:00",
            ),
            StageWorkItem(
                work_item_id="work_done",
                stage="evaluation",
                status=WorkItemStatus.COMPLETED,
                trial_hash="trial_hash_1",
                candidate_index=0,
                candidate_id="candidate_1",
                evaluation_hash="eval_1",
                ended_at="2026-03-15T10:01:00+00:00",
                last_error_code="metric_failure",
                last_error_message="judge timeout",
            ),
        ],
    )
    source_repo.save_manifest(manifest)

    destination_bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "destination_manifests"))
    )

    migrate_sqlite_store(
        source_db_path=source_root / "themis.sqlite3",
        destination_bundle=destination_bundle,
    )

    migrated = RunManifestRepository(destination_bundle.manager).get_manifest(
        manifest.run_id
    )

    assert migrated is not None
    migrated_payload = migrated.model_dump(mode="json")
    manifest_payload = manifest.model_dump(mode="json")
    assert sorted(
        migrated_payload["work_items"], key=lambda item: item["work_item_id"]
    ) == sorted(manifest_payload["work_items"], key=lambda item: item["work_item_id"])
    migrated_payload["work_items"] = []
    manifest_payload["work_items"] = []
    assert migrated_payload == manifest_payload


def test_migrate_sqlite_store_reads_legacy_source_schema_without_mutating_it(
    tmp_path,
):
    source_root = tmp_path / "source_legacy_columns"
    source_root.mkdir()
    source_manager = DatabaseManager(f"sqlite:///{source_root / 'themis.sqlite3'}")
    source_manager.initialize()
    benchmark = BenchmarkSpec(
        benchmark_id="legacy-benchmark",
        models=[ModelSpec(model_id="test", provider="fake")],
        slices=[
            SliceSpec(
                slice_id="task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="baseline",
                messages=[{"role": "user", "content": "Solve the task."}],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
    )
    manifest = RunManifest(
        run_id="legacy_run_manifest",
        backend_kind="local",
        experiment_spec=compile_benchmark(benchmark),
        benchmark_spec=benchmark,
    )
    RunManifestRepository(source_manager).save_manifest(manifest)
    with source_manager.get_connection() as conn:
        with conn:
            conn.execute("ALTER TABLE run_manifests DROP COLUMN benchmark_spec_json")

    destination_bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "destination_legacy_columns"))
    )

    migrate_sqlite_store(
        source_db_path=source_root / "themis.sqlite3",
        destination_bundle=destination_bundle,
    )

    migrated = RunManifestRepository(destination_bundle.manager).get_manifest(
        manifest.run_id
    )

    assert migrated is not None
    assert migrated.benchmark_spec is not None
    assert migrated.benchmark_spec.benchmark_id == benchmark.benchmark_id
    with source_manager.get_connection() as conn:
        source_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(run_manifests)")
        }
    assert "benchmark_spec_json" not in source_columns


def test_migrate_sqlite_store_rebuilds_overlay_projections_without_blob_copy(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "source_no_blobs"
    source_root.mkdir()
    source_manager = DatabaseManager(f"sqlite:///{source_root / 'themis.sqlite3'}")
    source_manager.initialize()
    event_repo = SqliteEventRepository(source_manager)

    def build_trial(
        trial_id: str, candidate_id: str, evaluation_hash: str
    ) -> TrialSpec:
        return TrialSpec(
            trial_id=trial_id,
            model=ModelSpec(model_id="test", provider="fake"),
            task=TaskSpec(
                task_id=f"task-{trial_id}",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
            ),
            item_id=f"item-{trial_id}",
            prompt=PromptTemplateSpec(id="baseline", messages=[]),
            params=InferenceParamsSpec(),
        )

    for index in range(2):
        trial = build_trial(
            trial_id=f"trial_{index}",
            candidate_id=f"candidate_{index}",
            evaluation_hash=f"eval_{index}",
        )
        candidate_id = f"candidate_{index}"
        evaluation_hash = f"eval_{index}"
        event_repo.save_spec(trial)
        for event in [
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=1,
                event_id=f"evt_{index}_1",
                event_type="item_loaded",
                stage=TimelineStage.ITEM_LOAD,
                metadata={"item_id": trial.item_id, "dataset_source": "memory"},
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=2,
                event_id=f"evt_{index}_2",
                event_type="candidate_started",
                candidate_id=candidate_id,
                payload={"sample_index": 0},
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=3,
                event_id=f"evt_{index}_3",
                event_type="evaluation_completed",
                candidate_id=candidate_id,
                stage="evaluation",
                metadata={
                    "metric_id": "exact_match",
                    "score": 1.0,
                    "transform_hash": None,
                    "evaluation_hash": evaluation_hash,
                },
                payload={
                    "spec_hash": candidate_id,
                    "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
                },
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=4,
                event_id=f"evt_{index}_4",
                event_type="projection_completed",
                stage=TimelineStage.PROJECTION,
                metadata={
                    "transform_hash": None,
                    "evaluation_hash": evaluation_hash,
                    "projection_version": "v1",
                },
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=5,
                event_id=f"evt_{index}_5",
                event_type="trial_completed",
                payload={"status": "ok"},
            ),
        ]:
            event_repo.append_event(event)

    destination_bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "destination_no_blobs"))
    )

    materialized: list[tuple[str, str | None, str | None]] = []
    original_materialize = destination_bundle.projection_repo.materialize_trial_record

    def record_materialize(
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        extra_events=None,
    ):
        materialized.append((trial_hash, transform_hash, evaluation_hash))
        return original_materialize(
            trial_hash,
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
            extra_events=extra_events,
        )

    monkeypatch.setattr(
        destination_bundle.projection_repo,
        "materialize_trial_record",
        record_materialize,
    )

    migrate_sqlite_store(
        source_db_path=source_root / "themis.sqlite3",
        destination_bundle=destination_bundle,
    )

    migrated_trial_hashes = [
        trial.spec_hash
        for trial in [
            build_trial("trial_0", "candidate_0", "eval_0"),
            build_trial("trial_1", "candidate_1", "eval_1"),
        ]
    ]
    for index, trial_hash in enumerate(migrated_trial_hashes):
        record = destination_bundle.projection_repo.get_trial_record(
            trial_hash,
            evaluation_hash=f"eval_{index}",
        )
        assert record is not None
        assert record.status == RecordStatus.OK
        assert (trial_hash, None, None) in materialized
        assert (trial_hash, None, f"eval_{index}") in materialized


def test_migrate_sqlite_store_rejects_legacy_observability_refs_without_links(
    tmp_path,
):
    source_root = tmp_path / "source_legacy"
    source_root.mkdir()
    source_manager = DatabaseManager(f"sqlite:///{source_root / 'themis.sqlite3'}")
    source_manager.initialize()
    with source_manager.get_connection() as conn:
        with conn:
            conn.execute("DROP TABLE observability_links")
            conn.execute(
                """
                INSERT INTO observability_refs (
                    trial_hash,
                    candidate_id,
                    overlay_key,
                    langfuse_trace_id,
                    langfuse_url,
                    extras_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "trial_hash",
                    "candidate_id",
                    "gen",
                    "trace-1",
                    "https://langfuse.example/trace/trace-1",
                    "{}",
                ),
            )

    destination_bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "destination_legacy"))
    )

    with pytest.raises(StorageError, match="observability_links"):
        migrate_sqlite_store(
            source_db_path=source_root / "themis.sqlite3",
            destination_bundle=destination_bundle,
        )


def test_migrate_sqlite_store_allows_empty_legacy_observability_refs_without_links(
    tmp_path,
):
    source_root = tmp_path / "source_legacy_empty_refs"
    source_root.mkdir()
    source_manager = DatabaseManager(f"sqlite:///{source_root / 'themis.sqlite3'}")
    source_manager.initialize()
    with source_manager.get_connection() as conn:
        with conn:
            conn.execute("DROP TABLE observability_links")

    destination_bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "destination_legacy_empty_refs"))
    )

    migrate_sqlite_store(
        source_db_path=source_root / "themis.sqlite3",
        destination_bundle=destination_bundle,
    )


def test_migrated_run_manifest_can_be_resumed_by_orchestrator(tmp_path):
    class DummyEngine:
        def infer(self, trial, context, runtime):
            raise NotImplementedError

    class SingleItemDatasetLoader:
        def load_task_items(self, task):
            del task
            return [{"item_id": "item-1", "question": "6 * 7"}]

    source_root = tmp_path / "source_resume"
    source_project = ProjectSpec(
        project_name="resume-project",
        researcher_id="researcher-1",
        global_seed=7,
        storage=SqliteBlobStorageSpec(root_dir=str(source_root)),
        execution_policy=ExecutionPolicySpec(),
        execution_backend=BatchExecutionBackendSpec(
            provider="openai",
            poll_interval_seconds=30,
            max_batch_items=250,
        ),
    )
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="test", provider="fake")],
        tasks=[
            TaskSpec(
                task_id="task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
    )
    source_orchestrator = Orchestrator.from_project_spec(
        source_project,
        registry=PluginRegistry(),
        dataset_loader=SingleItemDatasetLoader(),
    )
    source_orchestrator.registry.register_inference_engine("fake", DummyEngine())
    handle = source_orchestrator.submit(experiment)
    destination_bundle = build_storage_bundle(
        SqliteBlobStorageSpec(root_dir=str(tmp_path / "destination_resume"))
    )

    migrate_sqlite_store(
        source_db_path=source_root / "themis.sqlite3",
        destination_bundle=destination_bundle,
    )

    destination_orchestrator = Orchestrator.from_project_spec(
        source_project,
        registry=PluginRegistry(),
        dataset_loader=SingleItemDatasetLoader(),
        storage_bundle=destination_bundle,
    )
    destination_orchestrator.registry.register_inference_engine("fake", DummyEngine())
    resumed = destination_orchestrator.resume(handle.run_id)

    assert resumed.run_id == handle.run_id
    assert resumed.status == handle.status

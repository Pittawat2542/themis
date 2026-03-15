from __future__ import annotations

import os
import uuid

import pytest

from themis.orchestration.run_manifest import (
    RunManifest,
    StageWorkItem,
    WorkItemStatus,
)
from themis.records.observability import ObservabilityLink
from themis.specs.experiment import (
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    PostgresBlobStorageSpec,
    PromptTemplateSpec,
    TrialSpec,
)
from themis.specs.foundational import EvaluationSpec
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.storage import build_storage_bundle
from themis.storage.migrate import migrate_sqlite_to_postgres
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import RecordStatus, DatasetSource
from themis.types.events import TrialEvent


def _require_postgres_env() -> tuple[object, str]:
    psycopg = pytest.importorskip("psycopg")
    admin_url = os.environ.get("THEMIS_TEST_POSTGRES_ADMIN_URL")
    if not admin_url:
        pytest.skip("THEMIS_TEST_POSTGRES_ADMIN_URL is not configured")
    return psycopg, admin_url


def _create_database(admin_url: str, psycopg) -> tuple[str, str]:
    db_name = f"themis_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{db_name}"')
    base_url = admin_url.rsplit("/", 1)[0]
    return db_name, f"{base_url}/{db_name}"


def _drop_database(admin_url: str, db_name: str, psycopg) -> None:
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (db_name,),
        )
        conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')


def _fetch_scalar(database_url: str, query: str, params: tuple[object, ...], psycopg):
    rows = getattr(psycopg, "rows")
    with psycopg.connect(database_url, row_factory=rows.dict_row) as conn:
        row = conn.execute(query, params).fetchone()
    assert row is not None
    return next(iter(row.values()))


def test_postgres_bundle_materializes_trial_record_from_event_log(tmp_path):
    psycopg, admin_url = _require_postgres_env()
    db_name, database_url = _create_database(admin_url, psycopg)
    try:
        bundle = build_storage_bundle(
            PostgresBlobStorageSpec(
                database_url=database_url,
                blob_root_dir=str(tmp_path / "blobs"),
            )
        )
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
        bundle.event_repo.save_spec(trial)
        for event in [
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=1,
                event_id="evt_1",
                event_type="item_loaded",
                stage="item_load",  # type: ignore
                metadata={"item_id": trial.item_id, "dataset_source": "memory"},  # type: ignore
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=2,
                event_id="evt_2",
                event_type="prompt_rendered",
                stage="prompt_render",  # type: ignore
                metadata={"prompt_template_id": "baseline"},  # type: ignore
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
                stage="inference",  # type: ignore
                metadata={"provider": "fake", "model_id": "test"},  # type: ignore
                payload={"spec_hash": "inf_hash", "raw_text": "42"},
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=5,
                event_id="evt_5",
                event_type="evaluation_completed",
                candidate_id="candidate_1",
                stage="evaluation",  # type: ignore
                metadata={  # type: ignore
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
                stage="projection",  # type: ignore
                metadata={  # type: ignore
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
            bundle.event_repo.append_event(event)

        record = bundle.projection_repo.materialize_trial_record(
            trial.spec_hash,
            evaluation_hash="eval_1",
        )

        assert record.status == RecordStatus.OK
        assert record.candidates[0].evaluation.aggregate_scores["exact_match"] == 1.0
    finally:
        _drop_database(admin_url, db_name, psycopg)


def test_postgres_bundle_commits_repo_and_store_writes_to_fresh_connections(tmp_path):
    psycopg, admin_url = _require_postgres_env()
    db_name, database_url = _create_database(admin_url, psycopg)
    try:
        bundle = build_storage_bundle(
            PostgresBlobStorageSpec(
                database_url=database_url,
                blob_root_dir=str(tmp_path / "blobs"),
            )
        )
        trial = TrialSpec(
            trial_id="trial_commit_visibility",
            model=ModelSpec(model_id="test", provider="fake"),
            task=TaskSpec(
                task_id="task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                evaluations=[EvaluationSpec(name="default", metrics=["exact_match"])],
            ),
            item_id="item-1",
            prompt=PromptTemplateSpec(id="baseline", messages=[]),
            params=InferenceParamsSpec(),
        )

        bundle.event_repo.save_spec(trial)
        assert (
            _fetch_scalar(
                database_url,
                "SELECT COUNT(*) AS count FROM specs WHERE spec_hash = %s",
                (trial.spec_hash,),
                psycopg,
            )
            == 1
        )

        bundle.event_repo.append_event(
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=1,
                event_id="evt_1",
                event_type="candidate_started",
                candidate_id="candidate_1",
                payload={"sample_index": 0},
            )
        )
        assert (
            _fetch_scalar(
                database_url,
                "SELECT COUNT(*) AS count FROM trial_events WHERE trial_hash = %s",
                (trial.spec_hash,),
                psycopg,
            )
            == 1
        )

        manifest = RunManifest(
            run_id="run_postgres_commit_visibility",
            backend_kind="local",
            experiment_spec=ExperimentSpec(
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
            ),
        )
        RunManifestRepository(bundle.manager).save_manifest(manifest)
        assert (
            _fetch_scalar(
                database_url,
                "SELECT COUNT(*) AS count FROM run_manifests WHERE run_id = %s",
                (manifest.run_id,),
                psycopg,
            )
            == 1
        )

        SqliteObservabilityStore(bundle.manager).save_link(
            trial.spec_hash,
            "candidate_1",
            "gen",
            ObservabilityLink(
                provider="langfuse",
                external_id="trace-1",
                url="https://langfuse.example/trace/trace-1",
            ),
        )
        assert (
            _fetch_scalar(
                database_url,
                "SELECT COUNT(*) AS count FROM observability_links WHERE trial_hash = %s",
                (trial.spec_hash,),
                psycopg,
            )
            == 1
        )

        for event in [
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=2,
                event_id="evt_2",
                event_type="evaluation_completed",
                candidate_id="candidate_1",
                stage="evaluation",  # type: ignore
                metadata={  # type: ignore
                    "metric_id": "exact_match",
                    "score": 1.0,
                    "evaluation_hash": "eval_1",
                },
                payload={
                    "spec_hash": "candidate_1",
                    "metric_scores": [{"metric_id": "exact_match", "value": 1.0}],
                },
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=3,
                event_id="evt_3",
                event_type="candidate_completed",
                candidate_id="candidate_1",
                payload={"status": "ok"},
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=4,
                event_id="evt_4",
                event_type="projection_completed",
                stage="projection",  # type: ignore
                metadata={  # type: ignore
                    "evaluation_hash": "eval_1",
                    "projection_version": "v1",
                },
            ),
            TrialEvent(
                trial_hash=trial.spec_hash,
                event_seq=5,
                event_id="evt_5",
                event_type="trial_completed",
                payload={"status": "ok"},
            ),
        ]:
            bundle.event_repo.append_event(event)
        bundle.projection_repo.materialize_trial_record(
            trial.spec_hash,
            evaluation_hash="eval_1",
        )
        assert (
            _fetch_scalar(
                database_url,
                "SELECT COUNT(*) AS count FROM trial_summary WHERE trial_hash = %s AND overlay_key = %s",
                (trial.spec_hash, "ev:eval_1"),
                psycopg,
            )
            == 1
        )
    finally:
        _drop_database(admin_url, db_name, psycopg)


def test_migrate_sqlite_to_postgres_copies_run_manifests_and_stage_work_items(tmp_path):
    psycopg, admin_url = _require_postgres_env()
    db_name, database_url = _create_database(admin_url, psycopg)
    try:
        source_root = tmp_path / "source_sqlite"
        source_root.mkdir()
        source_manager = DatabaseManager(f"sqlite:///{source_root / 'themis.sqlite3'}")
        source_manager.initialize()
        source_repo = RunManifestRepository(source_manager)
        manifest = RunManifest(
            run_id="run_migrate_postgres",
            backend_kind="batch",
            experiment_spec=ExperimentSpec(
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
            ),
            work_items=[
                StageWorkItem(
                    work_item_id="work_pending",
                    stage="generation",  # type: ignore
                    status=WorkItemStatus.PENDING,
                    trial_hash="trial_hash_1",
                    candidate_index=0,
                    candidate_id="candidate_1",
                )
            ],
        )
        source_repo.save_manifest(manifest)

        destination_bundle = migrate_sqlite_to_postgres(
            source_db_path=source_root / "themis.sqlite3",
            database_url=database_url,
            blob_root_dir=tmp_path / "postgres_blobs",
        )

        migrated = RunManifestRepository(destination_bundle.manager).get_manifest(
            manifest.run_id
        )

        assert migrated is not None
        assert migrated.model_dump(mode="json") == manifest.model_dump(mode="json")
    finally:
        _drop_database(admin_url, db_name, psycopg)

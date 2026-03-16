import sqlite3

import pytest

from themis.errors import StorageError
from themis.storage import _schema as storage_schema
from themis.storage.postgres import manager as postgres_manager
from themis.storage.sqlite_schema import DatabaseManager
from themis import storage as storage_module


def test_database_manager_initialization(tmp_path):
    db_path = tmp_path / "test.db"
    uri = f"sqlite:///{db_path}"

    manager = DatabaseManager(uri)
    manager.initialize()

    assert db_path.exists()

    # Verify tables were created
    with manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row["name"] for row in cursor.fetchall()}

    expected_tables = {
        "specs",
        "store_metadata",
        "artifacts",
        "run_manifests",
        "stage_work_items",
        "trial_summary",
        "trial_events",
        "candidate_summary",
        "metric_scores",
        "record_timeline",
        "observability_refs",
    }

    # We allow more tables (like sqlite_sequence) but must have these
    assert expected_tables.issubset(tables)

    with manager.get_connection() as conn:
        spec_columns = {row["name"] for row in conn.execute("PRAGMA table_info(specs)")}
        assert {
            "spec_hash",
            "canonical_hash",
            "spec_type",
            "schema_version",
            "canonical_json",
        }.issubset(spec_columns)

        event_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(trial_events)")
        }
        assert {
            "trial_hash",
            "event_seq",
            "event_id",
            "candidate_id",
            "event_type",
            "stage",
            "event_ts",
            "metadata_json",
            "payload_json",
            "artifact_refs_json",
            "error_json",
        }.issubset(event_columns)

        timeline_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(record_timeline)")
        }
        assert {
            "record_id",
            "record_type",
            "trial_hash",
            "candidate_id",
            "stage_order",
            "stage_name",
            "status",
            "metadata_json",
            "artifacts_json",
            "error_json",
            "overlay_key",
        }.issubset(timeline_columns)

        candidate_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(candidate_summary)")
        }
        assert {
            "candidate_id",
            "trial_hash",
            "sample_index",
            "status",
            "overlay_key",
        }.issubset(candidate_columns)

        trial_summary_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(trial_summary)")
        }
        assert {
            "overlay_key",
            "started_at",
            "ended_at",
            "duration_ms",
            "has_conversation",
            "has_logprobs",
            "has_trace",
            "tags_json",
        }.issubset(trial_summary_columns)

        metric_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(metric_scores)")
        }
        assert {
            "candidate_id",
            "metric_id",
            "score",
            "details_json",
            "overlay_key",
        }.issubset(metric_columns)

        observability_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(observability_refs)")
        }
        assert {"trial_hash", "candidate_id", "overlay_key"}.issubset(
            observability_columns
        )

        store_format_row = conn.execute(
            """
            SELECT metadata_value
            FROM store_metadata
            WHERE metadata_key = 'store_format'
            """
        ).fetchone()
        assert store_format_row is not None
        assert store_format_row["metadata_value"] == "stage_overlays_v2"


def test_database_manager_invalid_uri():
    with pytest.raises(ValueError, match="must standard 'sqlite:///'"):
        DatabaseManager("postgres://bad")


def test_database_manager_rejects_pre_vnext_store(tmp_path):
    db_path = tmp_path / "legacy.db"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE specs (
            spec_hash TEXT PRIMARY KEY,
            spec_type TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            canonical_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    manager = DatabaseManager(f"sqlite:///{db_path}")

    with pytest.raises(StorageError, match="unsupported store format"):
        manager.initialize()


def test_database_manager_migrates_stage_work_item_progress_columns(tmp_path):
    db_path = tmp_path / "legacy-stage-work-items.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE store_metadata (
            metadata_key TEXT PRIMARY KEY,
            metadata_value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO store_metadata (metadata_key, metadata_value) VALUES (?, ?)",
        ("store_format", storage_schema.STORE_FORMAT_VERSION),
    )
    conn.execute(
        """
        CREATE TABLE specs (
            spec_hash TEXT PRIMARY KEY,
            canonical_hash TEXT,
            spec_type TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            canonical_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE trial_summary (
            trial_hash TEXT PRIMARY KEY,
            overlay_key TEXT,
            model_id TEXT,
            task_id TEXT,
            item_id TEXT,
            status TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE stage_work_items (
            work_item_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            trial_hash TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            candidate_id TEXT NOT NULL,
            transform_hash TEXT,
            evaluation_hash TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at TEXT,
            external_job_id TEXT,
            artifact_refs_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize()

    with manager.get_connection() as upgraded:
        columns = {
            row["name"]
            for row in upgraded.execute("PRAGMA table_info(stage_work_items)")
        }

    assert {
        "started_at",
        "ended_at",
        "last_error_code",
        "last_error_message",
    }.issubset(columns)


def test_sqlite_and_postgres_share_storage_schema_contract():
    assert storage_module.sqlite_schema.SCHEMA is storage_schema.SCHEMA
    assert postgres_manager.SCHEMA is storage_schema.SCHEMA
    assert (
        storage_module.sqlite_schema.STORE_FORMAT_KEY is storage_schema.STORE_FORMAT_KEY
    )
    assert postgres_manager.STORE_FORMAT_KEY is storage_schema.STORE_FORMAT_KEY
    assert (
        storage_module.sqlite_schema.STORE_FORMAT_VERSION
        is storage_schema.STORE_FORMAT_VERSION
    )
    assert postgres_manager.STORE_FORMAT_VERSION is storage_schema.STORE_FORMAT_VERSION


def test_apply_sql_script_executes_individual_statements():
    executed: list[tuple[str, tuple[object, ...]]] = []

    class FakeConnection:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] = (),
        ) -> None:
            executed.append((query, params))

    storage_schema.apply_sql_script(FakeConnection(), "SELECT 1; ; SELECT 2;")

    assert executed == [
        ("SELECT 1", ()),
        ("SELECT 2", ()),
    ]

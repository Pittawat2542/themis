import pytest
from themis.storage.sqlite_schema import DatabaseManager


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
        "artifacts",
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
            "eval_revision",
        }.issubset(timeline_columns)

        candidate_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(candidate_summary)")
        }
        assert {
            "candidate_id",
            "trial_hash",
            "sample_index",
            "status",
            "eval_revision",
        }.issubset(candidate_columns)

        trial_summary_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(trial_summary)")
        }
        assert {
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
            "eval_revision",
        }.issubset(metric_columns)


def test_database_manager_invalid_uri():
    with pytest.raises(ValueError, match="must standard 'sqlite:///'"):
        DatabaseManager("postgres://bad")

from __future__ import annotations

from themis.cli.quickcheck import main
from themis.storage.sqlite_schema import DatabaseManager


def _seed_quickcheck_db(manager: DatabaseManager) -> None:
    with manager.get_connection() as conn:
        with conn:
            conn.executemany(
                """
                INSERT INTO specs (spec_hash, spec_type, schema_version, canonical_json)
                VALUES (?, 'TrialSpec', '1.0', '{}')
                """,
                [("trial_err",), ("trial_ok",)],
            )
            conn.execute(
                """
                INSERT INTO trial_summary (
                    trial_hash, model_id, task_id, item_id, status, started_at, ended_at, duration_ms, error_fingerprint, error_preview
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "trial_err",
                    "baseline",
                    "math",
                    "item-1",
                    "error",
                    "2026-03-09T10:00:00+00:00",
                    "2026-03-09T10:00:02+00:00",
                    2000,
                    "fp-1",
                    "provider timeout",
                ),
            )
            conn.execute(
                """
                INSERT INTO trial_summary (
                    trial_hash, model_id, task_id, item_id, status, started_at, ended_at, duration_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "trial_ok",
                    "treatment",
                    "math",
                    "item-2",
                    "ok",
                    "2026-03-09T11:00:00+00:00",
                    "2026-03-09T11:00:01+00:00",
                    1000,
                ),
            )
            conn.executemany(
                """
                INSERT INTO candidate_summary (
                    candidate_id, trial_hash, eval_revision, sample_index, status, finish_reason, tokens_in, tokens_out, latency_ms
                )
                VALUES (?, ?, 'latest', ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("cand_err", "trial_err", 0, "error", None, 100, 20, 2100),
                    ("cand_ok", "trial_ok", 0, "ok", "stop", 120, 25, 900),
                ],
            )
            conn.executemany(
                """
                INSERT INTO metric_scores (
                    candidate_id, eval_revision, metric_id, score, details_json
                )
                VALUES (?, 'latest', ?, ?, ?)
                """,
                [
                    ("cand_err", "em", 0.0, "{}"),
                    ("cand_ok", "em", 1.0, "{}"),
                ],
            )


def test_quickcheck_cli_reads_failures_scores_and_latency_from_sqlite_summaries(
    tmp_path, capsys
):
    db_path = tmp_path / "quickcheck.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize()
    _seed_quickcheck_db(manager)

    assert main(["failures", "--db", str(db_path), "--limit", "1"]) == 0
    failures_output = capsys.readouterr().out
    assert "trial_err" in failures_output
    assert "provider timeout" in failures_output

    assert main(["scores", "--db", str(db_path), "--metric", "em"]) == 0
    scores_output = capsys.readouterr().out
    assert "baseline" in scores_output
    assert "treatment" in scores_output

    assert main(["latency", "--db", str(db_path)]) == 0
    latency_output = capsys.readouterr().out
    assert "count=2" in latency_output
    assert "latency_ms" in latency_output

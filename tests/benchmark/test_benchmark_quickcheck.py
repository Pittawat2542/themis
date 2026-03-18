from __future__ import annotations

from themis.cli.quickcheck import main
from themis.storage.sqlite_schema import DatabaseManager


def _seed_benchmark_quickcheck_db(manager: DatabaseManager) -> None:
    with manager.get_connection() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO specs (
                    spec_hash,
                    canonical_hash,
                    spec_type,
                    schema_version,
                    canonical_json
                )
                VALUES (?, ?, 'TrialSpec', '1.0', '{}')
                """,
                (
                    "trial-1",
                    "trial100000000000000000000000000000000000000000000000000000000000",
                ),
            )
            conn.execute(
                """
                INSERT INTO trial_summary (
                    trial_hash,
                    overlay_key,
                    benchmark_id,
                    model_id,
                    task_id,
                    slice_id,
                    prompt_variant_id,
                    dimensions_json,
                    item_id,
                    status,
                    started_at,
                    ended_at,
                    duration_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "trial-1",
                    "ev:eval_a",
                    "bench-1",
                    "model-a",
                    "qa",
                    "qa",
                    "qa-default",
                    '{"source":"synthetic","format":"qa"}',
                    "item-1",
                    "ok",
                    "2026-03-09T10:00:00+00:00",
                    "2026-03-09T10:00:01+00:00",
                    1000,
                ),
            )
            conn.execute(
                """
                INSERT INTO candidate_summary (
                    candidate_id,
                    trial_hash,
                    overlay_key,
                    sample_index,
                    status,
                    finish_reason,
                    tokens_in,
                    tokens_out,
                    latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("cand-1", "trial-1", "ev:eval_a", 0, "ok", "stop", 100, 20, 900),
            )
            conn.execute(
                """
                INSERT INTO metric_scores (
                    candidate_id,
                    overlay_key,
                    metric_id,
                    score,
                    details_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                ("cand-1", "ev:eval_a", "accuracy", 1.0, "{}"),
            )


def test_quickcheck_scores_supports_slice_and_dimension_filters(tmp_path, capsys):
    db_path = tmp_path / "quickcheck.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize()
    _seed_benchmark_quickcheck_db(manager)

    assert (
        main(
            [
                "scores",
                "--db",
                str(db_path),
                "--metric",
                "accuracy",
                "--slice",
                "qa",
                "--dimension",
                "source=synthetic",
                "--evaluation-hash",
                "eval_a",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "qa" in output
    assert "accuracy" in output
    assert "1.0000" in output

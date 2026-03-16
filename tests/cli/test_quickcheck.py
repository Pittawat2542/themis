from __future__ import annotations

from types import SimpleNamespace

from themis.cli.quickcheck import build_parser, main, run_with_args
from themis.storage.sqlite_schema import DatabaseManager


def _seed_quickcheck_db(manager: DatabaseManager) -> None:
    with manager.get_connection() as conn:
        with conn:
            conn.executemany(
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
                [
                    (
                        "trial_err",
                        "trialerr00000000000000000000000000000000000000000000000000000000",
                    ),
                    (
                        "trial_ok",
                        "trialok000000000000000000000000000000000000000000000000000000000",
                    ),
                    (
                        "trial_eval_err",
                        "trialevalerr0000000000000000000000000000000000000000000000000000",
                    ),
                ],
            )
            conn.execute(
                """
                INSERT INTO trial_summary (
                    trial_hash, overlay_key, model_id, task_id, item_id, status, started_at, ended_at, duration_ms, error_fingerprint, error_preview
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "trial_err",
                    "gen",
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
                    trial_hash, overlay_key, model_id, task_id, item_id, status, started_at, ended_at, duration_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "trial_ok",
                    "gen",
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
                INSERT INTO trial_summary (
                    trial_hash, overlay_key, model_id, task_id, item_id, status, started_at, ended_at, duration_ms, error_fingerprint, error_preview
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "trial_ok",
                        "ev:eval_a",
                        "treatment",
                        "math",
                        "item-2",
                        "ok",
                        "2026-03-09T11:00:00+00:00",
                        "2026-03-09T11:00:01+00:00",
                        1000,
                        None,
                        None,
                    ),
                    (
                        "trial_ok",
                        "ev:eval_b",
                        "treatment",
                        "math",
                        "item-2",
                        "ok",
                        "2026-03-09T11:00:00+00:00",
                        "2026-03-09T11:00:01+00:00",
                        1000,
                        None,
                        None,
                    ),
                    (
                        "trial_eval_err",
                        "ev:eval_failed",
                        "treatment",
                        "math",
                        "item-3",
                        "error",
                        "2026-03-09T12:00:00+00:00",
                        "2026-03-09T12:00:02+00:00",
                        2000,
                        "fp-eval",
                        "metric failed",
                    ),
                ],
            )
            conn.executemany(
                """
                INSERT INTO candidate_summary (
                    candidate_id, trial_hash, overlay_key, sample_index, status, finish_reason, tokens_in, tokens_out, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("cand_err", "trial_err", "gen", 0, "error", None, 100, 20, 2100),
                    ("cand_ok", "trial_ok", "gen", 0, "ok", "stop", 120, 25, 900),
                    (
                        "cand_eval_a",
                        "trial_ok",
                        "ev:eval_a",
                        0,
                        "ok",
                        "stop",
                        120,
                        25,
                        950,
                    ),
                    (
                        "cand_eval_b",
                        "trial_ok",
                        "ev:eval_b",
                        0,
                        "ok",
                        "stop",
                        120,
                        25,
                        975,
                    ),
                    (
                        "cand_eval_err",
                        "trial_eval_err",
                        "ev:eval_failed",
                        0,
                        "error",
                        None,
                        120,
                        25,
                        990,
                    ),
                ],
            )
            conn.executemany(
                """
                INSERT INTO metric_scores (
                    candidate_id, overlay_key, metric_id, score, details_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("cand_eval_a", "ev:eval_a", "em", 0.25, "{}"),
                    ("cand_eval_b", "ev:eval_b", "em", 0.75, "{}"),
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
    assert "gen" in failures_output
    assert "provider timeout" in failures_output

    assert main(["scores", "--db", str(db_path), "--metric", "em"]) == 0
    scores_output = capsys.readouterr().out
    assert "treatment" in scores_output
    assert "ev:eval_a" in scores_output
    assert "ev:eval_b" in scores_output

    assert main(["latency", "--db", str(db_path)]) == 0
    latency_output = capsys.readouterr().out
    assert "count=2" in latency_output
    assert "latency_ms" in latency_output


def test_quickcheck_scores_can_filter_to_one_evaluation_overlay(tmp_path, capsys):
    db_path = tmp_path / "quickcheck_eval.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize()
    _seed_quickcheck_db(manager)

    assert (
        main(
            [
                "scores",
                "--db",
                str(db_path),
                "--metric",
                "em",
                "--evaluation-hash",
                "eval_b",
            ]
        )
        == 0
    )
    scores_output = capsys.readouterr().out

    assert "ev:eval_b" in scores_output
    assert "0.7500" in scores_output
    assert "ev:eval_a" not in scores_output


def test_quickcheck_latency_can_select_evaluation_overlay(tmp_path, capsys):
    db_path = tmp_path / "quickcheck_latency_eval.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize()
    _seed_quickcheck_db(manager)

    assert (
        main(
            [
                "latency",
                "--db",
                str(db_path),
                "--evaluation-hash",
                "eval_a",
            ]
        )
        == 0
    )
    latency_output = capsys.readouterr().out

    assert "count=1" in latency_output
    assert "950.00" in latency_output


def test_quickcheck_failures_can_select_evaluation_overlay(tmp_path, capsys):
    db_path = tmp_path / "quickcheck_failures_eval.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize()
    _seed_quickcheck_db(manager)

    assert (
        main(
            [
                "failures",
                "--db",
                str(db_path),
                "--evaluation-hash",
                "eval_failed",
            ]
        )
        == 0
    )
    failures_output = capsys.readouterr().out

    assert "trial_eval_err" in failures_output
    assert "ev:eval_failed" in failures_output
    assert "metric failed" in failures_output


def test_quickcheck_parser_sets_parser_default() -> None:
    args = build_parser().parse_args(["failures", "--db", "example.db"])

    assert args._parser.prog == "themis-quickcheck"


def test_quickcheck_run_with_args_returns_error_code_for_unknown_command(
    tmp_path,
) -> None:
    db_path = tmp_path / "quickcheck_empty.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.initialize()

    assert (
        run_with_args(
            SimpleNamespace(
                command="unknown",
                db=str(db_path),
                limit=10,
                transform_hash=None,
                evaluation_hash=None,
                metric=None,
                task=None,
            )
        )
        == 2
    )

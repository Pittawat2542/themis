"""Small CLI for reading projection summaries from a Themis SQLite database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from statistics import mean


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="themis-quickcheck")
    subparsers = parser.add_subparsers(dest="command", required=True)

    failures = subparsers.add_parser("failures")
    failures.add_argument("--db", required=True)
    failures.add_argument("--limit", type=int, default=10)

    scores = subparsers.add_parser("scores")
    scores.add_argument("--db", required=True)
    scores.add_argument("--metric")
    scores.add_argument("--task")

    latency = subparsers.add_parser("latency")
    latency.add_argument("--db", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the quickcheck CLI and dispatch to the selected summary command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    with _connect(str(db_path)) as conn:
        if args.command == "failures":
            return _run_failures(conn, limit=args.limit)
        if args.command == "scores":
            return _run_scores(conn, metric_id=args.metric, task_id=args.task)
        if args.command == "latency":
            return _run_latency(conn)
    parser.error("Unknown command.")
    return 2


def _run_failures(conn: sqlite3.Connection, *, limit: int) -> int:
    rows = conn.execute(
        """
        SELECT trial_hash, model_id, task_id, item_id, error_fingerprint, error_preview
        FROM trial_summary
        WHERE status = 'error'
        ORDER BY COALESCE(ended_at, updated_at, created_at) DESC, trial_hash ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    for row in rows:
        print(
            "\t".join(
                [
                    row["trial_hash"],
                    row["model_id"] or "",
                    row["task_id"] or "",
                    row["item_id"] or "",
                    row["error_fingerprint"] or "",
                    row["error_preview"] or "",
                ]
            )
        )
    return 0


def _run_scores(
    conn: sqlite3.Connection, *, metric_id: str | None, task_id: str | None
) -> int:
    clauses = [
        "candidate_summary.eval_revision = 'latest'",
        "metric_scores.eval_revision = candidate_summary.eval_revision",
    ]
    params: list[object] = []
    if metric_id is not None:
        clauses.append("metric_scores.metric_id = ?")
        params.append(metric_id)
    if task_id is not None:
        clauses.append("trial_summary.task_id = ?")
        params.append(task_id)
    where_clause = f"WHERE {' AND '.join(clauses)}"
    rows = conn.execute(
        f"""
        SELECT trial_summary.model_id, trial_summary.task_id, metric_scores.metric_id, AVG(metric_scores.score) AS avg_score, COUNT(*) AS row_count
        FROM metric_scores
        JOIN candidate_summary
          ON candidate_summary.candidate_id = metric_scores.candidate_id
         AND candidate_summary.eval_revision = metric_scores.eval_revision
        JOIN trial_summary
          ON trial_summary.trial_hash = candidate_summary.trial_hash
        {where_clause}
        GROUP BY trial_summary.model_id, trial_summary.task_id, metric_scores.metric_id
        ORDER BY trial_summary.model_id ASC, metric_scores.metric_id ASC
        """,
        params,
    ).fetchall()
    for row in rows:
        print(
            "\t".join(
                [
                    row["model_id"] or "",
                    row["task_id"] or "",
                    row["metric_id"] or "",
                    f"{row['avg_score']:.4f}",
                    str(row["row_count"]),
                ]
            )
        )
    return 0


def _run_latency(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT latency_ms, tokens_in, tokens_out
        FROM candidate_summary
        WHERE eval_revision = 'latest'
        ORDER BY latency_ms ASC
        """
    ).fetchall()
    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    tokens_in = [row["tokens_in"] for row in rows if row["tokens_in"] is not None]
    tokens_out = [row["tokens_out"] for row in rows if row["tokens_out"] is not None]
    print(
        " ".join(
            [
                f"count={len(rows)}",
                f"latency_ms(avg={_format_stat(latencies)},p50={_percentile(latencies, 50)},p95={_percentile(latencies, 95)})",
                f"tokens_in(avg={_format_stat(tokens_in)})",
                f"tokens_out(avg={_format_stat(tokens_out)})",
            ]
        )
    )
    return 0


def _format_stat(values: list[int | float]) -> str:
    if not values:
        return "n/a"
    return f"{mean(values):.2f}"


def _percentile(values: list[int | float], percentile: int) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    index = int(round((percentile / 100) * (len(ordered) - 1)))
    return f"{ordered[index]:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())

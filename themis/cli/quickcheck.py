"""Small CLI for reading projection summaries from a Themis SQLite database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any

from themis.overlays import OverlaySelection


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def add_quickcheck_arguments(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Attach quickcheck subcommands to an argparse subparser collection."""

    failures = subparsers.add_parser("failures")
    failures.add_argument("--db", required=True)
    failures.add_argument("--limit", type=int, default=10)
    failures.add_argument("--transform-hash")
    failures.add_argument("--evaluation-hash")

    scores = subparsers.add_parser("scores")
    scores.add_argument("--db", required=True)
    scores.add_argument("--metric")
    scores.add_argument("--task")
    scores.add_argument("--evaluation-hash")

    latency = subparsers.add_parser("latency")
    latency.add_argument("--db", required=True)
    latency.add_argument("--transform-hash")
    latency.add_argument("--evaluation-hash")


def configure_quickcheck_parser(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Configure one parser to serve the quickcheck CLI."""

    subparsers = parser.add_subparsers(dest="command", required=True)
    add_quickcheck_arguments(subparsers)
    parser.set_defaults(handler=run_with_args)
    return parser


def build_parser(*, prog: str = "themis-quickcheck") -> argparse.ArgumentParser:
    """Build the quickcheck CLI parser."""

    parser = argparse.ArgumentParser(prog=prog)
    return configure_quickcheck_parser(parser)


def add_quickcheck_subparser(
    subparsers: argparse._SubParsersAction[Any],
) -> argparse.ArgumentParser:
    """Add the quickcheck command to a parent CLI."""

    parser = subparsers.add_parser("quickcheck")
    return configure_quickcheck_parser(parser)


def run_with_args(args: argparse.Namespace) -> int:
    """Execute the parsed quickcheck command."""

    db_path = Path(args.db)
    with _connect(str(db_path)) as conn:
        if args.command == "failures":
            return _run_failures(
                conn,
                limit=args.limit,
                transform_hash=args.transform_hash,
                evaluation_hash=args.evaluation_hash,
            )
        if args.command == "scores":
            return _run_scores(
                conn,
                metric_id=args.metric,
                task_id=args.task,
                evaluation_hash=args.evaluation_hash,
            )
        if args.command == "latency":
            return _run_latency(
                conn,
                transform_hash=args.transform_hash,
                evaluation_hash=args.evaluation_hash,
            )
    raise ValueError(f"Unknown quickcheck command '{args.command}'.")


def main(argv: list[str] | None = None) -> int:
    """Run the quickcheck CLI and dispatch to the selected summary command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_with_args(args)


def _run_failures(
    conn: sqlite3.Connection,
    *,
    limit: int,
    transform_hash: str | None = None,
    evaluation_hash: str | None = None,
) -> int:
    overlay_selection = OverlaySelection(
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )
    rows = conn.execute(
        """
        SELECT trial_hash, overlay_key, model_id, task_id, item_id, error_fingerprint, error_preview
        FROM trial_summary
        WHERE status = 'error' AND overlay_key = ?
        ORDER BY COALESCE(ended_at, updated_at, created_at) DESC, trial_hash ASC
        LIMIT ?
        """,
        (overlay_selection.overlay_key, limit),
    ).fetchall()
    for row in rows:
        print(
            "\t".join(
                [
                    row["trial_hash"],
                    row["overlay_key"] or "",
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
    conn: sqlite3.Connection,
    *,
    metric_id: str | None,
    task_id: str | None,
    evaluation_hash: str | None,
) -> int:
    clauses = ["metric_scores.overlay_key = candidate_summary.overlay_key"]
    params: list[object] = []
    if evaluation_hash is not None:
        clauses.append("candidate_summary.overlay_key = ?")
        params.append(OverlaySelection(evaluation_hash=evaluation_hash).overlay_key)
    else:
        clauses.append("candidate_summary.overlay_key LIKE 'ev:%'")
    if metric_id is not None:
        clauses.append("metric_scores.metric_id = ?")
        params.append(metric_id)
    if task_id is not None:
        clauses.append("trial_summary.task_id = ?")
        params.append(task_id)
    where_clause = f"WHERE {' AND '.join(clauses)}"
    rows = conn.execute(
        f"""
        SELECT candidate_summary.overlay_key, trial_summary.model_id, trial_summary.task_id, metric_scores.metric_id, AVG(metric_scores.score) AS avg_score, COUNT(*) AS row_count
        FROM metric_scores
        JOIN candidate_summary
          ON candidate_summary.candidate_id = metric_scores.candidate_id
         AND candidate_summary.overlay_key = metric_scores.overlay_key
        JOIN trial_summary
          ON trial_summary.trial_hash = candidate_summary.trial_hash
         AND trial_summary.overlay_key = candidate_summary.overlay_key
        {where_clause}
        GROUP BY candidate_summary.overlay_key, trial_summary.model_id, trial_summary.task_id, metric_scores.metric_id
        ORDER BY candidate_summary.overlay_key ASC, trial_summary.model_id ASC, metric_scores.metric_id ASC
        """,
        params,
    ).fetchall()
    for row in rows:
        print(
            "\t".join(
                [
                    row["overlay_key"] or "",
                    row["model_id"] or "",
                    row["task_id"] or "",
                    row["metric_id"] or "",
                    f"{row['avg_score']:.4f}",
                    str(row["row_count"]),
                ]
            )
        )
    return 0


def _run_latency(
    conn: sqlite3.Connection,
    *,
    transform_hash: str | None = None,
    evaluation_hash: str | None = None,
) -> int:
    overlay_selection = OverlaySelection(
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )
    rows = conn.execute(
        """
        SELECT latency_ms, tokens_in, tokens_out
        FROM candidate_summary
        WHERE overlay_key = ?
        ORDER BY latency_ms ASC
        """,
        (overlay_selection.overlay_key,),
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

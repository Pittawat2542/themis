"""Small CLI for reading projection summaries from a Themis SQLite database."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from statistics import mean

from cyclopts import App

from themis.cli._common import invoke_app
from themis.overlays import OverlaySelection

logger = logging.getLogger(__name__)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def build_app(*, standalone: bool = False) -> App:
    """Build the quickcheck Cyclopts app."""

    app = App(
        name="themis-quickcheck" if standalone else "quickcheck",
        help="Read stored SQLite projections without importing benchmark code.",
    )

    @app.command(name="failures")
    def failures(
        db: str,
        limit: int = 10,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> int:
        with _connect(str(Path(db))) as conn:
            return _run_failures(
                conn,
                limit=limit,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )

    @app.command(name="scores")
    def scores(
        db: str,
        metric: str | None = None,
        slice: str | None = None,
        dimension: list[str] | None = None,
        evaluation_hash: str | None = None,
    ) -> int:
        with _connect(str(Path(db))) as conn:
            return _run_scores(
                conn,
                metric_id=metric,
                slice_id=slice,
                dimension_filters=dimension or [],
                evaluation_hash=evaluation_hash,
            )

    @app.command(name="latency")
    def latency(
        db: str,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> int:
        with _connect(str(Path(db))) as conn:
            return _run_latency(
                conn,
                transform_hash=transform_hash,
                evaluation_hash=evaluation_hash,
            )

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the quickcheck CLI and dispatch to the selected summary command."""

    return invoke_app(build_app(standalone=True), argv)


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
        SELECT trial_hash, overlay_key, model_id, slice_id, item_id, error_fingerprint, error_preview
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
                    row["slice_id"] or "",
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
    slice_id: str | None,
    dimension_filters: list[str],
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
    if slice_id is not None:
        clauses.append("trial_summary.slice_id = ?")
        params.append(slice_id)
    where_clause = f"WHERE {' AND '.join(clauses)}"
    rows = conn.execute(
        f"""
        SELECT candidate_summary.overlay_key, trial_summary.model_id, trial_summary.slice_id, trial_summary.dimensions_json, metric_scores.metric_id, metric_scores.score
        FROM metric_scores
        JOIN candidate_summary
          ON candidate_summary.candidate_id = metric_scores.candidate_id
         AND candidate_summary.overlay_key = metric_scores.overlay_key
        JOIN trial_summary
          ON trial_summary.trial_hash = candidate_summary.trial_hash
         AND trial_summary.overlay_key = candidate_summary.overlay_key
        {where_clause}
        ORDER BY candidate_summary.overlay_key ASC, trial_summary.model_id ASC, trial_summary.slice_id ASC, metric_scores.metric_id ASC
        """,
        params,
    ).fetchall()
    parsed_dimension_filters = _parse_dimension_filters(dimension_filters)
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        dimensions = _load_dimensions(row["dimensions_json"])
        if not _dimensions_match(dimensions, parsed_dimension_filters):
            continue
        key = (
            row["overlay_key"] or "",
            row["model_id"] or "",
            row["slice_id"] or "",
            row["metric_id"] or "",
        )
        grouped.setdefault(key, []).append(float(row["score"]))
    for (
        overlay_key,
        model_id,
        resolved_slice_id,
        resolved_metric_id,
    ), scores in grouped.items():
        print(
            "\t".join(
                [
                    overlay_key,
                    model_id,
                    resolved_slice_id,
                    resolved_metric_id,
                    f"{mean(scores):.4f}",
                    str(len(scores)),
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
    clauses = []
    params: list[object] = []
    if overlay_selection.overlay_key is not None:
        clauses.append("candidate_summary.overlay_key = ?")
        params.append(overlay_selection.overlay_key)
    else:
        clauses.append("candidate_summary.overlay_key = 'gen'")
    where_clause = f"WHERE {' AND '.join(clauses)}"
    rows = conn.execute(
        f"""
        SELECT latency_ms, tokens_in, tokens_out
        FROM candidate_summary
        {where_clause}
        ORDER BY latency_ms ASC
        """,
        params,
    ).fetchall()
    latencies = [
        float(row["latency_ms"]) for row in rows if row["latency_ms"] is not None
    ]
    tokens_in = [
        float(row["tokens_in"]) for row in rows if row["tokens_in"] is not None
    ]
    tokens_out = [
        float(row["tokens_out"]) for row in rows if row["tokens_out"] is not None
    ]
    if not latencies:
        print(
            " ".join(
                [
                    "count=0",
                    "latency_ms(avg=n/a,p50=n/a,p95=n/a)",
                    "tokens_in(avg=n/a)",
                    "tokens_out(avg=n/a)",
                ]
            )
        )
        return 0
    print(
        " ".join(
            [
                f"count={len(rows)}",
                (
                    "latency_ms("
                    f"avg={_format_stat(latencies)},"
                    f"p50={_percentile(latencies, 50)},"
                    f"p95={_percentile(latencies, 95)})"
                ),
                f"tokens_in(avg={_format_stat(tokens_in)})",
                f"tokens_out(avg={_format_stat(tokens_out)})",
            ]
        )
    )
    return 0


def _parse_dimension_filters(filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in filters:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            logger.warning(
                "Ignoring malformed dimension filter %r; expected key=value.", item
            )
            continue
        parsed[key] = value
    return parsed


def _load_dimensions(payload: str | None) -> dict[str, str]:
    if not payload:
        return {}
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if value is not None and not isinstance(value, (dict, list))
    }


def _dimensions_match(
    dimensions: dict[str, str],
    expected: dict[str, str],
) -> bool:
    for key, value in expected.items():
        if dimensions.get(key) != value:
            return False
    return True


def _format_stat(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{mean(values):.2f}"


def _percentile(values: list[float], percentile: int) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    index = int(round((percentile / 100) * (len(ordered) - 1)))
    return f"{ordered[index]:.2f}"

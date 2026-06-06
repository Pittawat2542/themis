"""Projection-backed reporting and export helpers."""

from __future__ import annotations

import csv
import json
from io import StringIO

from typing import cast

from themis.core.base import JSONValue
from themis.core.inspection import get_execution_state, get_run_snapshot
from themis.core.models import MetricResult
from themis.core.read_models import (
    BenchmarkResult,
    FailureSlice,
    FailureSliceSummary,
    ReliabilitySummary,
)
from themis.core.reliability import calibration_error
from themis.core.snapshot import RunSnapshot
from themis.core.stats import StatsEngine, StatsSummary
from themis.core.store import RunStore


def snapshot_report(
    snapshot: RunSnapshot, run_metadata: dict[str, JSONValue] | None = None
) -> dict[str, JSONValue]:
    """Return a JSON-serializable summary for a compiled snapshot."""

    return {
        "run_id": snapshot.run_id,
        "identity": snapshot.identity.model_dump(mode="json"),
        "provenance": snapshot.provenance.model_dump(mode="json"),
        "component_refs": snapshot.component_refs.model_dump(mode="json"),
        "run_metadata": dict(run_metadata or {}),
    }


class Reporter:
    """Export persisted run projections in JSON, Markdown, CSV, or LaTeX."""

    def __init__(self, store: RunStore) -> None:
        self.store = store

    def export_json(self, run_id: str) -> str:
        """Export all major persisted projections for a run as formatted JSON."""

        payload = {
            "snapshot": get_run_snapshot(self.store, run_id).model_dump(mode="json"),
            "execution_state": get_execution_state(self.store, run_id).model_dump(
                mode="json"
            ),
            "run_result": self._projection(run_id, "run_result"),
            "benchmark_result": self._projection(run_id, "benchmark_result"),
            "stats_summary": self.summary(run_id).model_dump(mode="json"),
            "timeline_view": self._projection(run_id, "timeline_view"),
            "trace_view": self._projection(run_id, "trace_view"),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def summary(self, run_id: str) -> StatsSummary:
        """Return a typed statistical summary for a persisted run."""

        benchmark_result = BenchmarkResult.model_validate(
            self._projection(run_id, "benchmark_result")
        )
        return StatsEngine().summarize(benchmark_result)

    def export_markdown(self, run_id: str) -> str:
        """Export a summary-first Markdown report for a persisted run."""

        run_result = self._projection(run_id, "run_result")
        summary = self.summary(run_id)
        progress = _require_mapping(
            run_result.get("progress"), name="run_result.progress"
        )
        lines = [
            "# Run Report",
            "",
            f"- run_id: {run_result['run_id']}",
            f"- status: {run_result['status']}",
            f"- total_cases: {progress['total_cases']}",
            f"- completed_cases: {progress['completed_cases']}",
            f"- failed_cases: {progress['failed_cases']}",
            "",
            "## Metrics",
            "",
            "| metric_id | count | mean | min | max | ci_lower | ci_upper |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for metric in summary.metrics:
            lines.append(
                " | ".join(
                    [
                        f"| {_markdown_cell(metric.metric_id)}",
                        str(metric.count),
                        str(metric.mean),
                        str(metric.min),
                        str(metric.max),
                        str(metric.ci_lower),
                        f"{metric.ci_upper} |",
                    ]
                )
            )
        failures = [row for row in self.score_rows(run_id) if row["outcome"] == "error"]
        if failures:
            lines.extend(
                [
                    "",
                    "## Failures",
                    "",
                    "| case_id | metric_id | failure_category | error_message |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for row in failures:
                lines.append(
                    " | ".join(
                        [
                            f"| {_markdown_cell(row['case_id'])}",
                            _markdown_cell(row["metric_id"]),
                            _markdown_cell(row["failure_category"]),
                            f"{_markdown_cell(row['error_message'])} |",
                        ]
                    )
                )
        return "\n".join(lines) + "\n"

    def export_csv(self, run_id: str) -> str:
        """Export benchmark metric summaries as CSV."""

        buffer = StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "metric_id",
                "count",
                "mean",
                "min",
                "max",
                "ci_lower",
                "ci_upper",
            ],
        )
        writer.writeheader()
        writer.writerows(
            metric.model_dump(mode="json") for metric in self.summary(run_id).metrics
        )
        return buffer.getvalue()

    def export_latex(self, run_id: str) -> str:
        """Export benchmark metric summaries as a compact LaTeX table."""

        lines = [
            r"\begin{tabular}{lrrrrrr}",
            r"metric\_id & count & mean & min & max & ci\_lower & ci\_upper \\",
            r"\hline",
        ]
        for metric in self.summary(run_id).metrics:
            lines.append(
                " & ".join(
                    [
                        _latex_cell(metric.metric_id),
                        _latex_cell(metric.count),
                        _latex_cell(metric.mean),
                        _latex_cell(metric.min),
                        _latex_cell(metric.max),
                        _latex_cell(metric.ci_lower),
                        _latex_cell(metric.ci_upper),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\end{tabular}")
        return "\n".join(lines) + "\n"

    def score_rows(self, run_id: str) -> list[dict[str, JSONValue]]:
        """Return benchmark metric result rows in a normalized table structure."""

        benchmark_result = self._projection(run_id, "benchmark_result")
        score_rows = _require_rows(
            benchmark_result.get("score_rows"), name="benchmark_result.score_rows"
        )
        return [
            {
                "case_id": row["case_id"],
                "dataset_id": row.get("dataset_id"),
                "case_key": row.get("case_key"),
                "metric_id": row["metric_id"],
                "result_type": row.get("result_type"),
                "outcome": row["outcome"],
                "value": row["value"],
                "confidence": row.get("confidence"),
                "dimensions": row.get("dimensions", {}),
                "labels": row.get("labels", {}),
                "candidate_id": row["candidate_id"],
                "failure_category": row.get("failure_category"),
                "error_message": row.get("error_message"),
                "metadata": row.get("metadata", {}),
            }
            for row in score_rows
        ]

    def failure_slices(self, run_id: str) -> FailureSliceSummary:
        """Return deterministic failure slices grouped from benchmark error rows."""

        grouped: dict[tuple[str, str], set[str]] = {}
        for row in self.score_rows(run_id):
            if row["outcome"] != "error":
                continue
            case_key = _case_key_for_row(row)
            _add_failure_slice(
                grouped,
                dimension="category",
                value=row.get("failure_category"),
                case_key=case_key,
            )
            _add_failure_slice(
                grouped,
                dimension="dataset",
                value=row.get("dataset_id"),
                case_key=case_key,
            )
            _add_failure_slice(
                grouped,
                dimension="metric",
                value=row.get("metric_id"),
                case_key=case_key,
            )
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    _add_failure_slice(
                        grouped,
                        dimension=f"metadata.{key}",
                        value=value,
                        case_key=case_key,
                    )

        slices = [
            FailureSlice(
                dimension=dimension,
                value=value,
                count=len(case_keys),
                case_keys=sorted(case_keys),
            )
            for (dimension, value), case_keys in sorted(grouped.items())
        ]
        return FailureSliceSummary(run_id=run_id, slices=slices)

    def reliability(self, run_id: str) -> ReliabilitySummary:
        """Return calibration-style reliability summaries for scored rows."""

        metric_rows: dict[str, list[MetricResult]] = {}
        for row in self.score_rows(run_id):
            if row["outcome"] == "error":
                continue
            if row.get("value") is None or row.get("confidence") is None:
                continue
            metric_id = str(row["metric_id"])
            metric_rows.setdefault(metric_id, []).append(
                MetricResult(
                    metric_id=metric_id,
                    value=cast(float, row["value"]),
                    confidence=cast(float, row["confidence"]),
                )
            )
        return ReliabilitySummary(
            run_id=run_id,
            metrics=[
                calibration_error(metric_id, results)
                for metric_id, results in sorted(metric_rows.items())
            ],
        )

    def _projection(self, run_id: str, projection_name: str) -> dict[str, JSONValue]:
        """Load a named stored projection and validate its JSON object shape."""

        projection = self.store.get_projection(run_id, projection_name)
        if projection is None or not isinstance(projection, dict):
            raise ValueError(
                f"Projection not found: {projection_name} for run_id={run_id}"
            )
        return projection


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_cell(value: JSONValue) -> str:
    if value is None:
        return ""
    rendered = str(value)
    return "".join(_LATEX_ESCAPES.get(char, char) for char in rendered)


def _markdown_cell(value: JSONValue) -> str:
    if value is None:
        return ""
    return str(value).replace("|", r"\|").replace("\n", " ")


def _require_mapping(value: JSONValue | None, *, name: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object projection value for {name}")
    return value


def _require_rows(value: JSONValue | None, *, name: str) -> list[dict[str, JSONValue]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected row list for {name}")
    return cast(list[dict[str, JSONValue]], value)


def _case_key_for_row(row: dict[str, JSONValue]) -> str:
    case_key = row.get("case_key")
    if isinstance(case_key, str) and case_key:
        return case_key
    dataset_id = row.get("dataset_id")
    case_id = row["case_id"]
    if isinstance(dataset_id, str) and dataset_id:
        return f"{len(dataset_id)}:{dataset_id}:{case_id}"
    return str(case_id)


def _add_failure_slice(
    grouped: dict[tuple[str, str], set[str]],
    *,
    dimension: str,
    value: JSONValue | None,
    case_key: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, (str, int, float, bool)):
        return
    grouped.setdefault((dimension, str(value)), set()).add(case_key)

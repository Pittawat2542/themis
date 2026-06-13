"""Projection-backed reporting and export helpers."""

from __future__ import annotations

import csv
import json
from io import StringIO

from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

from themis.core.base import JSONValue
from themis.core.inspection import get_execution_state, get_run_snapshot
from themis.core.models import MetricResult
from themis.core.read_models import (
    BenchmarkResult,
    FailureSlice,
    FailureSliceSummary,
    PairwiseComparisonReport,
    PairwiseMetricClaim,
    RegressionFinding,
    RegressionSummary,
    ReliabilitySummary,
    SuiteCoverageSummary,
    TrendPoint,
    TrendView,
)
from themis.core.registry import RegressionPolicy, RunQuery, RunRecord
from themis.core.reliability import calibration_error
from themis.core.snapshot import RunSnapshot
from themis.core.stats import StatsEngine, StatsSummary
from themis.core.store import RunStore


@runtime_checkable
class ReporterProtocol(Protocol):
    """Protocol for replaceable reporters over persisted run evidence."""

    def export_json(self, run_id: str) -> str: ...


ReporterBuilder = Callable[[RunStore], ReporterProtocol]


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

    def compare_runs(
        self, baseline_run_id: str, candidate_run_id: str
    ) -> PairwiseComparisonReport:
        """Return a score claim comparing two persisted runs."""

        baseline = BenchmarkResult.model_validate(
            self._projection(baseline_run_id, "benchmark_result")
        )
        candidate = BenchmarkResult.model_validate(
            self._projection(candidate_run_id, "benchmark_result")
        )
        comparison = StatsEngine().compare(baseline, candidate)
        missing_baseline, missing_candidate = _pairwise_missing_counts(
            baseline, candidate
        )
        matched_pair_count = sum(metric.pairs for metric in comparison.metrics)
        return PairwiseComparisonReport(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            evidence_run_ids=[baseline_run_id, candidate_run_id],
            metrics=[
                PairwiseMetricClaim.model_validate(metric.model_dump(mode="json"))
                for metric in comparison.metrics
            ],
            matched_pair_count=matched_pair_count,
            missing_baseline_rows=missing_baseline,
            missing_candidate_rows=missing_candidate,
            dropped_rows=missing_baseline + missing_candidate,
        )

    def compare_latest(
        self, *, baseline_label: str, candidate_label: str | None = None
    ) -> PairwiseComparisonReport:
        """Compare latest persisted baseline and candidate labels."""

        baseline = _latest_record(
            self.store.query_runs(RunQuery(baseline_label=baseline_label))
        )
        candidate_records = (
            self.store.query_runs(RunQuery(baseline_label=candidate_label))
            if candidate_label is not None
            else self.store.query_runs()
        )
        candidate = _latest_record(
            [record for record in candidate_records if record.run_id != baseline.run_id]
        )
        return self.compare_runs(baseline.run_id, candidate.run_id)

    def suite_coverage(self, suite_id: str) -> SuiteCoverageSummary:
        """Return persisted run coverage for a suite tag."""

        records = self.store.query_runs(RunQuery(tags=[f"suite:{suite_id}"]))
        benchmark_ids = sorted(
            {
                tag.removeprefix("benchmark:")
                for record in records
                for tag in record.tags
                if tag.startswith("benchmark:")
            }
        )
        return SuiteCoverageSummary(
            suite_id=suite_id,
            total_runs=len(records),
            covered_count=len(records),
            covered_benchmark_ids=benchmark_ids,
            run_ids=[record.run_id for record in records],
        )

    def suite_summary(self, suite_id: str) -> SuiteCoverageSummary:
        """Alias for suite coverage until suite aggregation adds rollups."""

        return self.suite_coverage(suite_id)

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

    def trends(self, *, metric_id: str) -> TrendView:
        """Return metric means across stored runs ordered by registry creation time."""

        points: list[TrendPoint] = []
        for record in self.store.query_runs(RunQuery(metric_id=metric_id)):
            projection = self.store.get_projection(record.run_id, "benchmark_result")
            if not isinstance(projection, dict):
                continue
            metric_means = projection.get("metric_means", {})
            if not isinstance(metric_means, dict):
                continue
            value = metric_means.get(metric_id)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            points.append(
                TrendPoint(
                    run_id=record.run_id,
                    metric_id=metric_id,
                    value=float(value),
                    baseline_label=record.baseline_label,
                    created_at=record.created_at,
                )
            )
        return TrendView(metric_id=metric_id, points=points)

    def regressions(
        self, candidate_run_id: str, policy: RegressionPolicy
    ) -> RegressionSummary:
        """Compare a candidate run against the run pinned by a baseline label."""

        baseline_records = self.store.query_runs(
            RunQuery(baseline_label=policy.baseline_label)
        )
        if not baseline_records:
            raise ValueError(f"Unknown baseline label: {policy.baseline_label}")
        baseline_record = baseline_records[-1]
        baseline_means = _metric_means(
            self._projection(baseline_record.run_id, "benchmark_result")
        )
        candidate_means = _metric_means(
            self._projection(candidate_run_id, "benchmark_result")
        )
        findings: list[RegressionFinding] = []
        for metric_id, threshold in sorted(policy.metric_thresholds.items()):
            baseline_value = baseline_means.get(metric_id)
            candidate_value = candidate_means.get(metric_id)
            if baseline_value is None or candidate_value is None:
                continue
            delta = round(candidate_value - baseline_value, 12)
            findings.append(
                RegressionFinding(
                    metric_id=metric_id,
                    baseline_run_id=baseline_record.run_id,
                    candidate_run_id=candidate_run_id,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    delta=delta,
                    threshold=threshold,
                    regressed=delta < threshold,
                )
            )
        return RegressionSummary(
            candidate_run_id=candidate_run_id,
            baseline_label=policy.baseline_label,
            findings=findings,
        )

    def _projection(self, run_id: str, projection_name: str) -> dict[str, JSONValue]:
        """Load a named stored projection and validate its JSON object shape."""

        projection = self.store.get_projection(run_id, projection_name)
        if projection is None or not isinstance(projection, dict):
            raise ValueError(
                f"Projection not found: {projection_name} for run_id={run_id}"
            )
        return projection


_REPORTER_BUILDERS: dict[str, ReporterBuilder] = {
    "default": Reporter,
}


def register_reporter(name: str, builder: ReporterBuilder) -> None:
    """Register a reporter builder for `create_reporter`."""

    _REPORTER_BUILDERS[name] = builder


def available_reporters() -> list[str]:
    """Return the registered reporter names."""

    return sorted(_REPORTER_BUILDERS)


def create_reporter(name: str, store: RunStore) -> ReporterProtocol:
    """Instantiate a reporter by registered name."""

    try:
        builder = _REPORTER_BUILDERS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported reporter: {name}") from exc
    return builder(store)


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


def _pairwise_missing_counts(
    baseline: BenchmarkResult, candidate: BenchmarkResult
) -> tuple[int, int]:
    baseline_keys = {
        (_case_key_for_benchmark_row(row), row.metric_id)
        for row in baseline.score_rows
        if row.value is not None and row.outcome != "error"
    }
    candidate_keys = {
        (_case_key_for_benchmark_row(row), row.metric_id)
        for row in candidate.score_rows
        if row.value is not None and row.outcome != "error"
    }
    return (
        len(candidate_keys - baseline_keys),
        len(baseline_keys - candidate_keys),
    )


def _case_key_for_benchmark_row(row: object) -> str:
    case_key = getattr(row, "case_key", None)
    if isinstance(case_key, str) and case_key:
        return case_key
    dataset_id = getattr(row, "dataset_id", None)
    case_id = getattr(row, "case_id")
    if isinstance(dataset_id, str) and dataset_id:
        return f"{len(dataset_id)}:{dataset_id}:{case_id}"
    return str(case_id)


def _latest_record(records: list[RunRecord]) -> RunRecord:
    if not records:
        raise ValueError("No matching runs found")
    return sorted(records, key=lambda record: record.created_at)[-1]


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


def _metric_means(projection: dict[str, JSONValue]) -> dict[str, float]:
    value = projection.get("metric_means", {})
    if not isinstance(value, dict):
        return {}
    return {
        str(metric_id): float(metric_value)
        for metric_id, metric_value in value.items()
        if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool)
    }

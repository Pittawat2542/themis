"""Projection-backed reporting and export helpers."""

from __future__ import annotations

import csv
import json
from io import StringIO

from typing import cast

from themis.core.base import JSONValue
from themis.core.inspection import get_execution_state, get_run_snapshot
from themis.core.snapshot import RunSnapshot
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
            "timeline_view": self._projection(run_id, "timeline_view"),
            "trace_view": self._projection(run_id, "trace_view"),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def export_markdown(self, run_id: str) -> str:
        """Export a human-readable Markdown summary for a persisted run."""

        run_result = self._projection(run_id, "run_result")
        benchmark_result = self._projection(run_id, "benchmark_result")
        progress = _require_mapping(
            run_result.get("progress"), name="run_result.progress"
        )
        score_rows = _require_rows(
            benchmark_result.get("score_rows"), name="benchmark_result.score_rows"
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
        ]
        for row in score_rows:
            lines.append(
                f"- case={row['case_id']} metric={row['metric_id']} outcome={row['outcome']} value={row['value']} candidate={row['candidate_id']} error_category={row.get('error_category')} error_message={row.get('error_message')}"
            )
        return "\n".join(lines) + "\n"

    def export_csv(self, run_id: str) -> str:
        """Export benchmark score rows as CSV."""

        buffer = StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "case_id",
                "metric_id",
                "outcome",
                "value",
                "candidate_id",
                "error_category",
                "error_message",
                "details",
            ],
        )
        writer.writeheader()
        writer.writerows(self.export_score_table(run_id))
        return buffer.getvalue()

    def export_latex(self, run_id: str) -> str:
        """Export benchmark score rows as a compact LaTeX table."""

        lines = [
            r"\begin{tabular}{llllllll}",
            r"case\_id & metric\_id & outcome & value & candidate\_id & error\_category & error\_message & details \\",
            r"\hline",
        ]
        for row in self.export_score_table(run_id):
            lines.append(
                " & ".join(
                    [
                        _latex_cell(row["case_id"]),
                        _latex_cell(row["metric_id"]),
                        _latex_cell(row["outcome"]),
                        _latex_cell(row["value"]),
                        _latex_cell(row["candidate_id"]),
                        _latex_cell(row["error_category"]),
                        _latex_cell(row["error_message"]),
                        _latex_cell(row["details"]),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\end{tabular}")
        return "\n".join(lines) + "\n"

    def export_score_table(self, run_id: str) -> list[dict[str, JSONValue]]:
        """Return benchmark score rows in a normalized table structure."""

        benchmark_result = self._projection(run_id, "benchmark_result")
        score_rows = _require_rows(
            benchmark_result.get("score_rows"), name="benchmark_result.score_rows"
        )
        return [
            {
                "case_id": row["case_id"],
                "metric_id": row["metric_id"],
                "outcome": row["outcome"],
                "value": row["value"],
                "candidate_id": row["candidate_id"],
                "error_category": row.get("error_category"),
                "error_message": row.get("error_message"),
                "details": row.get("details", {}),
            }
            for row in score_rows
        ]

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


def _require_mapping(value: JSONValue | None, *, name: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object projection value for {name}")
    return value


def _require_rows(value: JSONValue | None, *, name: str) -> list[dict[str, JSONValue]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"Expected row list for {name}")
    return cast(list[dict[str, JSONValue]], value)

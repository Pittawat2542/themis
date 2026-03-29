"""Projection-backed reporting and export helpers."""

from __future__ import annotations

import csv
import json
from io import StringIO

from themis.core.base import JSONValue
from themis.core.store import RunStore


def snapshot_report(snapshot, run_metadata: dict[str, JSONValue] | None = None) -> dict[str, JSONValue]:
    return {
        "run_id": snapshot.run_id,
        "identity": snapshot.identity.model_dump(mode="json"),
        "provenance": snapshot.provenance.model_dump(mode="json"),
        "component_refs": snapshot.component_refs.model_dump(mode="json"),
        "run_metadata": dict(run_metadata or {}),
    }


class Reporter:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def export_json(self, run_id: str) -> str:
        payload = {
            "run_result": self._projection(run_id, "run_result"),
            "benchmark_result": self._projection(run_id, "benchmark_result"),
            "timeline_view": self._projection(run_id, "timeline_view"),
            "trace_view": self._projection(run_id, "trace_view"),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def export_markdown(self, run_id: str) -> str:
        run_result = self._projection(run_id, "run_result")
        benchmark_result = self._projection(run_id, "benchmark_result")
        lines = [
            "# Run Report",
            "",
            f"- run_id: {run_result['run_id']}",
            f"- status: {run_result['status']}",
            f"- total_cases: {run_result['progress']['total_cases']}",
            f"- completed_cases: {run_result['progress']['completed_cases']}",
            f"- failed_cases: {run_result['progress']['failed_cases']}",
            "",
            "## Metrics",
            "",
        ]
        for row in benchmark_result["score_rows"]:
            lines.append(
                f"- case={row['case_id']} metric={row['metric_id']} value={row['value']} candidate={row['candidate_id']}"
            )
        return "\n".join(lines) + "\n"

    def export_csv(self, run_id: str) -> str:
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["case_id", "metric_id", "value", "candidate_id"])
        writer.writeheader()
        writer.writerows(self.export_score_table(run_id))
        return buffer.getvalue()

    def export_latex(self, run_id: str) -> str:
        lines = [
            r"\begin{tabular}{llll}",
            r"case\_id & metric\_id & value & candidate\_id \\",
            r"\hline",
        ]
        for row in self.export_score_table(run_id):
            lines.append(
                f"{row['case_id']} & {row['metric_id']} & {row['value']} & {row['candidate_id']} \\\\"
            )
        lines.append(r"\end{tabular}")
        return "\n".join(lines) + "\n"

    def export_score_table(self, run_id: str) -> list[dict[str, JSONValue]]:
        benchmark_result = self._projection(run_id, "benchmark_result")
        return [
            {
                "case_id": row["case_id"],
                "metric_id": row["metric_id"],
                "value": row["value"],
                "candidate_id": row["candidate_id"],
            }
            for row in benchmark_result["score_rows"]
        ]

    def _projection(self, run_id: str, projection_name: str) -> dict[str, JSONValue]:
        projection = self.store.get_projection(run_id, projection_name)
        if projection is None or not isinstance(projection, dict):
            raise ValueError(f"Projection not found: {projection_name} for run_id={run_id}")
        return projection

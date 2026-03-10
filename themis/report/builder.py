"""Report assembly from projected score rows and optional hydrated trials."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

import themis

from themis._optional import import_optional
from themis.contracts.protocols import ReportExporter
from themis.records.report import EvaluationReport, ReportTable, ReportMetadata
from themis.records.trial import TrialRecord
from themis.report.exporters import CsvExporter, MarkdownExporter, LatexExporter
from themis.runtime.comparison import build_comparison_table
from themis.storage.events import ScoreRow, TrialSummaryRow
from themis.types.json_types import JSONDict, JSONList

if TYPE_CHECKING:
    from themis.stats.stats_engine import StatsEngine


class ReportBuilder:
    """Builds aggregate tables, comparisons, and metadata from trial projections."""

    def __init__(
        self,
        trials: list[TrialRecord],
        stats_engine: StatsEngine | None = None,
        trial_summaries: list[TrialSummaryRow] | None = None,
        score_rows: list[ScoreRow] | None = None,
        eval_revision: str | None = None,
    ) -> None:
        self.trials = trials
        self.stats_engine = stats_engine
        self.trial_summaries = list(trial_summaries or [])
        self.score_rows = list(score_rows or [])
        self.eval_revision = eval_revision
        self.report: EvaluationReport | None = None

    def _resolved_trial_summaries(self) -> list[TrialSummaryRow]:
        if self.trial_summaries:
            return list(self.trial_summaries)

        return [
            TrialSummaryRow(
                trial_hash=trial.spec_hash,
                model_id=trial.trial_spec.model.model_id if trial.trial_spec else None,
                task_id=trial.trial_spec.task.task_id if trial.trial_spec else None,
                item_id=trial.trial_spec.item_id if trial.trial_spec else None,
                status=trial.status,
            )
            for trial in self.trials
        ]

    def _extract_metric_rows(self):
        """Flattens all trial candidates into a DataFrame for the StatsEngine."""
        pd = import_optional("pandas", extra="stats")
        rows: list[JSONDict] = []
        trial_metadata = {
            trial.trial_hash: {
                "model_id": trial.model_id or "unknown",
                "task_id": trial.task_id or "unknown",
                "item_id": trial.item_id or "unknown",
            }
            for trial in self._resolved_trial_summaries()
        }

        if self.score_rows:
            for row in self.score_rows:
                metadata = trial_metadata.get(
                    row.trial_hash,
                    {"model_id": "unknown", "task_id": "unknown", "item_id": "unknown"},
                )
                rows.append(
                    {
                        "trial_hash": row.trial_hash,
                        "cand_hash": row.candidate_id,
                        "metric_id": row.metric_id,
                        "metric_value": row.score,
                        "model_id": metadata["model_id"],
                        "task_id": metadata["task_id"],
                        "item_id": metadata["item_id"],
                    }
                )
        else:
            for trial in self.trials:
                metadata = trial_metadata[trial.spec_hash]
                for cand in trial.candidates:
                    if cand.evaluation:
                        for metric_id, val in cand.evaluation.aggregate_scores.items():
                            rows.append(
                                {
                                    "trial_hash": trial.spec_hash,
                                    "cand_hash": cand.spec_hash,
                                    "metric_id": metric_id,
                                    "metric_value": val,
                                    "model_id": metadata["model_id"],
                                    "task_id": metadata["task_id"],
                                    "item_id": metadata["item_id"],
                                }
                            )
        return pd.DataFrame(rows)

    def _comparison_score_rows(self) -> list[ScoreRow]:
        if self.score_rows:
            return list(self.score_rows)

        rows: list[ScoreRow] = []
        for trial in self.trials:
            for candidate in trial.candidates:
                if candidate.evaluation is None:
                    continue
                for metric_score in candidate.evaluation.metric_scores:
                    rows.append(
                        ScoreRow(
                            trial_hash=trial.spec_hash,
                            candidate_id=candidate.spec_hash,
                            metric_id=metric_score.metric_id,
                            score=metric_score.value,
                            details=metric_score.details,
                        )
                    )
        return rows

    def _get_stats_engine(self) -> StatsEngine:
        if self.stats_engine is not None:
            return self.stats_engine

        stats_module = import_optional("themis.stats.stats_engine", extra="stats")
        self.stats_engine = stats_module.StatsEngine()
        return self.stats_engine

    def build(self, *, p_value_correction: str = "none") -> EvaluationReport:
        """
        Assembles tables and metadata from projections or in-memory trials.
        """
        pd = import_optional("pandas", extra="stats")
        df = self._extract_metric_rows()
        tables: list[ReportTable] = []

        if not df.empty:
            agg = self._get_stats_engine().aggregate(
                df,
                group_by=["model_id", "task_id", "metric_id"],
            )
            main_table = ReportTable(
                spec_hash=f"main_table_{hashlib.md5(str(pd.util.hash_pandas_object(df).sum()).encode()).hexdigest()[:8]}",
                id="main_results",
                title="Aggregate Metrics",
                description="Mean and variance of recorded metrics grouped by model, task, and metric.",
                data=cast(Any, agg).reset_index().to_dict(orient="records"),
            )
            tables.append(main_table)

        comparison_table = build_comparison_table(
            self._resolved_trial_summaries(),
            self._comparison_score_rows(),
            stats_engine=self._get_stats_engine(),
            p_value_correction=p_value_correction,
        )
        if comparison_table.rows:
            tables.append(
                ReportTable(
                    spec_hash=(
                        f"paired_comparisons_"
                        f"{hashlib.md5(str(len(comparison_table.rows)).encode()).hexdigest()[:8]}"
                    ),
                    id="paired_comparisons",
                    title="Paired Comparisons",
                    description="Paired deltas with bootstrap confidence intervals and p-values.",
                    data=[row.model_dump(mode="json") for row in comparison_table.rows],
                )
            )

        resolved_summaries = self._resolved_trial_summaries()
        spec_hashes = [summary.trial_hash for summary in resolved_summaries]
        dataset_revisions = sorted(
            {
                trial.trial_spec.task.dataset.revision
                for trial in self.trials
                if trial.trial_spec is not None
                and trial.trial_spec.task.dataset.revision is not None
            }
        )
        extras: JSONDict = {
            "dataset_revisions": cast(JSONList, dataset_revisions.copy())
        }
        if self.eval_revision is not None:
            extras["eval_revision"] = self.eval_revision
        extras["provenance"] = self._summarize_provenance()

        meta = ReportMetadata(
            spec_hash=f"meta_{len(self.trials)}",
            themis_version=themis.__version__,
            spec_hashes=spec_hashes,
            extras=extras,
        )

        self.report = EvaluationReport(
            spec_hash=f"report_{len(self.trials)}",
            tables=tables,
            metadata=meta,
        )
        return self.report

    def _summarize_provenance(self) -> JSONDict:
        provenances = [
            trial.provenance for trial in self.trials if trial.provenance is not None
        ]
        if not provenances:
            return {}
        return {
            "themis_versions": cast(
                JSONList,
                sorted({provenance.themis_version for provenance in provenances}),
            ),
            "git_commits": cast(
                JSONList,
                sorted(
                    {
                        provenance.git_commit
                        for provenance in provenances
                        if provenance.git_commit is not None
                    }
                ),
            ),
            "python_versions": cast(
                JSONList,
                sorted({provenance.python_version for provenance in provenances}),
            ),
            "platforms": cast(
                JSONList,
                sorted({provenance.platform for provenance in provenances}),
            ),
        }

    def export(self, exporter: ReportExporter, path: str) -> None:
        """Write the current report using a concrete exporter implementation."""
        if not self.report:
            self.build()
        assert self.report is not None
        exporter.export(self.report, path)

    def to_csv(self, path: str) -> None:
        """Build and export the report as CSV files."""
        self.export(CsvExporter(), path)

    def to_markdown(self, path: str) -> None:
        """Build and export the report as a Markdown document."""
        self.export(MarkdownExporter(), path)

    def to_latex(self, path: str) -> None:
        """Build and export the report as a LaTeX document."""
        self.export(LatexExporter(), path)

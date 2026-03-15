"""Report assembly from projected score rows and optional hydrated trials."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

from themis._optional import import_optional
from themis.contracts.protocols import ReportExporter
from themis.overlays import OverlaySelection
from themis.records.report import EvaluationReport, ReportTable
from themis.records.trial import TrialRecord
from themis.report.exporters import CsvExporter, MarkdownExporter, LatexExporter
from themis.report.metric_frame_builder import MetricFrameBuilder
from themis.report.report_metadata_builder import ReportMetadataBuilder
from themis.runtime.comparison import build_comparison_table
from themis.stats._typing import MetricFrame, PandasNamespace
from themis.types.enums import PValueCorrection
from themis.types.events import ScoreRow, TrialSummaryRow

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
        overlay_key: str = "gen",
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> None:
        self.trials = trials
        self.stats_engine = stats_engine
        self.trial_summaries = list(trial_summaries or [])
        self.score_rows = list(score_rows or [])
        self.overlay_selection = OverlaySelection(
            transform_hash=transform_hash,
            evaluation_hash=evaluation_hash,
        )
        if overlay_key != self.overlay_selection.overlay_key:
            raise ValueError(
                "overlay_key must match the provided transform/evaluation selection."
            )
        self.report: EvaluationReport | None = None
        self._metric_frame_builder = MetricFrameBuilder()
        self._metadata_builder = ReportMetadataBuilder()

    def _resolved_trial_summaries(self) -> list[TrialSummaryRow]:
        if self.trial_summaries:
            return list(self.trial_summaries)
        return self._metric_frame_builder.trial_summaries_from_trials(self.trials)

    def _extract_metric_rows(self) -> MetricFrame:
        """Build the aggregate-report frame used by the StatsEngine."""
        return self._metric_frame_builder.build_report_frame(
            self._resolved_trial_summaries(),
            self._comparison_score_rows(),
        )

    def _comparison_score_rows(self) -> list[ScoreRow]:
        if self.score_rows:
            return list(self.score_rows)
        return self._metric_frame_builder.score_rows_from_trials(self.trials)

    def _get_stats_engine(self) -> StatsEngine:
        if self.stats_engine is not None:
            return self.stats_engine

        stats_module = import_optional("themis.stats.stats_engine", extra="stats")
        self.stats_engine = stats_module.StatsEngine()
        return self.stats_engine

    def build(
        self,
        *,
        p_value_correction: PValueCorrection | str = PValueCorrection.NONE,
    ) -> EvaluationReport:
        """
        Assembles tables and metadata from projections or in-memory trials.
        """
        correction = PValueCorrection(p_value_correction)
        pd = cast(PandasNamespace, import_optional("pandas", extra="stats"))
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
                data=agg.reset_index().to_dict(orient="records"),
            )
            tables.append(main_table)

        comparison_table = build_comparison_table(
            self._resolved_trial_summaries(),
            self._comparison_score_rows(),
            stats_engine=self._get_stats_engine(),
            p_value_correction=correction,
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
        meta = self._metadata_builder.build(
            self.trials,
            resolved_summaries,
            self.overlay_selection,
        )

        self.report = EvaluationReport(
            spec_hash=f"report_{len(self.trials)}",
            tables=tables,
            metadata=meta,
        )
        return self.report

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

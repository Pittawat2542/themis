"""Shared frame assembly for reporting and paired comparison workflows."""

from __future__ import annotations

from typing import cast

from themis._optional import import_optional
from themis.records.trial import TrialRecord
from themis.stats._typing import MetricFrame, PandasNamespace
from themis.types.events import ScoreRow, TrialSummaryRow
from themis.types.json_types import JSONDict


class MetricFrameBuilder:
    """Builds score-oriented DataFrames and derived row models from projections."""

    def build_report_frame(
        self,
        trial_summaries: list[TrialSummaryRow],
        score_rows: list[ScoreRow],
    ) -> MetricFrame:
        """Build the report aggregation frame used by `ReportBuilder`."""
        return self._build_frame(
            trial_summaries,
            score_rows,
            candidate_key="cand_hash",
            score_key="metric_value",
        )

    def build_comparison_frame(
        self,
        trial_summaries: list[TrialSummaryRow],
        score_rows: list[ScoreRow],
    ) -> MetricFrame:
        """Build the paired-comparison frame used by runtime comparison views."""
        return self._build_frame(
            trial_summaries,
            score_rows,
            candidate_key="candidate_id",
            score_key="score",
        )

    def score_rows_from_trials(self, trials: list[TrialRecord]) -> list[ScoreRow]:
        """Flatten in-memory trial records into score rows."""
        rows: list[ScoreRow] = []
        for trial in trials:
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

    def trial_summaries_from_trials(
        self,
        trials: list[TrialRecord],
    ) -> list[TrialSummaryRow]:
        """Derive minimal trial-summary rows from hydrated trials."""
        return [
            TrialSummaryRow(
                trial_hash=trial.spec_hash,
                model_id=trial.trial_spec.model.model_id if trial.trial_spec else None,
                task_id=trial.trial_spec.task.task_id if trial.trial_spec else None,
                item_id=trial.trial_spec.item_id if trial.trial_spec else None,
                status=trial.status,
            )
            for trial in trials
        ]

    def _build_frame(
        self,
        trial_summaries: list[TrialSummaryRow],
        score_rows: list[ScoreRow],
        *,
        candidate_key: str,
        score_key: str,
    ) -> MetricFrame:
        pd = cast(PandasNamespace, import_optional("pandas", extra="stats"))
        trial_metadata = {
            summary.trial_hash: {
                "model_id": summary.model_id or "unknown",
                "task_id": summary.task_id or "unknown",
                "item_id": summary.item_id or "unknown",
            }
            for summary in trial_summaries
        }
        rows: list[JSONDict] = []
        for row in score_rows:
            metadata = trial_metadata.get(
                row.trial_hash,
                {"model_id": "unknown", "task_id": "unknown", "item_id": "unknown"},
            )
            rows.append(
                {
                    "trial_hash": row.trial_hash,
                    candidate_key: row.candidate_id,
                    "metric_id": row.metric_id,
                    score_key: row.score,
                    "model_id": metadata["model_id"],
                    "task_id": metadata["task_id"],
                    "item_id": metadata["item_id"],
                }
            )
        return pd.DataFrame(rows)

"""Output-side facade over trial projections, summaries, and reports."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

from themis._optional import import_optional
from themis.records.trial import TrialRecord
from themis.runtime.comparison import build_comparison_table
from themis.runtime.timeline_view import RecordTimelineView
from themis.storage.events import TrialSummaryRow

if TYPE_CHECKING:
    from themis.contracts.protocols import ProjectionRepository
    from themis.report.builder import ReportBuilder
    from themis.runtime.comparison import ComparisonTable


class ExperimentResult:
    """Stable output-side facade over projection-backed experiment results."""

    def __init__(
        self,
        *,
        projection_repo: ProjectionRepository,
        trial_hashes: list[str],
        eval_revision: str = "latest",
    ) -> None:
        self.projection_repo = projection_repo
        self.trial_hashes = list(trial_hashes)
        self.eval_revision = eval_revision

    def iter_trials(self) -> Iterator[TrialRecord]:
        """Yield materialized trial records for the configured trial hashes."""
        for trial_hash in self.trial_hashes:
            trial = self.get_trial(trial_hash)
            if trial is not None:
                yield trial

    def get_trial(self, trial_hash: str) -> TrialRecord | None:
        """Return one materialized trial record for the active revision."""
        return self.projection_repo.get_trial_record(trial_hash, self.eval_revision)

    def iter_trial_summaries(self) -> Iterator[TrialSummaryRow]:
        """Yield summary rows for the active trial set without full hydration."""
        yield from self.projection_repo.iter_trial_summaries(
            trial_hashes=self.trial_hashes,
        )

    def view_timeline(
        self,
        record_id: str,
        *,
        record_type: Literal["candidate", "trial"] = "candidate",
    ) -> RecordTimelineView | None:
        """Return the operator-facing timeline view for a trial or candidate."""
        return self.projection_repo.get_timeline_view(
            record_id, record_type, self.eval_revision
        )

    def compare(
        self,
        *,
        trial_hash: str | None = None,
        metric_id: str | None = None,
        task_id: str | None = None,
        baseline_model_id: str | None = None,
        treatment_model_id: str | None = None,
        p_value_correction: str = "none",
    ) -> ComparisonTable:
        """Build a paired comparison table from summary and metric projection rows."""
        import_optional("themis.stats.stats_engine", extra="stats")
        return build_comparison_table(
            list(self.iter_trial_summaries()),
            list(
                self.projection_repo.iter_candidate_scores(
                    eval_revision=self.eval_revision,
                    trial_hash=trial_hash,
                    metric_id=metric_id,
                )
            ),
            metric_id=metric_id,
            task_id=task_id,
            baseline_model_id=baseline_model_id,
            treatment_model_id=treatment_model_id,
            p_value_correction=p_value_correction,
        )

    def report(self) -> ReportBuilder:
        """Build a report builder backed by projections for the active revision."""
        from themis.report.builder import ReportBuilder

        import_optional("themis.stats.stats_engine", extra="stats")
        trial_summaries = list(self.iter_trial_summaries())
        return ReportBuilder(
            list(self.iter_trials()),
            trial_summaries=trial_summaries,
            score_rows=list(
                self.projection_repo.iter_candidate_scores(
                    eval_revision=self.eval_revision,
                )
            ),
            eval_revision=self.eval_revision,
        )

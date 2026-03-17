"""Output-side facade over trial projections, summaries, and reports."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from themis._optional import import_optional
from themis.records.trial import TrialRecord
from themis.report.builder import ReportBuilder
from themis.runtime.comparison import ComparisonTable
from themis.runtime.result_services import (
    ExperimentResultAnalysisService,
    ExperimentResultDiagnosticsService,
    ResultOverlayContext,
)
from themis.runtime.timeline_view import RecordTimelineView
from themis.types.enums import PValueCorrection, RecordType
from themis.types.events import ScoreRow, TrialSummaryRow


class _ExperimentProjectionRepository(Protocol):
    """Read-only projection contract needed by `ExperimentResult`."""

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None: ...

    def get_timeline_view(
        self,
        record_id: str,
        record_type: RecordType | str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None: ...

    def iter_candidate_scores(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[ScoreRow]: ...

    def iter_trial_summaries(
        self,
        *,
        trial_hashes: Sequence[str] | None = None,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[TrialSummaryRow]: ...


def _require_optional_dependency(module_name: str, *, extra: str) -> object:
    return import_optional(module_name, extra=extra)


class ExperimentResult:
    """Stable output-side facade over projection-backed experiment results."""

    def __init__(
        self,
        *,
        projection_repo: _ExperimentProjectionRepository,
        trial_hashes: list[str],
        transform_hashes: list[str] | None = None,
        evaluation_hashes: list[str] | None = None,
        active_transform_hash: str | None = None,
        active_evaluation_hash: str | None = None,
    ) -> None:
        self.projection_repo = projection_repo
        self.trial_hashes = list(trial_hashes)
        self.transform_hashes = list(transform_hashes or [])
        self.evaluation_hashes = list(evaluation_hashes or [])
        self.active_transform_hash = active_transform_hash
        self.active_evaluation_hash = active_evaluation_hash
        context = ResultOverlayContext(
            trial_hashes=self.trial_hashes,
            transform_hashes=self.transform_hashes,
            evaluation_hashes=self.evaluation_hashes,
            active_transform_hash=self.active_transform_hash,
            active_evaluation_hash=self.active_evaluation_hash,
        )
        self._analysis = ExperimentResultAnalysisService(
            projection_repo=projection_repo,
            context=context,
            iter_trials=self.iter_trials,
            iter_trial_summaries=self.iter_trial_summaries,
            require_optional=_require_optional_dependency,
        )
        self._diagnostics = ExperimentResultDiagnosticsService(
            context=context,
            iter_trials=self.iter_trials,
        )

    def iter_trials(self) -> Iterator[TrialRecord]:
        """Yield materialized trial records for the active overlay selection."""
        for trial_hash in self.trial_hashes:
            trial = self.get_trial(trial_hash)
            if trial is not None:
                yield trial

    def get_trial(self, trial_hash: str) -> TrialRecord | None:
        """Return one materialized trial record for the active overlay selection."""
        return self.projection_repo.get_trial_record(
            trial_hash,
            transform_hash=self.active_transform_hash,
            evaluation_hash=self.active_evaluation_hash,
        )

    def iter_trial_summaries(self) -> Iterator[TrialSummaryRow]:
        """Yield summary rows for the active trial set without full hydration."""
        yield from self.projection_repo.iter_trial_summaries(
            trial_hashes=self.trial_hashes,
            transform_hash=self.active_transform_hash,
            evaluation_hash=self.active_evaluation_hash,
        )

    def view_timeline(
        self,
        record_id: str,
        *,
        record_type: RecordType | str = RecordType.CANDIDATE,
    ) -> RecordTimelineView | None:
        """Return the operator-facing timeline view for a trial or candidate."""
        resolved_record_type = RecordType(record_type)
        return self.projection_repo.get_timeline_view(
            record_id,
            resolved_record_type,
            transform_hash=self.active_transform_hash,
            evaluation_hash=self.active_evaluation_hash,
        )

    def for_transform(self, transform_hash: str) -> "ExperimentResult":
        """Return a view pinned to one output-transform overlay.

        Use this when you want normalized candidate outputs without scoring.

        Args:
            transform_hash: Deterministic transform overlay hash to pin as the
                active read-side selection.

        Returns:
            A new `ExperimentResult` view scoped to the requested transform
            overlay.
        """
        return ExperimentResult(
            projection_repo=self.projection_repo,
            trial_hashes=self.trial_hashes,
            transform_hashes=self.transform_hashes,
            evaluation_hashes=self.evaluation_hashes,
            active_transform_hash=transform_hash,
            active_evaluation_hash=None,
        )

    def for_evaluation(self, evaluation_hash: str) -> "ExperimentResult":
        """Return a view pinned to one evaluation overlay.

        Use this when you want metric-backed candidates, comparisons, or report
        exports for one specific evaluation pass.

        Args:
            evaluation_hash: Deterministic evaluation overlay hash to pin as the
                active read-side selection.

        Returns:
            A new `ExperimentResult` view scoped to the requested evaluation
            overlay.
        """
        return ExperimentResult(
            projection_repo=self.projection_repo,
            trial_hashes=self.trial_hashes,
            transform_hashes=self.transform_hashes,
            evaluation_hashes=self.evaluation_hashes,
            active_transform_hash=None,
            active_evaluation_hash=evaluation_hash,
        )

    def compare(
        self,
        *,
        trial_hash: str | None = None,
        metric_id: str | None = None,
        task_id: str | None = None,
        baseline_model_id: str | None = None,
        treatment_model_id: str | None = None,
        p_value_correction: PValueCorrection | str = PValueCorrection.NONE,
    ) -> ComparisonTable:
        """Build a paired comparison table from projection rows.

        Args:
            trial_hash: Optional trial hash to limit the comparison to one trial.
            metric_id: Optional metric ID used to filter projected score rows.
            task_id: Optional task ID used to limit the comparison to one task.
            baseline_model_id: Optional model ID to force as the baseline side.
            treatment_model_id: Optional model ID to force as the treatment side.
            p_value_correction: Multiple-comparison correction mode applied to
                the reported p-values.

        Returns:
            A comparison table built from the currently active overlay.

        Raises:
            ModuleNotFoundError: If the `stats` extra is not installed.
        """
        return self._analysis.compare(
            trial_hash=trial_hash,
            metric_id=metric_id,
            task_id=task_id,
            baseline_model_id=baseline_model_id,
            treatment_model_id=treatment_model_id,
            p_value_correction=p_value_correction,
        )

    def leaderboard(
        self,
        *,
        metric_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return aggregate leaderboard rows for the active overlay selection."""
        return self._analysis.leaderboard(metric_id=metric_id, task_id=task_id)

    def iter_invalid_extractions(self) -> Iterator[dict[str, object]]:
        """Yield extraction failures or null parses for the active overlay."""
        yield from self._diagnostics.iter_invalid_extractions()

    def iter_failures(self) -> Iterator[dict[str, object]]:
        """Yield trial-level and candidate-level failure diagnostics."""
        yield from self._diagnostics.iter_failures()

    def iter_tagged_examples(
        self,
        *,
        tag: str | None = None,
    ) -> Iterator[dict[str, object]]:
        """Yield candidate examples whose evaluation details or payloads carry tags."""
        yield from self._diagnostics.iter_tagged_examples(tag=tag)

    def export_json(
        self,
        path: str | None = None,
        *,
        include_trials: bool = True,
    ) -> dict[str, object]:
        """Return a JSON-serializable payload for the active overlay.

        Args:
            path: Optional path where the JSON payload should also be written.
            include_trials: Whether to include fully hydrated trial records in
                the exported payload.

        Returns:
            A JSON-serializable dictionary describing the active overlay.

        Raises:
            OSError: If `path` is supplied and the payload cannot be written.
        """
        return self._analysis.export_json(path, include_trials=include_trials)

    def report(self) -> ReportBuilder:
        """Build a report builder for the active overlay.

        Returns:
            A report builder bound to the currently active overlay selection.

        Raises:
            ModuleNotFoundError: If the `stats` extra is not installed.
        """
        return self._analysis.report()

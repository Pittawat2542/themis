"""Read-side services used by the ExperimentResult facade."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from themis._optional import import_optional
from themis.overlays import OverlaySelection
from themis.records.trial import TrialRecord
from themis.runtime.comparison import build_comparison_table
from themis.types.enums import PValueCorrection
from themis.types.events import ScoreRow, TrialSummaryRow

if TYPE_CHECKING:
    from themis.report.builder import ReportBuilder
    from themis.runtime.comparison import ComparisonTable


class _ScoreQueryRepository(Protocol):
    def iter_candidate_scores(
        self,
        *,
        trial_hashes: list[str] | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ) -> Iterator[ScoreRow]: ...


class OptionalImporter(Protocol):
    """Callable signature used to import optional dependencies lazily."""

    def __call__(self, module_name: str, *, extra: str) -> object: ...


@dataclass(frozen=True, slots=True)
class ResultOverlayContext:
    """Overlay selection and trial scope for one ExperimentResult view."""

    trial_hashes: list[str]
    transform_hashes: list[str]
    evaluation_hashes: list[str]
    active_transform_hash: str | None
    active_evaluation_hash: str | None

    def overlay_selection(self) -> OverlaySelection:
        """Returns the active overlay selection for this result view."""

        return OverlaySelection(
            transform_hash=self.active_transform_hash,
            evaluation_hash=self.active_evaluation_hash,
        )


class ExperimentResultAnalysisService:
    """Projection-backed analysis and export helpers for one result view."""

    def __init__(
        self,
        *,
        projection_repo: _ScoreQueryRepository,
        context: ResultOverlayContext,
        iter_trials: Callable[[], Iterator[TrialRecord]],
        iter_trial_summaries: Callable[[], Iterator[TrialSummaryRow]],
        require_optional: OptionalImporter = import_optional,
    ) -> None:
        self.projection_repo = projection_repo
        self.context = context
        self.iter_trials = iter_trials
        self.iter_trial_summaries = iter_trial_summaries
        self.require_optional = require_optional

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
        """Builds a statistical comparison table for the current result view."""

        self.require_optional("themis.stats.stats_engine", extra="stats")
        correction = PValueCorrection(p_value_correction)
        selected_trial_hashes = self._selected_trial_hashes(trial_hash)
        return build_comparison_table(
            [
                row
                for row in self.iter_trial_summaries()
                if row.trial_hash in set(selected_trial_hashes)
            ],
            list(
                self.projection_repo.iter_candidate_scores(
                    trial_hashes=selected_trial_hashes,
                    metric_id=metric_id,
                    evaluation_hash=self.context.active_evaluation_hash,
                )
            ),
            metric_id=metric_id,
            task_id=task_id,
            baseline_model_id=baseline_model_id,
            treatment_model_id=treatment_model_id,
            p_value_correction=correction,
        )

    def leaderboard(
        self,
        *,
        metric_id: str | None = None,
        task_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Aggregates candidate scores into model/task leaderboard rows."""

        summary_by_trial = {
            row.trial_hash: row
            for row in self.iter_trial_summaries()
            if task_id is None or row.task_id == task_id
        }
        grouped_scores: dict[tuple[str | None, str | None, str], list[float]] = {}
        for row in self.projection_repo.iter_candidate_scores(
            trial_hashes=self.context.trial_hashes,
            metric_id=metric_id,
            evaluation_hash=self.context.active_evaluation_hash,
        ):
            summary = summary_by_trial.get(row.trial_hash)
            if summary is None:
                continue
            key = (summary.model_id, summary.task_id, row.metric_id)
            grouped_scores.setdefault(key, []).append(row.score)

        leaderboard_rows: list[dict[str, object]] = []
        for (model_id, resolved_task_id, resolved_metric_id), scores in sorted(
            grouped_scores.items(),
            key=lambda item: (
                item[0][0] or "",
                item[0][1] or "",
                item[0][2],
            ),
        ):
            leaderboard_rows.append(
                {
                    "model_id": model_id,
                    "task_id": resolved_task_id,
                    "metric_id": resolved_metric_id,
                    "mean": sum(scores) / len(scores),
                    "count": len(scores),
                    "min": min(scores),
                    "max": max(scores),
                }
            )
        return leaderboard_rows

    def export_json(
        self,
        path: str | None = None,
        *,
        include_trials: bool = True,
    ) -> dict[str, object]:
        """Exports trial summaries and score rows for downstream consumers."""

        payload: dict[str, object] = {
            "trial_hashes": list(self.context.trial_hashes),
            "transform_hashes": list(self.context.transform_hashes),
            "evaluation_hashes": list(self.context.evaluation_hashes),
            "overlay": self.context.overlay_selection().metadata(),
            "trial_summaries": [
                row.model_dump(mode="json") for row in self.iter_trial_summaries()
            ],
            "score_rows": [
                row.model_dump(mode="json")
                for row in self.projection_repo.iter_candidate_scores(
                    trial_hashes=self.context.trial_hashes,
                    evaluation_hash=self.context.active_evaluation_hash,
                )
            ],
        }
        if include_trials:
            payload["trials"] = [
                trial.model_dump(mode="json") for trial in self.iter_trials()
            ]
        if path is not None:
            Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def report(self) -> ReportBuilder:
        """Builds a report builder for the current overlay selection."""

        from themis.report.builder import ReportBuilder

        self.require_optional("themis.stats.stats_engine", extra="stats")
        trial_summaries = list(self.iter_trial_summaries())
        overlay_selection = self.context.overlay_selection()
        return ReportBuilder(
            list(self.iter_trials()),
            trial_summaries=trial_summaries,
            score_rows=list(
                self.projection_repo.iter_candidate_scores(
                    trial_hashes=self.context.trial_hashes,
                    evaluation_hash=self.context.active_evaluation_hash,
                )
            ),
            overlay_key=overlay_selection.overlay_key,
            transform_hash=self.context.active_transform_hash,
            evaluation_hash=self.context.active_evaluation_hash,
        )

    def _selected_trial_hashes(self, trial_hash: str | None) -> list[str]:
        if trial_hash is None:
            return list(self.context.trial_hashes)
        if trial_hash not in self.context.trial_hashes:
            return []
        return [trial_hash]


class ExperimentResultDiagnosticsService:
    """Diagnostics and example queries for one result view."""

    def __init__(
        self,
        *,
        context: ResultOverlayContext,
        iter_trials: Callable[[], Iterator[TrialRecord]],
    ) -> None:
        self.context = context
        self.iter_trials = iter_trials

    def iter_invalid_extractions(self) -> Iterator[dict[str, object]]:
        """Yields rows describing extraction failures in the current result view."""

        for trial in self.iter_trials():
            trial_spec = trial.trial_spec
            if trial_spec is None:
                continue
            for candidate in trial.candidates:
                candidate_id = candidate.candidate_id or candidate.spec_hash
                for extraction in candidate.extractions:
                    if extraction.success and extraction.parsed_answer is not None:
                        continue
                    yield {
                        "trial_hash": trial.spec_hash,
                        "candidate_id": candidate_id,
                        "model_id": trial_spec.model.model_id,
                        "task_id": trial_spec.task.task_id,
                        "item_id": trial_spec.item_id,
                        "extractor_id": extraction.extractor_id,
                        "failure_reason": extraction.failure_reason,
                        "warnings": list(extraction.warnings),
                        "transform_hash": self.context.active_transform_hash,
                        "evaluation_hash": self.context.active_evaluation_hash,
                    }

    def iter_failures(self) -> Iterator[dict[str, object]]:
        """Yields trial- and candidate-level failure rows."""

        for trial in self.iter_trials():
            trial_spec = trial.trial_spec
            if trial_spec is None:
                continue
            if trial.status.value != "ok":
                yield self._failure_row(
                    level="trial",
                    trial=trial,
                    candidate_id=None,
                    status=trial.status.value,
                    error=trial.error,
                )
            for candidate in trial.candidates:
                if candidate.status.value == "ok" and candidate.error is None:
                    continue
                yield self._failure_row(
                    level="candidate",
                    trial=trial,
                    candidate_id=candidate.candidate_id or candidate.spec_hash,
                    status=candidate.status.value,
                    error=candidate.error,
                )

    def iter_tagged_examples(
        self,
        *,
        tag: str | None = None,
    ) -> Iterator[dict[str, object]]:
        """Yields examples tagged in metric details, optionally filtered by tag."""

        for trial in self.iter_trials():
            trial_spec = trial.trial_spec
            if trial_spec is None:
                continue
            for candidate in trial.candidates:
                tags = self._candidate_tags(candidate)
                if not tags:
                    continue
                if tag is not None and tag not in tags:
                    continue
                yield {
                    "trial_hash": trial.spec_hash,
                    "candidate_id": candidate.candidate_id or candidate.spec_hash,
                    "model_id": trial_spec.model.model_id,
                    "task_id": trial_spec.task.task_id,
                    "item_id": trial_spec.item_id,
                    "tags": tags,
                    "source": "metric_details",
                    "transform_hash": self.context.active_transform_hash,
                    "evaluation_hash": self.context.active_evaluation_hash,
                }

    def _failure_row(
        self,
        *,
        level: str,
        trial: TrialRecord,
        candidate_id: str | None,
        status: str,
        error,
    ) -> dict[str, object]:
        trial_spec = trial.trial_spec
        assert trial_spec is not None
        return {
            "level": level,
            "trial_hash": trial.spec_hash,
            "candidate_id": candidate_id,
            "model_id": trial_spec.model.model_id,
            "task_id": trial_spec.task.task_id,
            "item_id": trial_spec.item_id,
            "status": status,
            "where": error.where.value if error is not None else None,
            "code": error.code.value if error is not None else None,
            "message": error.message if error is not None else None,
            "transform_hash": self.context.active_transform_hash,
            "evaluation_hash": self.context.active_evaluation_hash,
        }

    def _candidate_tags(self, candidate) -> list[str]:
        tags: set[str] = set()
        candidate_tags = getattr(candidate, "tags", None)
        if isinstance(candidate_tags, list):
            tags.update(str(tag) for tag in candidate_tags if str(tag))
        evaluation = getattr(candidate, "evaluation", None)
        if evaluation is not None:
            for score in evaluation.metric_scores:
                tags.update(self._tags_from_metric_details(score.details))
        return sorted(tags)

    def _tags_from_metric_details(self, details: object) -> list[str]:
        if not isinstance(details, dict):
            return []
        tags: list[str] = []
        raw_tags = details.get("tags")
        if isinstance(raw_tags, list):
            tags.extend(str(tag) for tag in raw_tags if str(tag))
        raw_tag = details.get("tag")
        if isinstance(raw_tag, str) and raw_tag:
            tags.append(raw_tag)
        return tags

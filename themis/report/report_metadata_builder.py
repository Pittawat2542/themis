"""Metadata and provenance assembly for evaluation reports."""

from __future__ import annotations

from typing import cast

import themis

from themis.overlays import OverlaySelection
from themis.records.report import ReportMetadata
from themis.records.trial import TrialRecord
from themis.types.json_types import JSONDict, JSONList
from themis.types.events import TrialSummaryRow


class ReportMetadataBuilder:
    """Builds report metadata without coupling it to table assembly."""

    def build(
        self,
        trials: list[TrialRecord],
        trial_summaries: list[TrialSummaryRow],
        overlay_selection: OverlaySelection,
    ) -> ReportMetadata:
        """Build one metadata record from trials, summaries, and overlay selection."""
        dataset_revisions = sorted(
            {
                trial.trial_spec.task.dataset.revision
                for trial in trials
                if trial.trial_spec is not None
                and trial.trial_spec.task.dataset.revision is not None
            }
        )
        extras: JSONDict = {
            "dataset_revisions": cast(JSONList, dataset_revisions.copy())
        }
        extras.update(overlay_selection.metadata())
        extras["provenance"] = self.summarize_provenance(trials)

        return ReportMetadata(
            spec_hash=f"meta_{len(trials)}",
            themis_version=themis.__version__,
            spec_hashes=[summary.trial_hash for summary in trial_summaries],
            extras=extras,
        )

    def summarize_provenance(self, trials: list[TrialRecord]) -> JSONDict:
        """Collect de-duplicated provenance fields across the included trials."""
        provenances = [
            trial.provenance for trial in trials if trial.provenance is not None
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

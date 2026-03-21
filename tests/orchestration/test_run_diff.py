"""Tests for RunDiff.has_invalidated_resume_work and CostEstimate.trial_count."""

from __future__ import annotations

from themis.orchestration.run_manifest import CostEstimate, RunDiff


class TestRunDiffHasInvalidatedResumeWork:
    def test_false_when_no_removed_trial_hashes(self) -> None:
        diff = RunDiff(
            experiment_hash_before="abc",
            experiment_hash_after="abc",
            removed_trial_hashes=[],
        )
        assert diff.has_invalidated_resume_work is False

    def test_true_when_removed_trial_hashes_present(self) -> None:
        diff = RunDiff(
            experiment_hash_before="abc",
            experiment_hash_after="def",
            removed_trial_hashes=["hash1", "hash2"],
        )
        assert diff.has_invalidated_resume_work is True

    def test_false_when_only_added_hashes(self) -> None:
        """Adding new models/prompts does not invalidate existing completed work."""
        diff = RunDiff(
            experiment_hash_before="abc",
            experiment_hash_after="def",
            added_trial_hashes=["new-hash"],
            removed_trial_hashes=[],
        )
        assert diff.has_invalidated_resume_work is False

    def test_true_when_single_removed_hash(self) -> None:
        diff = RunDiff(
            experiment_hash_before="abc",
            experiment_hash_after="def",
            removed_trial_hashes=["hash1"],
        )
        assert diff.has_invalidated_resume_work is True


class TestCostEstimateTrialCount:
    def _make_estimate(self, **kwargs) -> CostEstimate:
        defaults = dict(
            run_id="run-1",
            backend_kind="local",
            total_work_items=4,
            estimated_prompt_tokens=100,
            estimated_completion_tokens=50,
            estimated_total_tokens=150,
        )
        defaults.update(kwargs)
        return CostEstimate.model_validate(defaults)

    def test_trial_count_defaults_to_zero(self) -> None:
        estimate = self._make_estimate()
        assert estimate.trial_count == 0

    def test_trial_count_can_be_set(self) -> None:
        estimate = self._make_estimate(trial_count=6)
        assert estimate.trial_count == 6

    def test_trial_matrix_defaults_to_empty(self) -> None:
        estimate = self._make_estimate()
        assert estimate.trial_matrix == {}

    def test_trial_matrix_records_dimensions(self) -> None:
        estimate = self._make_estimate(
            trial_matrix={"models": 2, "prompt_variants": 3, "inference_params": 1}
        )
        assert estimate.trial_matrix["models"] == 2
        assert estimate.trial_matrix["prompt_variants"] == 3
        assert estimate.trial_matrix["inference_params"] == 1

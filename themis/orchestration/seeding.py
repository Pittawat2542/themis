"""Shared deterministic seeding helpers for planning and execution."""

from __future__ import annotations

import hashlib

from themis.specs.experiment import TrialSpec


def _seed_digest_int(*parts: object) -> int:
    digest = hashlib.sha256(
        ":".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def derive_candidate_seed(
    project_seed: int | None,
    trial_hash: str,
    cand_index: int,
) -> int | None:
    """Return the deterministic per-candidate seed for one trial sample."""
    if project_seed is None:
        return None
    return _seed_digest_int(project_seed, trial_hash, cand_index)


def effective_trial_seed(
    trial: TrialSpec,
    *,
    candidate_seed: int | None,
) -> int | None:
    """Return the execution seed that should drive one inference call."""
    return trial.params.seed if trial.params.seed is not None else candidate_seed


def trial_with_effective_seed(
    trial: TrialSpec,
    *,
    candidate_seed: int | None,
) -> TrialSpec:
    """Return a trial view whose params carry the effective execution seed."""
    effective_seed = effective_trial_seed(trial, candidate_seed=candidate_seed)
    if effective_seed is None or trial.params.seed is not None:
        return trial
    return trial.model_copy(
        update={
            "params": trial.params.model_copy(update={"seed": effective_seed}),
        }
    )


def judge_call_id(
    parent_candidate_hash: str,
    metric_id: str,
    judge_call_index: int,
) -> str:
    """Return the deterministic judge trial ID for one audit-tracked call."""
    return f"judge_{parent_candidate_hash}_{metric_id}_{judge_call_index}"


def derive_judge_seed(
    parent_candidate_seed: int | None,
    metric_id: str,
    judge_call_index: int,
) -> int | None:
    """Return a deterministic seed for one judge call when none is explicit."""
    if parent_candidate_seed is None:
        return None
    return _seed_digest_int(parent_candidate_seed, metric_id, judge_call_index)

"""Lazy work-item planner for Themis Phase 2."""

from __future__ import annotations

import hashlib

from themis.core.results import GenerationWorkItem
from themis.core.snapshot import RunSnapshot


class Planner:
    def validate_snapshot(self, snapshot: RunSnapshot) -> None:
        candidate_policy = snapshot.identity.candidate_policy
        if candidate_policy.get("self_consistency") and "self_consistency_count" not in candidate_policy:
            raise ValueError("self_consistency_count is required when self_consistency is enabled")

        candidate_count = self.candidate_count(snapshot)
        if len(snapshot.component_refs.parsers) > 1:
            raise ValueError("Phase 2 supports at most one parser")
        if candidate_count > 1 and snapshot.component_refs.reducer is None:
            raise ValueError("Multi-candidate runs require an explicit reducer")
        workflow_metric_kinds = [kind for kind in snapshot.metric_kinds if kind != "pure"]
        if workflow_metric_kinds and not snapshot.component_refs.judge_models:
            raise ValueError("Workflow-backed metrics require at least one judge model")
        if "selection" in snapshot.metric_kinds and candidate_count < 2:
            raise ValueError("Selection metrics require at least two candidates")
        if snapshot.identity.seeds and len(snapshot.identity.seeds) != candidate_count:
            raise ValueError("Explicit seeds must match the planned candidate count")

    def candidate_count(self, snapshot: RunSnapshot) -> int:
        policy = snapshot.identity.candidate_policy
        values = [1]
        for key in ("num_samples", "pass_k", "best_of", "self_consistency_count"):
            value = policy.get(key)
            if isinstance(value, int) and value > 0:
                values.append(value)
        return max(values)

    async def iter_work_items(self, snapshot: RunSnapshot):
        self.validate_snapshot(snapshot)
        candidate_count = self.candidate_count(snapshot)
        seeds = self._candidate_seeds(snapshot, candidate_count)

        for dataset in snapshot.datasets:
            for case in dataset.cases:
                for candidate_index, seed in enumerate(seeds):
                    yield GenerationWorkItem(
                        run_id=snapshot.run_id,
                        dataset_id=dataset.dataset_id,
                        case=case,
                        case_id=case.case_id,
                        candidate_index=candidate_index,
                        candidate_id=f"{case.case_id}-candidate-{candidate_index}",
                        seed=seed,
                    )

    def _candidate_seeds(self, snapshot: RunSnapshot, candidate_count: int) -> list[int]:
        if snapshot.identity.seeds:
            return list(snapshot.identity.seeds)

        seeds: list[int] = []
        for candidate_index in range(candidate_count):
            digest = hashlib.sha256(f"{snapshot.run_id}:{candidate_index}".encode("utf-8")).hexdigest()
            seeds.append(int(digest[:8], 16))
        return seeds

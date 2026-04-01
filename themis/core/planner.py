"""Lazy work-item planner for the Themis v4 runtime."""

from __future__ import annotations

import hashlib

from themis.core.results import GenerationWorkItem, RunEstimate
from themis.core.snapshot import RunSnapshot
from themis.core.workflows import JudgeCall


class Planner:
    @staticmethod
    def judge_seed_for_call(
        *,
        run_id: str,
        case_id: str,
        metric_id: str,
        judge_model_id: str = "",
        judge_index: int = 0,
        repeat_index: int = 0,
        dimension_id: str | None = None,
        candidate_indices: list[int] | None = None,
    ) -> int:
        digest = hashlib.sha256(
            ":".join(
                [
                    run_id,
                    case_id,
                    metric_id,
                    judge_model_id,
                    str(judge_index),
                    str(repeat_index),
                    dimension_id or "",
                    ",".join(str(index) for index in (candidate_indices or [])),
                ]
            ).encode("utf-8")
        ).hexdigest()
        return int(digest[:8], 16)

    def plan_judge_calls(
        self,
        *,
        run_id: str,
        case_id: str,
        metric_id: str,
        calls: list[JudgeCall],
    ) -> list[JudgeCall]:
        planned: list[JudgeCall] = []
        for call in sorted(
            calls,
            key=lambda item: (
                item.call_id,
                item.dimension_id or "",
                item.judge_model_id,
                item.repeat_index,
                tuple(item.candidate_indices),
            ),
        ):
            planned.append(
                call.model_copy(
                    update={
                        "effective_seed": self.judge_seed_for_call(
                            run_id=run_id,
                            case_id=case_id,
                            metric_id=metric_id,
                            judge_model_id=call.judge_model_id,
                            repeat_index=call.repeat_index,
                            dimension_id=call.dimension_id,
                            candidate_indices=call.candidate_indices,
                        )
                    }
                )
            )
        return planned

    def validate_snapshot(self, snapshot: RunSnapshot) -> None:
        candidate_policy = snapshot.identity.candidate_policy
        if (
            candidate_policy.get("self_consistency")
            and "self_consistency_count" not in candidate_policy
        ):
            raise ValueError(
                "self_consistency_count is required when self_consistency is enabled"
            )

        candidate_count = self.candidate_count(snapshot)
        if len(snapshot.component_refs.parsers) > 1:
            raise ValueError("Phase 2 supports at most one parser")
        if (
            candidate_count > 1
            and snapshot.component_refs.reducer is None
            and snapshot.component_refs.selector is None
        ):
            raise ValueError(
                "Multi-candidate runs require an explicit reducer or selector"
            )
        workflow_metric_kinds = [
            kind for kind in snapshot.metric_kinds if kind != "pure"
        ]
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

    def estimate(self, snapshot: RunSnapshot) -> RunEstimate:
        self.validate_snapshot(snapshot)
        total_cases = sum(len(dataset.cases) for dataset in snapshot.datasets)
        candidate_count = self.candidate_count(snapshot)
        metric_count = len(snapshot.component_refs.metrics)
        pure_metric_count = sum(1 for kind in snapshot.metric_kinds if kind == "pure")
        workflow_metric_count = metric_count - pure_metric_count
        return RunEstimate(
            run_id=snapshot.run_id,
            total_cases=total_cases,
            candidate_count=candidate_count,
            metric_count=metric_count,
            pure_metric_count=pure_metric_count,
            workflow_metric_count=workflow_metric_count,
            planned_generation_tasks=total_cases * candidate_count,
            planned_reduction_tasks=total_cases if candidate_count > 1 else 0,
            planned_parse_tasks=total_cases if snapshot.component_refs.parsers else 0,
            planned_score_tasks=total_cases * metric_count,
        )

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

    def _candidate_seeds(
        self, snapshot: RunSnapshot, candidate_count: int
    ) -> list[int]:
        if snapshot.identity.seeds:
            return list(snapshot.identity.seeds)

        seeds: list[int] = []
        for candidate_index in range(candidate_count):
            digest = hashlib.sha256(
                f"{snapshot.run_id}:{candidate_index}".encode("utf-8")
            ).hexdigest()
            seeds.append(int(digest[:8], 16))
        return seeds

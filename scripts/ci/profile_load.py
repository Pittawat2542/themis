#!/usr/bin/env python3
"""Profile a synthetic load scenario and emit JSON."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    RuntimeConfig,
    StorageConfig,
)  # noqa: E402
from themis.core.events import RunEvent  # noqa: E402
from themis.core.experiment import Experiment  # noqa: E402
from themis.core.models import (
    Case,
    Dataset,
    GenerationResult,
    ParsedOutput,
    ReducedCandidate,
    Score,
)  # noqa: E402
from themis.core.protocols import JudgeModel  # noqa: E402
from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
)  # noqa: E402
from themis.core.snapshot import RunSnapshot, StoredRun  # noqa: E402
from themis.core.store import RunStore  # noqa: E402
from themis.core.workflows import (
    AggregationResult,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
)  # noqa: E402


class ProfileGenerator:
    component_id = "generator/profile"
    version = "1.0"

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def fingerprint(self) -> str:
        return "generator-profile"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return GenerationResult(
                candidate_id=f"{case.case_id}-candidate-{ctx.seed}",
                final_output=case.expected_output,
            )
        finally:
            self.active -= 1


class ProfileReducer:
    component_id = "reducer/profile"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-profile"

    async def reduce(
        self, candidates: list[GenerationResult], ctx: ReduceContext
    ) -> ReducedCandidate:
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[0].final_output,
        )


class ProfileParser:
    component_id = "parser/profile"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-profile"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        return ParsedOutput(value=candidate.final_output, format="json")


class ProfileJudgeModel:
    version = "1.0"

    def __init__(self, index: int) -> None:
        self.component_id = f"judge/profile-{index}"
        self.active = 0
        self.max_active = 0

    def fingerprint(self) -> str:
        return f"{self.component_id}-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del prompt, seed
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return JudgeResponse(
                judge_model_id=self.component_id,
                judge_model_version=self.version,
                judge_model_fingerprint=self.fingerprint(),
                raw_response="pass",
            )
        finally:
            self.active -= 1


class ProfileWorkflow:
    component_id = "workflow/profile"
    version = "1.0"

    def __init__(self, judge_models: list[ProfileJudgeModel]) -> None:
        self._judge_models = judge_models

    def fingerprint(self) -> str:
        return "workflow-profile"

    def judge_calls(self) -> list[JudgeCall]:
        return [
            JudgeCall(call_id=f"call-{index}", judge_model_id=model.component_id)
            for index, model in enumerate(self._judge_models)
        ]

    def render_prompt(
        self, call: JudgeCall, subject, ctx: EvalScoreContext
    ) -> RenderedJudgePrompt:
        del ctx
        return RenderedJudgePrompt(
            prompt_id=f"prompt-{call.call_id}",
            content=str(subject.candidates[0].final_output),
        )

    def parse_judgment(
        self, call: JudgeCall, response: JudgeResponse, ctx: EvalScoreContext
    ) -> ParsedJudgment:
        del call, ctx
        return ParsedJudgment(label=response.raw_response, score=1.0)

    def score_judgment(
        self, call: JudgeCall, judgment: ParsedJudgment, ctx: EvalScoreContext
    ) -> Score | None:
        del call, ctx
        return Score(metric_id="metric/profile", value=float(judgment.score or 0.0))

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del judgments, ctx
        if not scores:
            return None
        return AggregationResult(
            method="mean",
            value=sum(score.value for score in scores) / len(scores),
        )


class ProfileMetric:
    component_id = "metric/profile"
    version = "1.0"
    metric_family = "llm"

    def __init__(self, judge_models: list[ProfileJudgeModel]) -> None:
        self._judge_models = judge_models

    def fingerprint(self) -> str:
        return "metric-profile"

    def build_workflow(self, subject, ctx: EvalScoreContext) -> ProfileWorkflow:
        del subject, ctx
        return ProfileWorkflow(self._judge_models)


class ProfileStore(RunStore):
    def __init__(self) -> None:
        self._snapshots: dict[str, RunSnapshot] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._blobs: dict[str, tuple[str, bytes]] = {}
        self._stage_cache: dict[tuple[str, str], object] = {}

    def initialize(self) -> None:
        return None

    def persist_snapshot(self, snapshot: RunSnapshot) -> None:
        self._snapshots[snapshot.run_id] = snapshot

    def persist_event(self, event: RunEvent) -> None:
        self._events.setdefault(event.run_id, []).append(event)

    def query_events(self, run_id: str) -> list[RunEvent]:
        return list(self._events.get(run_id, []))

    def get_projection(self, run_id: str, projection_name: str):
        del run_id, projection_name
        return None

    def store_blob(self, blob: bytes, media_type: str) -> str:
        import hashlib

        ref = f"sha256:{hashlib.sha256(blob).hexdigest()}"
        self._blobs.setdefault(ref, (media_type, blob))
        return ref

    def load_blob(self, blob_ref: str) -> tuple[str, bytes] | None:
        return self._blobs.get(blob_ref)

    def resume(self, run_id: str) -> StoredRun | None:
        snapshot = self._snapshots.get(run_id)
        if snapshot is None:
            return None
        return StoredRun(snapshot=snapshot, events=self.query_events(run_id))

    def load_stage_cache(self, stage_name: str, cache_key: str):
        return self._stage_cache.get((stage_name, cache_key))

    def store_stage_cache(self, stage_name: str, cache_key: str, payload) -> None:
        self._stage_cache[(stage_name, cache_key)] = payload

    def clear_run(self, run_id: str) -> None:
        self._snapshots.pop(run_id, None)
        self._events.pop(run_id, None)


def _build_experiment(
    *,
    cases: int,
    samples: int,
    judge_count: int,
) -> tuple[Experiment, ProfileGenerator, list[ProfileJudgeModel]]:
    generator = ProfileGenerator()
    judge_models = [ProfileJudgeModel(index) for index in range(judge_count)]
    configured_judge_models = cast(list[JudgeModel | str], judge_models)
    experiment = Experiment(
        generation=GenerationConfig(
            generator=generator,
            candidate_policy={"num_samples": samples},
            reducer=ProfileReducer(),
        ),
        evaluation=EvaluationConfig(
            metrics=[ProfileMetric(judge_models)],
            parsers=[ProfileParser()],
            judge_models=configured_judge_models,
        ),
        storage=StorageConfig(store="memory"),
        runtime=RuntimeConfig(
            max_concurrent_tasks=min(32, max(1, judge_count + samples)),
            stage_concurrency={
                "generation": min(16, max(1, samples)),
                "evaluation": min(16, max(1, judge_count)),
            },
        ),
        datasets=[
            Dataset(
                dataset_id="profile",
                cases=[
                    Case(
                        case_id=f"case-{index}",
                        input={"question": f"{index}+{index}"},
                        expected_output={"answer": str(index * 2)},
                    )
                    for index in range(cases)
                ],
            )
        ],
        seeds=list(range(samples)),
    )
    return experiment, generator, judge_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--judges", type=int, default=3)
    args = parser.parse_args()

    experiment, generator, judge_models = _build_experiment(
        cases=max(1, args.cases),
        samples=max(1, args.samples),
        judge_count=max(1, args.judges),
    )
    store = ProfileStore()

    started_at = time.perf_counter()
    result = experiment.run(store=store)
    duration = time.perf_counter() - started_at

    payload = {
        "cases": args.cases,
        "samples": args.samples,
        "judges": args.judges,
        "run_id": result.run_id,
        "status": result.status.value,
        "duration_seconds": round(duration, 6),
        "completed_cases": result.progress.completed_cases,
        "failed_cases": result.progress.failed_cases,
        "max_generation_concurrency": generator.max_active,
        "max_judge_concurrency": max(
            (model.max_active for model in judge_models), default=0
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

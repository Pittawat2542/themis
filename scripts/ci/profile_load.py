#!/usr/bin/env python3
"""Profile a synthetic load scenario and emit JSON."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from themis.core.config import (
    EvaluationConfig,
    GenerationConfig,
    RuntimeConfig,
    Stage,
    StorageConfig,
    TargetSpec,
)  # noqa: E402
from themis.core.dataset_sources import inline_dataset_source  # noqa: E402
from themis.core.experiment import Experiment  # noqa: E402
from themis.core.models import (
    Case,
    Dataset,
    Candidate,
    ParsedOutput,
    ReducedCandidate,
    MetricResult,
    MetricDirection,
    MetricInterpretation,
)  # noqa: E402
from themis.core.protocols import JudgeModel  # noqa: E402
from themis.core.contexts import (
    EvalScoreContext,
    GenerationContext,
    ParseContext,
    ReduceContext,
)  # noqa: E402
from themis.core.stores.sqlite import SqliteRunStore  # noqa: E402
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

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return Candidate(
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
        self, candidates: list[Candidate], ctx: ReduceContext
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
    ) -> MetricResult | None:
        del call, ctx
        return MetricResult(
            metric_id="metric/profile", value=float(judgment.score or 0.0)
        )

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        metric_results: list[MetricResult],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del judgments, ctx
        if not metric_results:
            return None
        values = [result.value for result in metric_results if result.value is not None]
        return AggregationResult(
            method="mean",
            value=sum(values) / len(values) if values else 0.0,
        )


class ProfileMetric:
    component_id = "metric/profile"
    version = "1.0"
    metric_family = "workflow"
    subject_kind = "candidate"
    interpretation = MetricInterpretation(direction=MetricDirection.HIGHER_IS_BETTER)

    def __init__(self, judge_models: list[ProfileJudgeModel]) -> None:
        self._judge_models = judge_models

    def fingerprint(self) -> str:
        return "metric-profile"

    def build_workflow(self, subject, ctx: EvalScoreContext) -> ProfileWorkflow:
        del subject, ctx
        return ProfileWorkflow(self._judge_models)


class TimedSqliteRunStore(SqliteRunStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.append_latencies_ms: list[float] = []

    def persist_event(self, event):
        started = time.perf_counter()
        try:
            return super().persist_event(event)
        finally:
            self.append_latencies_ms.append((time.perf_counter() - started) * 1000)


def _build_experiment(
    *,
    cases: int,
    samples: int,
    judge_count: int,
    store_path: Path,
) -> tuple[Experiment, ProfileGenerator, list[ProfileJudgeModel]]:
    generator = ProfileGenerator()
    judge_models = [ProfileJudgeModel(index) for index in range(judge_count)]
    configured_judge_models = cast(list[JudgeModel | TargetSpec | str], judge_models)
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
        storage=StorageConfig(target="sqlite", kwargs={"path": str(store_path)}),
        runtime=RuntimeConfig(
            max_concurrent_tasks=min(32, max(1, judge_count + samples)),
            stage_concurrency={
                Stage.GENERATE: min(16, max(1, samples)),
                Stage.JUDGE: min(16, max(1, judge_count)),
            },
        ),
        dataset_sources=[
            inline_dataset_source(
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
    parser.add_argument("--store-path", default="/tmp/themis-load-profile.sqlite3")
    args = parser.parse_args()
    store_path = Path(args.store_path).expanduser().resolve()
    store_path.unlink(missing_ok=True)

    experiment, generator, judge_models = _build_experiment(
        cases=max(1, args.cases),
        samples=max(1, args.samples),
        judge_count=max(1, args.judges),
        store_path=store_path,
    )
    store = TimedSqliteRunStore(store_path)

    tracemalloc.start()
    started_at = time.perf_counter()
    result = experiment.run(store=store)
    duration = time.perf_counter() - started_at
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    projection_started = time.perf_counter()
    store.get_projection(result.run_id, "benchmark_result")
    projection_duration = time.perf_counter() - projection_started
    resume_started = time.perf_counter()
    store.resume(result.run_id)
    resume_duration = time.perf_counter() - resume_started
    latencies = sorted(store.append_latencies_ms)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)

    payload = {
        "cases": args.cases,
        "samples": args.samples,
        "judges": args.judges,
        "run_id": result.run_id,
        "status": result.status.value,
        "duration_seconds": round(duration, 6),
        "event_count": store.count_events(result.run_id),
        "append_latency_p95_ms": round(latencies[p95_index], 6) if latencies else 0.0,
        "projection_duration_seconds": round(projection_duration, 6),
        "bytes_written": store_path.stat().st_size,
        "peak_memory_bytes": peak_memory,
        "resume_duration_seconds": round(resume_duration, 6),
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

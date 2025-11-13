import json
from dataclasses import dataclass

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider
from themis.evaluation import metrics


@dataclass
class StubJudgeProvider(ModelProvider):
    payload: dict

    def generate(self, task: core_entities.GenerationTask) -> core_entities.GenerationRecord:  # type: ignore[override]
        text = json.dumps(self.payload)
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text=text, raw=self.payload),
            error=None,
            metrics={},
        )


def make_task_model() -> core_entities.ModelSpec:
    return core_entities.ModelSpec(identifier="judge-model", provider="stub")


def test_pairwise_judge_metric_handles_preference_and_confidence():
    judge_payload = {
        "preference": "A",
        "confidence": 0.8,
        "rationale": "A is clearer.",
    }
    provider = StubJudgeProvider(payload=judge_payload)
    metric = metrics.PairwiseJudgeMetric(
        judge_model=make_task_model(),
        judge_provider=provider,
        rubric=["clarity"],
    )

    score = metric.compute(prediction=("A text", "B text"), references=["ref"])

    assert score.value == 1.0
    assert score.details["preference"] == "a"
    assert score.details["confidence"] == 0.8

import json
from dataclasses import dataclass

from themis.core import entities as core_entities
from themis.interfaces import ModelProvider
from themis.evaluation import metrics


@dataclass
class StubJudgeProvider(ModelProvider):
    payload: dict

    def generate(
        self, task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:  # type: ignore[override]
        text = json.dumps(self.payload)
        return core_entities.GenerationRecord(
            task=task,
            output=core_entities.ModelOutput(text=text, raw=self.payload),
            error=None,
            metrics={},
        )


def make_task_model() -> core_entities.ModelSpec:
    return core_entities.ModelSpec(identifier="judge-model", provider="stub")


def test_rubric_judge_metric_aggregates_scores_and_verdict():
    judge_payload = {
        "scores": {"correctness": 1.0, "reasoning": 0.5},
        "verdict": "pass",
        "rationale": "Looks good overall.",
    }
    provider = StubJudgeProvider(payload=judge_payload)
    metric = metrics.RubricJudgeMetric(
        judge_model=make_task_model(),
        judge_provider=provider,
        rubric=["correctness", "reasoning"],
    )

    score = metric.compute(prediction="Answer text", references=["Answer ref"])

    assert score.value == 0.75
    assert score.details["verdict"] == "pass"
    assert score.details["scores"]["correctness"] == 1.0
    assert score.details["scores"]["reasoning"] == 0.5

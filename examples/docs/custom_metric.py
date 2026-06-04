from __future__ import annotations

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import ScoreContext
from themis.core.dataset_sources import inline_dataset_source
from themis.core.models import Case, Dataset, ParsedOutput, MetricResult


class ExactAnswerMetric:
    """Example pure metric with the minimum required contract."""

    component_id = "metric/exact_answer"
    version = "1.0"
    metric_family = "pure"

    def fingerprint(self) -> str:
        return "metric-exact-answer"

    def score(
        self, parsed: ParsedOutput, case: Case, ctx: ScoreContext
    ) -> MetricResult:
        del ctx
        matched = parsed.value == case.expected_output
        return MetricResult(
            metric_id=self.component_id,
            value=1.0 if matched else 0.0,
            metadata={"matched": matched},
        )


def run_example() -> dict[str, object]:
    """Execute an experiment with a custom metric object."""

    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator", reducer="builtin/majority_vote"
        ),
        evaluation=EvaluationConfig(
            metrics=[ExactAnswerMetric()], parsers=["builtin/json_identity"]
        ),
        storage=StorageConfig(target="memory"),
        dataset_sources=[
            inline_dataset_source(
                Dataset(
                    dataset_id="sample",
                    cases=[
                        Case(
                            case_id="case-1",
                            input={"question": "2+2"},
                            expected_output={"answer": "4"},
                        )
                    ],
                )
            )
        ],
    )
    result = experiment.run()
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "score_ids": [score.metric_id for score in result.cases[0].metric_results],
    }


if __name__ == "__main__":
    print(run_example())

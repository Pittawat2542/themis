from __future__ import annotations

from themis.storage import memory_store

from themis import Experiment
from themis import Evaluation, Generation
from themis.core.contexts import ScoreContext
from themis.core.dataset_sources import inline_dataset_source
from themis import Case, Dataset, MetricResult
from themis.core.models import (
    MetricDirection,
    MetricInterpretation,
    ParsedOutput,
)


class ExactAnswerMetric:
    """Example pure metric with the minimum required contract."""

    component_id = "metric/exact_answer"
    version = "1.0"
    metric_family = "pure"
    interpretation = MetricInterpretation(
        direction=MetricDirection.HIGHER_IS_BETTER,
        valid_range=(0.0, 1.0),
        correctness_threshold=1.0,
    )

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
        generation=Generation(
            generator="builtin/demo_generator", reducer="builtin/majority_vote"
        ),
        evaluation=Evaluation(
            metrics=[ExactAnswerMetric()], parser="builtin/json_identity"
        ),
        datasets=[
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
    result = experiment.run(store=memory_store())
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "score_ids": [score.metric_id for score in result.cases[0].metric_results],
    }


if __name__ == "__main__":
    print(run_example())

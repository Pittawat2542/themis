from __future__ import annotations

from typing import cast

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.protocols import LifecycleSubscriber, TracingProvider


class RecordingSubscriber:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def before_generate(self, case, ctx) -> None:
        del ctx
        self.calls.append(f"before_generate:{case.case_id}")

    def on_event(self, event) -> None:
        self.calls.append(type(event).__name__)


class RecordingTracer:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[tuple[str, str]] = []

    def start_span(self, name: str, attributes: dict[str, object]) -> object:
        del attributes
        self.started.append(name)
        return name

    def end_span(self, span: object, status: str) -> None:
        self.ended.append((str(span), status))


def run_example() -> dict[str, object]:
    """Run a small experiment with subscriber and tracing hooks attached."""

    subscriber = RecordingSubscriber()
    tracer = RecordingTracer()
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/exact_match"],
            parsers=["builtin/json_identity"],
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
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
        ],
        seeds=[7],
    )
    result = experiment.run(
        subscribers=[cast(LifecycleSubscriber, subscriber)],
        tracing_provider=cast(TracingProvider, tracer),
    )
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "subscriber_calls": subscriber.calls,
        "span_names": tracer.started,
        "ended_spans": tracer.ended,
    }


if __name__ == "__main__":
    print(run_example())

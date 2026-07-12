from __future__ import annotations

from typing import cast

from themis import Evaluation, Generation
from themis.storage import memory_store
from themis.core.dataset_sources import inline_dataset_source
from themis import Experiment
from themis import Case, Dataset
from themis.core.protocols import EventSubscriber, TracingProvider


class RecordingSubscriber:
    def __init__(self) -> None:
        self.calls: list[str] = []

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
        generation=Generation(
            generator="builtin/demo_generator",
            samples=1,
            reducer="builtin/majority_vote",
        ),
        evaluation=Evaluation(
            metrics=["builtin/exact_match"],
            parser="builtin/json_identity",
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
        seeds=[7],
    )
    result = experiment.run(
        store=memory_store(),
        subscribers=[cast(EventSubscriber, subscriber)],
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

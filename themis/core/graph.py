"""Typed graph runtime primitives for evaluation execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import Field

from themis.core.base import FrozenModel, JSONValue


class EvaluationStep(FrozenModel):
    """One typed step in an evaluation graph."""

    step_id: str
    kind: str
    depends_on: list[str] = Field(default_factory=list)
    config: dict[str, JSONValue] = Field(default_factory=dict)


class EvaluationGraph(FrozenModel):
    """Executable graph of evaluation steps."""

    graph_id: str
    steps: list[EvaluationStep] = Field(default_factory=list)


class StepOutput(FrozenModel):
    """Result emitted by one graph step."""

    step_id: str
    kind: str
    payload: dict[str, JSONValue] = Field(default_factory=dict)


class StepInput(FrozenModel):
    """Input passed to one graph step handler."""

    run_id: str
    payload: dict[str, JSONValue] = Field(default_factory=dict)
    step_outputs: dict[str, StepOutput] = Field(default_factory=dict)


class GraphRunResult(FrozenModel):
    """Ordered outputs from one graph execution."""

    run_id: str
    graph_id: str
    outputs: list[StepOutput] = Field(default_factory=list)

    def output_for(self, step_id: str) -> StepOutput:
        for output in self.outputs:
            if output.step_id == step_id:
                return output
        raise KeyError(step_id)


GraphStepHandler = Callable[[StepInput], Mapping[str, JSONValue]]


class GraphRuntime:
    """Small deterministic graph executor used by Themis runtime surfaces."""

    def __init__(self, handlers: Mapping[str, GraphStepHandler]) -> None:
        self._handlers = dict(handlers)

    def run(self, graph: EvaluationGraph, initial_input: StepInput) -> GraphRunResult:
        outputs: dict[str, StepOutput] = {}
        ordered_outputs: list[StepOutput] = []
        for step in graph.steps:
            for dependency in step.depends_on:
                if dependency not in outputs:
                    raise ValueError(
                        f"Step {step.step_id} depends on unknown step {dependency}"
                    )
            handler = self._handlers.get(step.kind)
            if handler is None:
                raise ValueError(
                    f"No graph handler registered for step kind {step.kind}"
                )
            step_input = StepInput(
                run_id=initial_input.run_id,
                payload={**initial_input.payload, **step.config},
                step_outputs=dict(outputs),
            )
            output = StepOutput(
                step_id=step.step_id,
                kind=step.kind,
                payload=dict(handler(step_input)),
            )
            outputs[step.step_id] = output
            ordered_outputs.append(output)
        return GraphRunResult(
            run_id=initial_input.run_id,
            graph_id=graph.graph_id,
            outputs=ordered_outputs,
        )

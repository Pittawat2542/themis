from __future__ import annotations

from typing import cast

from themis.core.graph import EvaluationGraph, EvaluationStep, GraphRuntime, StepInput


def test_graph_runtime_executes_steps_in_order_and_threads_outputs() -> None:
    graph = EvaluationGraph(
        graph_id="graph/demo",
        steps=[
            EvaluationStep(step_id="first", kind="prepare"),
            EvaluationStep(step_id="second", kind="score", depends_on=["first"]),
        ],
    )

    runtime = GraphRuntime(
        handlers={
            "prepare": lambda step_input: {
                "value": cast(int, step_input.payload["seed"]) + 1
            },
            "score": lambda step_input: {
                "value": cast(
                    int, step_input.step_outputs["first"].payload["value"]
                )
                * 2
            },
        }
    )

    result = runtime.run(graph, StepInput(run_id="run-1", payload={"seed": 3}))

    assert [output.step_id for output in result.outputs] == ["first", "second"]
    assert result.output_for("first").payload == {"value": 4}
    assert result.output_for("second").payload == {"value": 8}


def test_graph_runtime_rejects_missing_dependencies() -> None:
    graph = EvaluationGraph(
        graph_id="graph/missing",
        steps=[EvaluationStep(step_id="score", kind="score", depends_on=["parse"])],
    )
    runtime = GraphRuntime(handlers={"score": lambda step_input: step_input.payload})

    try:
        runtime.run(graph, StepInput(run_id="run-1", payload={}))
    except ValueError as exc:
        assert "depends on unknown step parse" in str(exc)
    else:
        raise AssertionError("expected missing dependency failure")


def test_graph_runtime_requires_handler_for_each_step_kind() -> None:
    graph = EvaluationGraph(
        graph_id="graph/no-handler",
        steps=[EvaluationStep(step_id="first", kind="prepare")],
    )
    runtime = GraphRuntime(handlers={})

    try:
        runtime.run(graph, StepInput(run_id="run-1", payload={}))
    except ValueError as exc:
        assert "No graph handler registered for step kind prepare" in str(exc)
    else:
        raise AssertionError("expected missing handler failure")

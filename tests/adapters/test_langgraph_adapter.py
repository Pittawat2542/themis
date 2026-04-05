from __future__ import annotations

import pytest

from themis.adapters.langgraph import langgraph
from themis.core.contexts import GenerateContext
from themis.core.models import Case


class FakeGraph:
    async def ainvoke(self, payload):
        return {"answer": "4", "payload": payload}

    async def astream_events(self, payload, version="v2"):
        del version
        yield {
            "event": "on_chain_start",
            "name": "graph",
            "data": {"input": payload},
        }
        yield {
            "event": "on_chain_end",
            "name": "graph",
            "data": {"output": {"answer": "4"}},
        }


@pytest.mark.asyncio
async def test_langgraph_adapter_invokes_graph_and_captures_trace() -> None:
    generator = langgraph(FakeGraph(), graph_id="graph/demo", output_key="answer")

    result = await generator.generate(
        Case(case_id="case-1", input={"question": "2+2"}, expected_output="4"),
        GenerateContext(run_id="run-1", case_id="case-1", seed=7),
    )

    assert result.final_output == "4"
    assert result.trace is not None
    assert [step.step_name for step in result.trace] == ["graph", "graph"]
    assert result.trace[0].input == {"input": {"question": "2+2"}}


def test_langgraph_adapter_fingerprint_is_deterministic() -> None:
    left = langgraph(FakeGraph(), graph_id="graph/demo", output_key="answer")
    right = langgraph(FakeGraph(), graph_id="graph/demo", output_key="answer")
    changed = langgraph(FakeGraph(), graph_id="graph/demo", output_key="payload")

    assert left.fingerprint() == right.fingerprint()
    assert left.fingerprint() != changed.fingerprint()

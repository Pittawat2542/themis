"""LangGraph generator adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from themis.adapters._utils import normalize_json_value, stable_fingerprint
from themis.core.contexts import GenerationContext
from themis.core.models import (
    Candidate,
    Case,
    GenerationTurn,
    TraceStep,
    SeedCapability,
)


class LangGraphGenerator:
    """Generator adapter for LangGraph graphs."""

    component_id = "generator/langgraph"
    version = "1.0"
    seed_capability = SeedCapability.UNSUPPORTED

    def __init__(
        self,
        graph: Any,
        *,
        graph_id: str,
        graph_version: str = "1.0",
        output_key: str | None = None,
    ) -> None:
        self.graph = graph
        self.graph_id = graph_id
        self.graph_version = graph_version
        self.output_key = output_key
        self.provider_key = f"langgraph:{graph_id}"

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "component_id": self.component_id,
                "graph_id": self.graph_id,
                "graph_version": self.graph_version,
                "output_key": self.output_key,
                "seed_capability": self.seed_capability.value,
            }
        )

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate:
        payload = (
            ctx.prompt_spec.render_input(case.input)
            if ctx.prompt_spec is not None
            else case.input
        )
        output, trace = await self._execute_once(payload)
        final_output = output
        if self.output_key is not None:
            if not isinstance(output, Mapping):
                raise TypeError(
                    "LangGraph adapter expected mapping output when output_key is provided."
                )
            final_output = output[self.output_key]
        return Candidate(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=normalize_json_value(final_output),
            turns=[GenerationTurn(turn_index=0, trace=trace)],
            trace=trace or None,
            termination_reason="completed",
            artifacts={"graph_id": self.graph_id},
        )

    async def _execute_once(self, payload: object) -> tuple[object, list[TraceStep]]:
        if hasattr(self.graph, "astream_events"):
            return await self._stream_once(payload)
        if hasattr(self.graph, "ainvoke"):
            return await self.graph.ainvoke(payload), []
        if hasattr(self.graph, "invoke"):
            return await asyncio.to_thread(self.graph.invoke, payload), []
        raise TypeError(
            "LangGraph adapter requires a graph with ainvoke() or invoke()."
        )

    async def _stream_once(self, payload: object) -> tuple[object, list[TraceStep]]:
        steps: list[TraceStep] = []
        final_output: object | None = None
        async for event in self.graph.astream_events(payload, version="v2"):
            data = event.get("data", {})
            if "output" in data:
                final_output = data["output"]
            steps.append(
                TraceStep(
                    step_name=event.get("name", "langgraph"),
                    step_type=event.get("event", "event"),
                    input={"input": data.get("input")} if "input" in data else {},
                    output={"output": data.get("output")} if "output" in data else {},
                    metadata={
                        key: value
                        for key, value in event.items()
                        if key not in {"name", "event", "data"}
                    },
                )
            )
        if final_output is None:
            raise RuntimeError(
                "LangGraph event stream completed without a final output event."
            )
        return final_output, steps


def langgraph(graph: Any, **kwargs) -> LangGraphGenerator:
    """Construct a `LangGraphGenerator`."""

    return LangGraphGenerator(graph, **kwargs)

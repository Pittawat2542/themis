"""LangGraph generator adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from themis.adapters._utils import normalize_json_value, stable_fingerprint
from themis.core.contexts import GenerateContext
from themis.core.models import Case, GenerationResult, TraceStep


class LangGraphGenerator:
    component_id = "generator/langgraph"
    version = "1.0"

    def __init__(
        self,
        graph: object,
        *,
        graph_id: str,
        graph_version: str = "1.0",
        input_builder: Any | None = None,
        output_key: str | None = None,
    ) -> None:
        self.graph = graph
        self.graph_id = graph_id
        self.graph_version = graph_version
        self.input_builder = input_builder
        self.output_key = output_key
        self.provider_key = f"langgraph:{graph_id}"

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "component_id": self.component_id,
                "graph_id": self.graph_id,
                "graph_version": self.graph_version,
                "output_key": self.output_key,
            }
        )

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        payload = self.input_builder(case) if self.input_builder is not None else case.input
        trace = await self._collect_trace(payload)
        output = await self._invoke(payload)
        final_output = output
        if self.output_key is not None:
            if not isinstance(output, Mapping):
                raise TypeError("LangGraph adapter expected mapping output when output_key is provided.")
            final_output = output[self.output_key]
        return GenerationResult(
            candidate_id=f"{case.case_id}-candidate-{ctx.seed if ctx.seed is not None else 0}",
            final_output=normalize_json_value(final_output),
            trace=trace or None,
            artifacts={"graph_id": self.graph_id},
        )

    async def _invoke(self, payload: object) -> object:
        if hasattr(self.graph, "ainvoke"):
            return await self.graph.ainvoke(payload)
        if hasattr(self.graph, "invoke"):
            return await asyncio.to_thread(self.graph.invoke, payload)
        raise TypeError("LangGraph adapter requires a graph with ainvoke() or invoke().")

    async def _collect_trace(self, payload: object) -> list[TraceStep]:
        if not hasattr(self.graph, "astream_events"):
            return []
        steps: list[TraceStep] = []
        async for event in self.graph.astream_events(payload, version="v2"):
            data = event.get("data", {})
            steps.append(
                TraceStep(
                    step_name=event.get("name", "langgraph"),
                    step_type=event.get("event", "event"),
                    input={"input": data.get("input")} if "input" in data else {},
                    output={"output": data.get("output")} if "output" in data else {},
                    metadata={key: value for key, value in event.items() if key not in {"name", "event", "data"}},
                )
            )
        return steps


def langgraph(graph: object, **kwargs) -> LangGraphGenerator:
    return LangGraphGenerator(graph, **kwargs)

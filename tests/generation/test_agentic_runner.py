"""Unit tests for the AgenticRunner module."""

from themis.core.entities import GenerationRecord, GenerationTask
from themis.core.tools import ToolDefinition, ToolRegistry
from themis.generation.agentic_runner import AgenticRunner
from themis.interfaces import StatelessTaskExecutor
from themis.generation.router import ProviderRouter


class MockExecutor(StatelessTaskExecutor):
    """Mock executor for testing agentic runner."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def execute(self, task: GenerationTask, **kwargs) -> GenerationRecord:
        response = self.responses[self._get_index()]
        self.call_count += 1
        return GenerationRecord(task=task, output=response)

    def _get_index(self):
        return min(self.call_count, len(self.responses) - 1)


def test_agentic_runner_initialization():
    runner = AgenticRunner(executor=ProviderRouter({}), tool_registry=ToolRegistry())
    assert runner is not None
    assert len(runner._tools.list_tools()) == 0


def test_agentic_runner_with_tools():
    runner = AgenticRunner(executor=ProviderRouter({}), tool_registry=ToolRegistry())

    def simple_tool(x: int) -> int:
        return x * 2

    tool_def = ToolDefinition(
        name="simple",
        description="test tool",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        handler=simple_tool,
    )
    runner._tools.register(tool_def)
    assert runner._tools.get("simple") is not None

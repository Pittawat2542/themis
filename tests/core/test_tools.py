"""Comprehensive tests for themis.core.tools module."""

import pytest

from themis.core.tools import (
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    create_calculator_tool,
    create_counter_tool,
)


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_to_dict_excludes_handler(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: None,
        )
        d = tool.to_dict()
        assert d["name"] == "t"
        assert "handler" not in d

    # -- validate_arguments: required / unknown --

    def test_validate_missing_required(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            handler=lambda args: None,
        )
        errors = tool.validate_arguments({})
        assert any("Missing required field: a" in e for e in errors)

    def test_validate_unknown_field(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "string"}},
            },
            handler=lambda args: None,
        )
        errors = tool.validate_arguments({"a": "ok", "z": 99})
        assert any("Unknown field: z" in e for e in errors)

    # -- validate_arguments: type checks --

    def test_validate_type_string(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
            handler=lambda args: None,
        )
        assert tool.validate_arguments({"x": "hello"}) == []
        errors = tool.validate_arguments({"x": 123})
        assert any("expected string" in e for e in errors)

    def test_validate_type_number(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
            handler=lambda args: None,
        )
        assert tool.validate_arguments({"x": 1.5}) == []
        assert tool.validate_arguments({"x": 1}) == []  # int is also number
        errors = tool.validate_arguments({"x": "nope"})
        assert any("expected number" in e for e in errors)

    def test_validate_type_integer_rejects_bool(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
            handler=lambda args: None,
        )
        assert tool.validate_arguments({"x": 42}) == []
        errors = tool.validate_arguments({"x": True})
        assert any("expected integer" in e for e in errors)

    def test_validate_type_boolean(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "boolean"}},
            },
            handler=lambda args: None,
        )
        assert tool.validate_arguments({"x": True}) == []
        errors = tool.validate_arguments({"x": 1})
        assert any("expected boolean" in e for e in errors)

    def test_validate_type_array(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "array"}},
            },
            handler=lambda args: None,
        )
        assert tool.validate_arguments({"x": [1, 2]}) == []
        errors = tool.validate_arguments({"x": "not a list"})
        assert any("expected array" in e for e in errors)

    def test_validate_all_valid(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            handler=lambda args: None,
        )
        assert tool.validate_arguments({"a": "hello", "b": 1}) == []


# ---------------------------------------------------------------------------
# ToolDefinition.from_function
# ---------------------------------------------------------------------------


class TestFromFunction:
    def test_basic_introspection(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = ToolDefinition.from_function(add)
        assert tool.name == "add"
        assert tool.description == "Add two numbers."
        assert tool.auto_unpack is True
        assert set(tool.parameters["required"]) == {"a", "b"}
        assert tool.parameters["properties"]["a"]["type"] == "integer"

    def test_optional_params(self):
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        tool = ToolDefinition.from_function(greet)
        assert "name" in tool.parameters["required"]
        assert "greeting" not in tool.parameters.get("required", [])

    def test_name_and_description_override(self):
        def fn(x: int) -> int:
            return x

        tool = ToolDefinition.from_function(
            fn, name="custom", description="Custom desc"
        )
        assert tool.name == "custom"
        assert tool.description == "Custom desc"

    def test_no_annotations(self):
        def fn(x):
            return x

        tool = ToolDefinition.from_function(fn)
        assert tool.parameters["properties"]["x"]["type"] == "string"  # default

    def test_from_function_execution(self):
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b

        tool = ToolDefinition.from_function(multiply)
        registry = ToolRegistry()
        registry.register(tool)
        call = ToolCall(tool_name="multiply", arguments={"a": 3.0, "b": 4.0})
        result = registry.execute(call)
        assert result.is_success()
        assert result.result == 12.0


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_to_dict(self):
        call = ToolCall(tool_name="foo", arguments={"x": 1}, call_id="abc")
        d = call.to_dict()
        assert d["tool_name"] == "foo"
        assert d["call_id"] == "abc"


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_is_success(self):
        call = ToolCall(tool_name="t", arguments={})
        result = ToolResult(call=call, result=42, error=None, execution_time_ms=1.0)
        assert result.is_success() is True

    def test_is_failure(self):
        call = ToolCall(tool_name="t", arguments={})
        result = ToolResult(call=call, result=None, error="boom", execution_time_ms=1.0)
        assert result.is_success() is False


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="t", description="d", parameters={}, handler=lambda a: None
        )
        registry.register(tool)
        assert registry.get("t") is tool
        assert registry.get("missing") is None

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="t", description="d", parameters={}, handler=lambda a: None
        )
        registry.register(tool)
        with pytest.raises(Exception, match="already registered"):
            registry.register(tool)

    def test_unregister(self):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="t", description="d", parameters={}, handler=lambda a: None
        )
        registry.register(tool)
        registry.unregister("t")
        assert registry.get("t") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        t1 = ToolDefinition(
            name="a", description="d", parameters={}, handler=lambda a: None
        )
        t2 = ToolDefinition(
            name="b", description="d", parameters={}, handler=lambda a: None
        )
        registry.register(t1)
        registry.register(t2)
        assert len(registry.list_tools()) == 2

    def test_execute_success(self):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="double",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=lambda args: args["x"] * 2,
        )
        registry.register(tool)
        result = registry.execute(ToolCall(tool_name="double", arguments={"x": 5}))
        assert result.is_success()
        assert result.result == 10

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute(ToolCall(tool_name="nope", arguments={}))
        assert not result.is_success()
        assert "Unknown tool" in result.error

    def test_execute_validation_error(self):
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=lambda args: args["x"],
        )
        registry.register(tool)
        result = registry.execute(ToolCall(tool_name="t", arguments={}))
        assert not result.is_success()
        assert "Missing required" in result.error

    def test_execute_handler_exception(self):
        def boom(args):
            raise ValueError("kaboom")

        registry = ToolRegistry()
        tool = ToolDefinition(name="boom", description="d", parameters={}, handler=boom)
        registry.register(tool)
        result = registry.execute(ToolCall(tool_name="boom", arguments={}))
        assert not result.is_success()
        assert "kaboom" in result.error

    def test_execute_auto_unpack(self):
        def add(a: int, b: int) -> int:
            return a + b

        tool = ToolDefinition.from_function(add)
        registry = ToolRegistry()
        registry.register(tool)
        result = registry.execute(ToolCall(tool_name="add", arguments={"a": 2, "b": 3}))
        assert result.is_success()
        assert result.result == 5

    def test_to_dict_list(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="t", description="d", parameters={}, handler=lambda a: None
            )
        )
        dicts = registry.to_dict_list()
        assert len(dicts) == 1
        assert dicts[0]["name"] == "t"


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


class TestBuiltinTools:
    def test_calculator_add(self):
        tool = create_calculator_tool()
        registry = ToolRegistry()
        registry.register(tool)
        result = registry.execute(
            ToolCall(
                tool_name="calculator",
                arguments={"operation": "add", "a": 2, "b": 3},
            )
        )
        assert result.is_success()
        assert result.result == 5.0

    def test_calculator_divide_by_zero(self):
        tool = create_calculator_tool()
        registry = ToolRegistry()
        registry.register(tool)
        result = registry.execute(
            ToolCall(
                tool_name="calculator",
                arguments={"operation": "divide", "a": 1, "b": 0},
            )
        )
        assert not result.is_success()
        assert "zero" in result.error.lower()

    def test_counter_tool(self):
        tool = create_counter_tool()
        registry = ToolRegistry()
        registry.register(tool)

        def run(action):
            return registry.execute(
                ToolCall(tool_name="counter", arguments={"action": action})
            )

        assert run("increment").result == 1
        assert run("increment").result == 2
        assert run("decrement").result == 1
        assert run("reset").result == 0

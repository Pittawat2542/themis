"""Tool use primitives for agentic workflows.

This module provides abstractions for defining and executing tools
(functions) that models can call during generation. This enables
agentic workflows, function calling, and tool-augmented generation.

Examples:
    # Define a tool
    def calculator(operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        raise ValueError(f"Unknown operation: {operation}")

    tool = ToolDefinition(
        name="calculator",
        description="Perform arithmetic operations",
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "multiply"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
        handler=calculator
    )

    # Register tool
    registry = ToolRegistry()
    registry.register(tool)

    # Execute tool
    call = ToolCall(tool_name="calculator", arguments={"operation": "add", "a": 2, "b": 3})
    result = registry.execute(call)
    print(result.result)  # 5.0
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """Defines a tool/function available to the model.

    Attributes:
        name: Tool name (should be unique)
        description: Human-readable description of what tool does
        parameters: JSON Schema describing parameters
        handler: Function to execute when tool is called
        metadata: Additional metadata
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert tool definition to dictionary (without handler).

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments against parameter schema.

        Args:
            arguments: Arguments to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Simple validation - check required fields
        if "required" in self.parameters:
            for field in self.parameters["required"]:
                if field not in arguments:
                    errors.append(f"Missing required field: {field}")

        # Check for unknown fields
        if "properties" in self.parameters:
            known_fields = set(self.parameters["properties"].keys())
            for field in arguments.keys():
                if field not in known_fields:
                    errors.append(f"Unknown field: {field}")

        return errors


@dataclass
class ToolCall:
    """Represents a request to execute a tool.

    Attributes:
        tool_name: Name of tool to execute
        arguments: Arguments to pass to tool
        call_id: Unique identifier for this call
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


@dataclass
class ToolResult:
    """Result from executing a tool.

    Attributes:
        call: Original tool call
        result: Result value (if successful)
        error: Error message (if failed)
        execution_time_ms: Time taken to execute (milliseconds)
        metadata: Additional metadata
    """

    call: ToolCall
    result: Any | None
    error: str | None
    execution_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if tool execution was successful.

        Returns:
            True if no error
        """
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "call": self.call.to_dict(),
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


class ToolRegistry:
    """Registry for managing and executing tools.

    This class maintains a registry of available tools and provides
    methods for registering, retrieving, and executing them.

    Examples:
        registry = ToolRegistry()

        # Register tools
        registry.register(calculator_tool)
        registry.register(search_tool)

        # Execute tool
        call = ToolCall(tool_name="calculator", arguments={...})
        result = registry.execute(call)
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool.

        Args:
            tool: Tool definition to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Tool name to unregister
        """
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """Get all registered tools.

        Returns:
            List of tool definitions
        """
        return list(self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call.

        Args:
            call: Tool call to execute

        Returns:
            ToolResult with execution result or error
        """
        tool = self._tools.get(call.tool_name)

        if tool is None:
            return ToolResult(
                call=call,
                result=None,
                error=f"Unknown tool: {call.tool_name}",
                execution_time_ms=0.0,
            )

        # Validate arguments
        validation_errors = tool.validate_arguments(call.arguments)
        if validation_errors:
            return ToolResult(
                call=call,
                result=None,
                error=f"Invalid arguments: {'; '.join(validation_errors)}",
                execution_time_ms=0.0,
            )

        # Execute tool
        start = time.perf_counter()
        try:
            result = tool.handler(call.arguments)
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                call=call,
                result=result,
                error=None,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                call=call,
                result=None,
                error=f"{e.__class__.__name__}: {str(e)}",
                execution_time_ms=elapsed,
            )

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Get all tools as dictionary list (for sending to model).

        Returns:
            List of tool definitions as dictionaries
        """
        return [tool.to_dict() for tool in self._tools.values()]


# Built-in tools for common use cases


def create_calculator_tool() -> ToolDefinition:
    """Create a basic calculator tool.

    Returns:
        ToolDefinition for calculator
    """

    def handler(args: dict[str, Any]) -> float:
        operation = args["operation"]
        a = float(args["a"])
        b = float(args["b"])

        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return ToolDefinition(
        name="calculator",
        description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        },
        handler=handler,
    )


def create_counter_tool() -> ToolDefinition:
    """Create a stateful counter tool for testing.

    Returns:
        ToolDefinition for counter
    """
    counter = {"value": 0}

    def handler(args: dict[str, Any]) -> int:
        action = args["action"]

        if action == "increment":
            counter["value"] += 1
        elif action == "decrement":
            counter["value"] -= 1
        elif action == "reset":
            counter["value"] = 0
        elif action == "get":
            pass  # Just return current value
        else:
            raise ValueError(f"Unknown action: {action}")

        return counter["value"]

    return ToolDefinition(
        name="counter",
        description="Simple counter that can be incremented, decremented, or reset",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["increment", "decrement", "reset", "get"],
                    "description": "Action to perform on counter",
                },
            },
            "required": ["action"],
        },
        handler=handler,
    )


__all__ = [
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "create_calculator_tool",
    "create_counter_tool",
]

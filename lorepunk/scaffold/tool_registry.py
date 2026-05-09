"""Tool Registry — the hands of the agent.

Defines the tool interface and manages registration. Each tool
has a name, description, parameter schema, and execution function.

The LLM sees the tool descriptions and schemas. When it decides
to use a tool, the scaffold dispatches to the right executor.

Pattern borrowed from Claude Code's tool architecture:
  - Tools are pure functions with typed inputs and outputs
  - Each tool has a JSON schema for parameter validation
  - Tools are sandboxed (file access restricted, bash configurable)
  - Results are returned as structured data
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """A single parameter for a tool."""
    name: str
    type: str  # string, integer, number, boolean, array
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """A tool the LLM can use."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: str = "general"  # general, file, code, web, marketing
    requires_confirmation: bool = False

    def to_schema(self) -> dict:
        """Generate JSON schema for LLM tool use."""
        props = {}
        required = []
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            props[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    data: Any = None

    def to_message(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}"


class ToolRegistry:
    """Registry of available tools.

    Tools are registered with their definitions and executors.
    The scaffold calls execute() with the tool name and arguments.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable[..., Awaitable[ToolResult]]] = {}

    def register(
        self,
        definition: ToolDefinition,
        executor: Callable[..., Awaitable[ToolResult]],
    ) -> None:
        self._tools[definition.name] = definition
        self._executors[definition.name] = executor
        logger.debug("Registered tool: %s", definition.name)

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self, category: str = "") -> list[ToolDefinition]:
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Get all tool schemas for the LLM system prompt."""
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        executor = self._executors.get(tool_name)
        if not executor:
            return ToolResult(
                tool_name=tool_name, success=False,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            result = await executor(**kwargs)
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return ToolResult(
                tool_name=tool_name, success=False,
                error=str(e),
            )

    @property
    def tool_count(self) -> int:
        return len(self._tools)

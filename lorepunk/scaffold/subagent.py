"""Subagent Spawner — parallel work through delegation.

When the main agent needs to do multiple things at once — research
competitors while drafting copy while analyzing metrics — it spawns
subagents. Each subagent gets:
  - A focused task description
  - A subset of tools (no file writes for research agents, etc.)
  - A separate conversation context
  - A timeout

The main agent collects results and synthesizes.

Pattern: Claude Code's Agent tool, adapted for local LLM execution.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from lorepunk.scaffold.tool_registry import ToolRegistry, ToolResult, ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)


@dataclass
class SubagentResult:
    """Result from a completed subagent."""
    agent_id: str
    task: str
    success: bool
    output: str
    duration_seconds: float = 0.0
    tool_calls_made: int = 0
    error: str = ""


@dataclass
class SubagentConfig:
    """Configuration for a subagent."""
    task: str
    model: str = ""  # defaults to parent's model
    api_base: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    timeout_seconds: int = 120
    max_turns: int = 5


class SubagentSpawner:
    """Spawn and manage subagents for parallel work.

    Args:
        parent_registry: The parent agent's tool registry
        default_model: Default model for subagents
        default_api_base: Default Ollama endpoint
    """

    MAX_CONCURRENT = 5

    def __init__(
        self,
        parent_registry: ToolRegistry,
        default_model: str = "",
        default_api_base: str = "http://localhost:11434",
    ) -> None:
        self._parent_registry = parent_registry
        self._default_model = default_model
        self._api_base = default_api_base
        self._active: dict[str, asyncio.Task] = {}
        self._results: list[SubagentResult] = []

    async def spawn(self, config: SubagentConfig) -> str:
        """Spawn a subagent. Returns agent_id."""
        if len(self._active) >= self.MAX_CONCURRENT:
            return ""

        agent_id = f"sub_{uuid.uuid4().hex[:8]}"
        model = config.model or self._default_model
        api_base = config.api_base or self._api_base

        task = asyncio.create_task(
            self._run_subagent(agent_id, config, model, api_base)
        )
        self._active[agent_id] = task
        logger.info("Spawned subagent %s: %s", agent_id, config.task[:60])
        return agent_id

    async def spawn_parallel(self, configs: list[SubagentConfig]) -> list[str]:
        """Spawn multiple subagents in parallel."""
        ids = []
        for config in configs:
            agent_id = await self.spawn(config)
            ids.append(agent_id)
        return ids

    async def wait(self, agent_id: str, timeout: int = 300) -> SubagentResult:
        """Wait for a specific subagent to complete."""
        task = self._active.get(agent_id)
        if not task:
            return SubagentResult(
                agent_id=agent_id, task="", success=False,
                error="Agent not found",
            )

        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            task.cancel()
            return SubagentResult(
                agent_id=agent_id, task="", success=False,
                error=f"Agent timed out after {timeout}s",
            )

    async def wait_all(self, agent_ids: list[str], timeout: int = 300) -> list[SubagentResult]:
        """Wait for all specified subagents to complete."""
        results = await asyncio.gather(
            *[self.wait(aid, timeout) for aid in agent_ids],
            return_exceptions=True,
        )
        return [
            r if isinstance(r, SubagentResult)
            else SubagentResult(agent_id="?", task="", success=False, error=str(r))
            for r in results
        ]

    async def _run_subagent(
        self, agent_id: str, config: SubagentConfig,
        model: str, api_base: str,
    ) -> SubagentResult:
        """Execute a subagent's task."""
        start = datetime.now(timezone.utc)

        try:
            import aiohttp
        except ImportError:
            return SubagentResult(
                agent_id=agent_id, task=config.task, success=False,
                error="aiohttp required",
            )

        system_prompt = (
            f"You are a focused subagent. Your task:\n{config.task}\n\n"
            f"Complete this task efficiently. Be concise in your response. "
            f"Report your findings clearly."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": config.task},
        ]

        tool_calls_made = 0
        final_response = ""

        for turn in range(config.max_turns):
            url = f"{api_base}/api/chat"
            tools = []
            if config.allowed_tools:
                tools = [
                    t.to_schema() for t in self._parent_registry.list_tools()
                    if t.name in config.allowed_tools
                ]

            payload = {
                "model": model,
                "messages": messages,
                "tools": tools if tools else None,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2048},
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
                    ) as resp:
                        data = await resp.json()

                message = data.get("message", {})

                if not message.get("tool_calls"):
                    final_response = message.get("content", "")
                    break

                messages.append(message)

                for tc in message.get("tool_calls", []):
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")

                    if not config.allowed_tools or tool_name not in config.allowed_tools:
                        messages.append({
                            "role": "tool",
                            "content": f"Tool {tool_name} not available to this subagent",
                        })
                        continue

                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                    result = await self._parent_registry.execute(tool_name, **args)
                    tool_calls_made += 1
                    messages.append({
                        "role": "tool",
                        "content": result.to_message(),
                    })

            except asyncio.TimeoutError:
                final_response = "(subagent timed out)"
                break
            except Exception as e:
                final_response = f"(subagent error: {e})"
                break

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        result = SubagentResult(
            agent_id=agent_id,
            task=config.task,
            success=bool(final_response and "error" not in final_response.lower()),
            output=final_response[:5000],
            duration_seconds=elapsed,
            tool_calls_made=tool_calls_made,
        )
        self._results.append(result)

        del self._active[agent_id]
        return result

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def completed_results(self) -> list[SubagentResult]:
        return list(self._results)


def register_subagent_tools(registry: ToolRegistry, spawner: SubagentSpawner) -> None:
    """Register subagent spawning as a tool the main agent can use."""

    async def spawn_agent(task: str, tools: str = "") -> ToolResult:
        allowed = [t.strip() for t in tools.split(",")] if tools else []
        config = SubagentConfig(task=task, allowed_tools=allowed)
        agent_id = await spawner.spawn(config)
        result = await spawner.wait(agent_id)
        return ToolResult(
            "spawn_agent", result.success,
            output=f"Subagent {agent_id} ({result.duration_seconds:.1f}s):\n{result.output}",
            error=result.error,
        )

    registry.register(
        ToolDefinition(
            name="spawn_agent",
            description="Spawn a subagent for a focused task. The subagent runs independently and returns results.",
            parameters=[
                ToolParameter("task", "string", "Detailed task description for the subagent"),
                ToolParameter("tools", "string",
                              "Comma-separated tool names the subagent can use (empty = no tools, text only)",
                              required=False, default=""),
            ],
            category="agents",
        ),
        spawn_agent,
    )

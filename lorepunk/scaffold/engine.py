"""Scaffold Engine — the brain-to-hands connection.

Takes an LLM backend (Ollama, OpenAI-compatible API, etc.)
and a tool registry, then runs the conversation loop:

  1. User sends message
  2. LLM sees message + tool schemas + conversation history
  3. LLM responds (text or tool call)
  4. If tool call: execute tool, feed result back to LLM
  5. Repeat until LLM responds with text (no more tools)
  6. Return final response to user

This is the Claude Code pattern — the LLM decides what to do,
the scaffold makes it happen.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from lorepunk.scaffold.tool_registry import ToolRegistry, ToolResult
from lorepunk.scaffold.cache_recorder import CacheDeltaRecorder

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # system, user, assistant, tool
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str = ""
    name: str = ""
    timestamp: str = ""

    def to_api(self) -> dict:
        """Convert to API message format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class EngineConfig:
    """Configuration for the scaffold engine."""
    model: str = "lorepunk:latest"
    api_base: str = "http://localhost:11434"
    api_type: str = "ollama"  # ollama, openai
    system_prompt: str = ""
    max_tool_rounds: int = 10
    temperature: float = 0.7
    max_tokens: int = 4096


class ScaffoldEngine:
    """The conversation + tool execution engine.

    Works with any OpenAI-compatible API (Ollama, vLLM, etc.)
    """

    def __init__(
        self,
        config: EngineConfig,
        registry: ToolRegistry,
        recorder: CacheDeltaRecorder | None = None,
    ) -> None:
        self.config = config
        self.recorder = recorder or CacheDeltaRecorder(model_name=config.model)
        self.registry = registry
        self.history: list[Message] = []

        if config.system_prompt:
            self.history.append(Message(
                role="system", content=config.system_prompt,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

    async def chat(self, user_message: str) -> str:
        """Send a message and get a response, executing tools as needed."""
        self.history.append(Message(
            role="user", content=user_message,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        for round_num in range(self.config.max_tool_rounds):
            response = await self._call_llm()

            if not response.get("tool_calls"):
                text = response.get("content") or ""
                # Handle Qwen3 thinking mode
                if "<think>" in text:
                    if "</think>" in text:
                        # Has both tags — extract content after thinking
                        think_end = text.index("</think>") + len("</think>")
                        after_think = text[think_end:].strip()
                        if after_think:
                            text = after_think
                        else:
                            # Everything was in think tags — extract the thinking as the response
                            think_start = text.index("<think>") + len("<think>")
                            think_content = text[think_start:text.index("</think>")].strip()
                            text = think_content
                    else:
                        # Unclosed think tag — model got cut off, strip the tag and use content
                        text = text.replace("<think>", "").strip()
                # If still empty after all processing
                if not text.strip():
                    text = "(The model returned an empty response. Try rephrasing your request.)"
                self.history.append(Message(
                    role="assistant", content=text,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                return text

            self.history.append(Message(
                role="assistant", content=response.get("content", ""),
                tool_calls=response["tool_calls"],
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

            for tool_call in response["tool_calls"]:
                fn = tool_call.get("function", {})
                tool_name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                logger.info("Tool call: %s(%s)", tool_name, list(args.keys()))
                result = await self.registry.execute(tool_name, **args)

                self.history.append(Message(
                    role="tool",
                    content=result.to_message(),
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

        return "(Max tool rounds reached. Please try a simpler request.)"

    async def _call_llm(self) -> dict:
        """Call the LLM with current history and tool schemas."""
        if self.config.api_type == "ollama":
            return await self._call_ollama()
        else:
            return await self._call_openai_compat()

    async def _call_ollama(self) -> dict:
        """Call Ollama's chat API with tool support."""
        try:
            import aiohttp
        except ImportError:
            return {"content": "aiohttp required — pip install aiohttp"}

        url = f"{self.config.api_base}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [m.to_api() for m in self.history],
            "tools": self.registry.get_schemas(),
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                    data = await resp.json()

            message = data.get("message", {})

            # Record telemetry + cache delta data (raw content before any stripping)
            user_msg = next((m.content for m in reversed(self.history) if m.role == "user"), "")
            tool_names = [tc.get("function", {}).get("name", "") for tc in (message.get("tool_calls") or [])]
            raw_content = message.get("content") or data.get("response") or ""
            self.recorder.record_ollama_turn(
                ollama_response=data,
                user_query=user_msg,
                response_text=raw_content,
                tool_calls=tool_names,
            )

            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls"),
            }
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            return {"content": f"LLM call failed: {e}"}

    async def _call_openai_compat(self) -> dict:
        """Call an OpenAI-compatible API (vLLM, LiteLLM, etc.)."""
        try:
            import aiohttp
        except ImportError:
            return {"content": "aiohttp required"}

        url = f"{self.config.api_base}/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [m.to_api() for m in self.history],
            "tools": self.registry.get_schemas(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                    data = await resp.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls"),
            }
        except Exception as e:
            logger.error("OpenAI-compat call failed: %s", e)
            return {"content": f"LLM call failed: {e}"}

    def clear_history(self) -> None:
        """Clear conversation history (keep system prompt)."""
        system = [m for m in self.history if m.role == "system"]
        self.history = system

    @property
    def message_count(self) -> int:
        return len(self.history)

"""Hook System — customizable behaviors without touching internals.

Hooks fire at key moments in the agent's lifecycle. Margaret can
add custom behaviors (post to Slack, validate brand voice, log
activity) without editing engine code.

Hook points:
  - session_start: agent boots up
  - session_stop: agent shuts down
  - pre_tool_call: before any tool executes (can block)
  - post_tool_call: after tool returns (logging, triggers)
  - pre_response: before response is shown to user (content checks)
  - post_response: after response delivered (analytics, memory)
  - on_error: when something goes wrong

Hooks can be:
  - Python callables (registered in code)
  - Shell commands (loaded from .lorepunk_hooks.json)
  - Async or sync (both supported)
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class HookEvent:
    """Data passed to hooks."""
    event_type: str
    timestamp: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    user_message: str = ""
    response: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class HookResult:
    """Result from a hook execution."""
    allow: bool = True  # False = block the action
    modified_content: str = ""  # non-empty = replace content
    message: str = ""  # feedback message


HookFn = Callable[[HookEvent], Awaitable[HookResult] | HookResult]


class HookRegistry:
    """Register and dispatch hooks.

    Usage:
        hooks = HookRegistry()

        @hooks.on("pre_tool_call")
        async def check_permissions(event):
            if event.tool_name == "bash" and "rm" in event.tool_args.get("command", ""):
                return HookResult(allow=False, message="Blocked: destructive command")
            return HookResult()

        hooks.on("post_response")(lambda e: log_to_slack(e.response))
    """

    VALID_EVENTS = {
        "session_start", "session_stop",
        "pre_tool_call", "post_tool_call",
        "pre_response", "post_response",
        "on_error",
    }

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookFn]] = {e: [] for e in self.VALID_EVENTS}
        self._shell_hooks: dict[str, list[str]] = {e: [] for e in self.VALID_EVENTS}

    def on(self, event_type: str) -> Callable:
        """Decorator to register a hook."""
        def decorator(fn: HookFn) -> HookFn:
            self.register(event_type, fn)
            return fn
        return decorator

    def register(self, event_type: str, fn: HookFn) -> None:
        if event_type not in self.VALID_EVENTS:
            raise ValueError(f"Invalid event: {event_type}. Valid: {self.VALID_EVENTS}")
        self._hooks[event_type].append(fn)

    def register_shell(self, event_type: str, command: str) -> None:
        if event_type not in self.VALID_EVENTS:
            raise ValueError(f"Invalid event: {event_type}")
        self._shell_hooks[event_type].append(command)

    async def fire(self, event: HookEvent) -> HookResult:
        """Fire all hooks for an event. Returns combined result."""
        combined = HookResult()

        for fn in self._hooks.get(event.event_type, []):
            try:
                result = fn(event)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, HookResult):
                    if not result.allow:
                        return result  # Short-circuit on block
                    if result.modified_content:
                        combined.modified_content = result.modified_content
                    if result.message:
                        combined.message = result.message
            except Exception as e:
                logger.warning("Hook failed for %s: %s", event.event_type, e)

        for cmd in self._shell_hooks.get(event.event_type, []):
            try:
                env = {
                    "HOOK_EVENT": event.event_type,
                    "HOOK_TOOL": event.tool_name,
                    "HOOK_USER_MSG": event.user_message[:500],
                    "HOOK_RESPONSE": event.response[:500],
                }
                proc = await asyncio.create_subprocess_shell(
                    cmd, env={**__import__("os").environ, **env},
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode != 0:
                    logger.warning("Shell hook failed: %s", stderr.decode()[:200])
            except Exception as e:
                logger.warning("Shell hook error: %s", e)

        return combined

    def load_from_file(self, workspace: str = ".") -> None:
        """Load shell hooks from .lorepunk_hooks.json."""
        path = Path(workspace) / ".lorepunk_hooks.json"
        if not path.exists():
            return

        try:
            config = json.loads(path.read_text())
            for event_type, commands in config.items():
                if event_type in self.VALID_EVENTS:
                    if isinstance(commands, str):
                        commands = [commands]
                    for cmd in commands:
                        self.register_shell(event_type, cmd)
            logger.info("Loaded %d shell hooks from %s",
                        sum(len(v) for v in self._shell_hooks.values()), path)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load hooks file: %s", e)

    @property
    def hook_count(self) -> int:
        return sum(len(v) for v in self._hooks.values()) + sum(len(v) for v in self._shell_hooks.values())


# ── Built-in hooks ──

def dangerous_command_guard(event: HookEvent) -> HookResult:
    """Block obviously dangerous commands before they execute."""
    if event.tool_name != "bash":
        return HookResult()

    cmd = event.tool_args.get("command", "")
    dangerous = [
        ("rm -rf /", "Cannot delete root filesystem"),
        ("rm -rf /*", "Cannot delete root filesystem"),
        ("mkfs", "Cannot format disks"),
        ("dd if=", "Cannot raw-write to devices"),
        ("> /dev/sda", "Cannot write to raw devices"),
        ("chmod -R 777 /", "Cannot open all permissions"),
    ]
    for pattern, reason in dangerous:
        if pattern in cmd:
            return HookResult(allow=False, message=f"Blocked: {reason}")

    return HookResult()


def log_tool_usage(event: HookEvent) -> HookResult:
    """Log all tool usage for audit trail."""
    logger.info("TOOL: %s(%s) → %s",
                event.tool_name,
                list(event.tool_args.keys()),
                "success" if "Error" not in event.tool_result else "error")
    return HookResult()

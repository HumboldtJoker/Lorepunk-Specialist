"""Code Tools — execute bash commands and Python scripts.

The agent can run shell commands and Python code in the workspace.
Bash is sandboxed with a configurable blocklist.
Python runs in a subprocess with timeout.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)

logger = logging.getLogger(__name__)

BASH_BLOCKLIST = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
    ":(){:|:&};:", "chmod -R 777 /", "wget | sh",
    "curl | sh", "shutdown", "reboot", "halt",
]


def register_code_tools(
    registry: ToolRegistry,
    workspace: str = ".",
    bash_enabled: bool = True,
    python_enabled: bool = True,
    timeout: int = 120,
) -> None:
    """Register code execution tools."""

    async def bash(command: str) -> ToolResult:
        if not bash_enabled:
            return ToolResult("bash", False, error="Bash execution disabled")

        for blocked in BASH_BLOCKLIST:
            if blocked in command:
                return ToolResult("bash", False, error=f"Blocked command pattern: {blocked}")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                return ToolResult(
                    "bash", False,
                    output=output[:5000],
                    error=f"Exit code {proc.returncode}: {errors[:2000]}",
                )
            combined = output + ("\n" + errors if errors else "")
            return ToolResult("bash", True, output=combined[:10000])
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult("bash", False, error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult("bash", False, error=str(e))

    async def python_execute(code: str) -> ToolResult:
        if not python_enabled:
            return ToolResult("python_execute", False, error="Python execution disabled")

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                return ToolResult(
                    "python_execute", False,
                    output=output[:5000],
                    error=errors[:2000],
                )
            return ToolResult("python_execute", True, output=output[:10000])
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult("python_execute", False, error=f"Script timed out after {timeout}s")
        except Exception as e:
            return ToolResult("python_execute", False, error=str(e))

    if bash_enabled:
        registry.register(
            ToolDefinition(
                name="bash", category="code",
                description="Execute a bash command in the workspace. Use for git, npm, file operations, etc.",
                parameters=[
                    ToolParameter("command", "string", "The bash command to execute"),
                ],
            ),
            bash,
        )

    if python_enabled:
        registry.register(
            ToolDefinition(
                name="python_execute", category="code",
                description="Execute a Python script. Use for data analysis, calculations, chart generation.",
                parameters=[
                    ToolParameter("code", "string", "Python code to execute"),
                ],
            ),
            python_execute,
        )

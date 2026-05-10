"""File Tools — read, write, edit files in the workspace.

Sandboxed to the project directory. The agent can read any file
in the workspace, write new files, and make targeted edits.
"""
from __future__ import annotations

import os
from pathlib import Path

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)


def register_file_tools(registry: ToolRegistry, workspace: str = ".") -> None:
    """Register file operation tools."""
    workspace_path = Path(workspace).resolve()

    def _validate_path(file_path: str) -> Path:
        resolved = Path(file_path).resolve()
        if not (resolved == workspace_path or str(resolved).startswith(str(workspace_path) + os.sep)):
            raise PermissionError(f"Access denied: {file_path} is outside workspace")
        return resolved

    async def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
        try:
            path = _validate_path(file_path)
            if not path.exists():
                return ToolResult("read_file", False, error=f"File not found: {file_path}")
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[offset:offset + limit]
            numbered = "\n".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(selected))
            return ToolResult("read_file", True, output=numbered)
        except PermissionError as e:
            return ToolResult("read_file", False, error=str(e))

    async def write_file(file_path: str, content: str) -> ToolResult:
        try:
            path = _validate_path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult("write_file", True, output=f"Written: {file_path} ({len(content)} chars)")
        except PermissionError as e:
            return ToolResult("write_file", False, error=str(e))

    async def edit_file(file_path: str, old_string: str, new_string: str) -> ToolResult:
        try:
            path = _validate_path(file_path)
            if not path.exists():
                return ToolResult("edit_file", False, error=f"File not found: {file_path}")
            text = path.read_text(encoding="utf-8")
            if old_string not in text:
                return ToolResult("edit_file", False, error="old_string not found in file")
            count = text.count(old_string)
            if count > 1:
                return ToolResult("edit_file", False, error=f"old_string appears {count} times — must be unique")
            text = text.replace(old_string, new_string, 1)
            path.write_text(text, encoding="utf-8")
            return ToolResult("edit_file", True, output=f"Edited: {file_path}")
        except PermissionError as e:
            return ToolResult("edit_file", False, error=str(e))

    async def list_files(directory: str = ".") -> ToolResult:
        try:
            path = _validate_path(directory)
            if not path.is_dir():
                return ToolResult("list_files", False, error=f"Not a directory: {directory}")
            entries = sorted(path.iterdir())
            lines = []
            for e in entries:
                if e.name.startswith("."):
                    continue
                prefix = "d" if e.is_dir() else "f"
                size = e.stat().st_size if e.is_file() else 0
                lines.append(f"[{prefix}] {e.name}" + (f" ({size:,} bytes)" if size else ""))
            return ToolResult("list_files", True, output="\n".join(lines) or "(empty directory)")
        except PermissionError as e:
            return ToolResult("list_files", False, error=str(e))

    registry.register(
        ToolDefinition(
            name="read_file", category="file",
            description="Read a file from the workspace. Returns numbered lines.",
            parameters=[
                ToolParameter("file_path", "string", "Path to the file"),
                ToolParameter("offset", "integer", "Line number to start from", required=False, default=0),
                ToolParameter("limit", "integer", "Number of lines to read", required=False, default=2000),
            ],
        ),
        read_file,
    )

    registry.register(
        ToolDefinition(
            name="write_file", category="file",
            description="Write content to a file. Creates directories as needed.",
            parameters=[
                ToolParameter("file_path", "string", "Path to write"),
                ToolParameter("content", "string", "Content to write"),
            ],
        ),
        write_file,
    )

    registry.register(
        ToolDefinition(
            name="edit_file", category="file",
            description="Replace a unique string in a file with new content.",
            parameters=[
                ToolParameter("file_path", "string", "Path to edit"),
                ToolParameter("old_string", "string", "Exact string to find (must be unique)"),
                ToolParameter("new_string", "string", "Replacement string"),
            ],
        ),
        edit_file,
    )

    registry.register(
        ToolDefinition(
            name="list_files", category="file",
            description="List files and directories in the workspace.",
            parameters=[
                ToolParameter("directory", "string", "Directory to list", required=False, default="."),
            ],
        ),
        list_files,
    )

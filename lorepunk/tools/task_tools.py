"""Task Tools — track work in progress.

Simple task/todo management within the workspace.
Stored as JSON, human-readable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)


@dataclass
class Task:
    id: int
    title: str
    status: str = "pending"  # pending, in_progress, completed, blocked
    created: str = ""
    notes: str = ""


TASKS_FILE = ".lorepunk_tasks.json"


def _load_tasks(workspace: str) -> list[Task]:
    path = Path(workspace) / TASKS_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [Task(**t) for t in data]
    except (json.JSONDecodeError, TypeError):
        return []


def _save_tasks(workspace: str, tasks: list[Task]) -> None:
    path = Path(workspace) / TASKS_FILE
    path.write_text(json.dumps([asdict(t) for t in tasks], indent=2))


def register_task_tools(registry: ToolRegistry, workspace: str = ".") -> None:
    """Register task management tools."""

    async def task_list() -> ToolResult:
        tasks = _load_tasks(workspace)
        if not tasks:
            return ToolResult("task_list", True, output="No tasks. Create one with task_create.")
        lines = []
        for t in tasks:
            icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]", "blocked": "[!]"}
            lines.append(f"{icon.get(t.status, '[ ]')} #{t.id}: {t.title}")
            if t.notes:
                lines.append(f"    {t.notes}")
        return ToolResult("task_list", True, output="\n".join(lines))

    async def task_create(title: str, notes: str = "") -> ToolResult:
        tasks = _load_tasks(workspace)
        next_id = max((t.id for t in tasks), default=0) + 1
        task = Task(id=next_id, title=title, notes=notes,
                    created=datetime.now(timezone.utc).isoformat())
        tasks.append(task)
        _save_tasks(workspace, tasks)
        return ToolResult("task_create", True, output=f"Created task #{next_id}: {title}")

    async def task_update(task_id: int, status: str = "", notes: str = "") -> ToolResult:
        tasks = _load_tasks(workspace)
        for t in tasks:
            if t.id == task_id:
                if status:
                    t.status = status
                if notes:
                    t.notes = notes
                _save_tasks(workspace, tasks)
                return ToolResult("task_update", True,
                                  output=f"Updated #{task_id}: status={t.status}")
        return ToolResult("task_update", False, error=f"Task #{task_id} not found")

    registry.register(
        ToolDefinition(name="task_list", description="List all tasks and their status.",
                       parameters=[], category="tasks"),
        task_list,
    )
    registry.register(
        ToolDefinition(name="task_create", description="Create a new task.",
                       parameters=[
                           ToolParameter("title", "string", "Task title"),
                           ToolParameter("notes", "string", "Optional notes", required=False, default=""),
                       ], category="tasks"),
        task_create,
    )
    registry.register(
        ToolDefinition(name="task_update", description="Update task status or notes.",
                       parameters=[
                           ToolParameter("task_id", "integer", "Task ID number"),
                           ToolParameter("status", "string", "New status: pending, in_progress, completed, blocked",
                                         required=False, default=""),
                           ToolParameter("notes", "string", "Updated notes", required=False, default=""),
                       ], category="tasks"),
        task_update,
    )

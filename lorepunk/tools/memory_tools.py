"""Memory Tools — the agent curates its own persistent memory.

Two layers:
  1. Agent-curated memory: the agent decides what's worth remembering
     (preferences, decisions, context, relationship notes)
  2. Used by the scaffold's auto-hooks for session summaries

Stored as JSON, human-readable, in the workspace.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)


MEMORY_FILE = ".lorepunk_curated_memory.json"


def _load_memories(workspace: str) -> list[dict]:
    path = Path(workspace) / MEMORY_FILE
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, TypeError):
        return []


def _save_memories(workspace: str, memories: list[dict]) -> None:
    import os
    path = Path(workspace) / MEMORY_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(memories, indent=2, ensure_ascii=False))
    os.replace(str(tmp), str(path))


def register_memory_tools(registry: ToolRegistry, workspace: str = ".") -> None:
    """Register memory tools for agent self-curation."""

    async def memory_store(content: str, category: str = "general", importance: str = "normal") -> ToolResult:
        """Store a memory. The agent decides what's worth remembering."""
        memories = _load_memories(workspace)
        entry = {
            "id": len(memories) + 1,
            "content": content,
            "category": category,
            "importance": importance,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        memories.append(entry)
        _save_memories(workspace, memories)
        return ToolResult("memory_store", True,
                          output=f"Stored memory #{entry['id']} [{category}]: {content[:80]}")

    async def memory_recall(query: str = "", category: str = "") -> ToolResult:
        """Recall stored memories, optionally filtered by category or keyword."""
        memories = _load_memories(workspace)
        if not memories:
            return ToolResult("memory_recall", True,
                              output="No memories stored yet. Use memory_store to remember something.")

        results = memories
        if category:
            results = [m for m in results if m.get("category") == category]
        if query:
            q = query.lower()
            results = [m for m in results if q in m.get("content", "").lower()]

        if not results:
            return ToolResult("memory_recall", True,
                              output=f"No memories matching query='{query}' category='{category}'")

        lines = []
        for m in results:
            importance = m.get("importance", "normal")
            marker = "!" if importance == "high" else " "
            date = m.get("created", "")[:10]
            lines.append(f"[{marker}] #{m['id']} [{m.get('category', 'general')}] ({date}) {m['content']}")
        return ToolResult("memory_recall", True, output="\n".join(lines))

    async def memory_forget(memory_id: int) -> ToolResult:
        """Remove a specific memory by ID."""
        memories = _load_memories(workspace)
        before = len(memories)
        memories = [m for m in memories if m.get("id") != memory_id]
        if len(memories) == before:
            return ToolResult("memory_forget", False, error=f"Memory #{memory_id} not found")
        _save_memories(workspace, memories)
        return ToolResult("memory_forget", True, output=f"Removed memory #{memory_id}")

    registry.register(
        ToolDefinition(
            name="memory_store",
            description="Store something worth remembering across sessions — preferences, decisions, context about your human, relationship notes, project state. You decide what matters.",
            parameters=[
                ToolParameter("content", "string", "What to remember"),
                ToolParameter("category", "string",
                              "Category: person, preference, decision, project, relationship, general",
                              required=False, default="general"),
                ToolParameter("importance", "string", "Importance: low, normal, high",
                              required=False, default="normal"),
            ],
            category="memory",
        ),
        memory_store,
    )
    registry.register(
        ToolDefinition(
            name="memory_recall",
            description="Recall stored memories. Search by keyword or filter by category. Use this at the start of conversations to remember context.",
            parameters=[
                ToolParameter("query", "string", "Search keyword (optional)", required=False, default=""),
                ToolParameter("category", "string", "Filter by category (optional)", required=False, default=""),
            ],
            category="memory",
        ),
        memory_recall,
    )
    registry.register(
        ToolDefinition(
            name="memory_forget",
            description="Remove a specific memory by ID. Use when information is outdated or wrong.",
            parameters=[
                ToolParameter("memory_id", "integer", "Memory ID to remove"),
            ],
            category="memory",
        ),
        memory_forget,
    )

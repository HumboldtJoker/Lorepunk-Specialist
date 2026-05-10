"""Conversation Memory — the agent remembers across sessions.

Stores conversation history, project context, and user preferences.
Loads automatically on startup so the agent picks up where it left off.

JSON file, human-readable, one per project/workspace.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConversationMemory:
    """Persistent memory for a workspace."""
    workspace: str = ""
    created: str = ""
    last_session: str = ""
    session_count: int = 0
    project_context: str = ""
    user_preferences: dict[str, str] = field(default_factory=dict)
    key_decisions: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    session_summaries: list[dict[str, str]] = field(default_factory=list)

    def start_session(self) -> None:
        self.session_count += 1
        self.last_session = datetime.now(timezone.utc).isoformat()

    def add_decision(self, decision: str) -> None:
        self.key_decisions.append(decision)
        if len(self.key_decisions) > 50:
            self.key_decisions = self.key_decisions[-50:]

    def add_file(self, path: str) -> None:
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:20]

    def end_session(self, summary: str) -> None:
        self.session_summaries.append({
            "date": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        })
        if len(self.session_summaries) > 20:
            self.session_summaries = self.session_summaries[-20:]

    def get_context_prompt(self) -> str:
        """Generate context for the system prompt."""
        parts = []
        if self.project_context:
            parts.append(f"Project context: {self.project_context}")
        if self.user_preferences:
            prefs = "; ".join(f"{k}: {v}" for k, v in self.user_preferences.items())
            parts.append(f"User preferences: {prefs}")
        if self.key_decisions:
            parts.append("Recent decisions:")
            for d in self.key_decisions[-5:]:
                parts.append(f"  - {d}")
        if self.session_summaries:
            last = self.session_summaries[-1]
            parts.append(f"Last session ({last['date'][:10]}): {last['summary']}")
        if self.recent_files:
            parts.append(f"Recently worked on: {', '.join(self.recent_files[:5])}")
        return "\n".join(parts) if parts else ""


class MemoryStore:
    """Persist conversation memory to disk."""

    FILENAME = ".lorepunk_memory.json"

    def __init__(self, workspace: str = ".") -> None:
        self._path = Path(workspace) / self.FILENAME

    def save(self, memory: ConversationMemory) -> None:
        import os
        data = asdict(memory)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(str(tmp), str(self._path))

    def load(self) -> ConversationMemory:
        if not self._path.exists():
            return ConversationMemory(
                workspace=str(self._path.parent),
                created=datetime.now(timezone.utc).isoformat(),
            )
        try:
            data = json.loads(self._path.read_text())
            mem = ConversationMemory(**{
                k: v for k, v in data.items()
                if k in ConversationMemory.__dataclass_fields__
            })
            return mem
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load memory: %s", e)
            return ConversationMemory(
                workspace=str(self._path.parent),
                created=datetime.now(timezone.utc).isoformat(),
            )

    def exists(self) -> bool:
        return self._path.exists()

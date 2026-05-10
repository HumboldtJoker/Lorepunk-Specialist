"""Conversation Compaction — keep context fresh, not stale.

Long conversations accumulate context. Eventually the LLM's
context window fills up and quality degrades. Compaction
summarizes earlier messages while preserving recent detail.

Strategy:
  - Keep system prompt (always)
  - Keep last N messages verbatim (recent context)
  - Summarize everything before that into a compact narrative
  - Preserve tool results that produced lasting artifacts (files, commits)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompactionConfig:
    keep_recent: int = 20  # messages to keep verbatim
    summary_max_tokens: int = 500
    compact_after: int = 40  # trigger compaction at this many messages


def compact_history(messages: list[dict], config: CompactionConfig | None = None) -> list[dict]:
    """Compact conversation history.

    Returns a new message list with older messages summarized.
    """
    cfg = config or CompactionConfig()

    if len(messages) <= cfg.compact_after:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= cfg.keep_recent:
        return messages

    to_summarize = non_system[:-cfg.keep_recent]
    to_keep = non_system[-cfg.keep_recent:]

    summary_parts = []
    files_created = []
    commits_made = []

    for msg in to_summarize:
        role = msg.get("role", "")
        content = msg.get("content", "")[:200]

        if role == "user":
            summary_parts.append(f"User asked: {content}")
        elif role == "assistant" and content:
            summary_parts.append(f"Agent: {content}")
        elif role == "tool":
            name = msg.get("name", "tool")
            if "written" in content.lower() or "created" in content.lower():
                files_created.append(content[:100])
            if "commit" in content.lower():
                commits_made.append(content[:100])

    summary = "Earlier in this conversation:\n"
    summary += "\n".join(f"- {p}" for p in summary_parts[:15])

    if files_created:
        summary += "\n\nFiles created/modified:\n"
        summary += "\n".join(f"- {f}" for f in files_created[:10])

    if commits_made:
        summary += "\n\nCommits made:\n"
        summary += "\n".join(f"- {c}" for c in commits_made[:5])

    compacted_msg = {
        "role": "system",
        "content": f"[Conversation compacted — {len(to_summarize)} messages summarized]\n\n{summary}",
    }

    return system + [compacted_msg] + to_keep

"""Transcript Recorder — verbatim conversation capture.

Saves every message (user, assistant, tool) as full-content JSONL.
One file per session. No truncation, no summarization — the complete record.

Separate from telemetry (cache_recorder.py) which captures inference metrics.
Separate from curated memory (memory_tools.py) which the agent manages.
This is the raw tape.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptRecorder:
    """Record full conversation transcripts to JSONL."""

    def __init__(self, output_dir: str = "") -> None:
        self._dir = Path(output_dir) if output_dir else Path.home() / ".lorepunk" / "transcripts"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._file = self._dir / f"transcript_{self._session_id}.jsonl"
        self._turn = 0

    def record(self, role: str, content: str, tool_name: str = "", tool_calls: list | None = None) -> None:
        """Record a single conversation turn."""
        self._turn += 1
        entry = {
            "turn": self._turn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session": self._session_id,
            "role": role,
            "content": content,
        }
        if tool_name:
            entry["tool_name"] = tool_name
        if tool_calls:
            entry["tool_calls"] = tool_calls
        try:
            with open(self._file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write transcript: %s", e)

    @property
    def transcript_file(self) -> Path:
        return self._file

    @property
    def session_id(self) -> str:
        return self._session_id

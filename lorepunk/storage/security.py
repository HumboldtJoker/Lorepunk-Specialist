"""Security utilities for Oracle Harness memory layer.

Validates identifiers and paths to prevent traversal and injection.
"""

import re

# Valid identifier: hex chars, underscores, hyphens, dots. No path separators.
_SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
_HEX_PATTERN = re.compile(r'^[a-fA-F0-9]+$')

MAX_DELTA_DEPTH = 50


def validate_snapshot_id(snapshot_id: str) -> str:
    """Validate and sanitize a snapshot ID for use in file paths."""
    if not snapshot_id or not _SAFE_ID_PATTERN.match(snapshot_id):
        raise ValueError(f"Invalid snapshot_id: {snapshot_id!r}")
    if '..' in snapshot_id or '/' in snapshot_id or '\\' in snapshot_id:
        raise ValueError(f"Path traversal in snapshot_id: {snapshot_id!r}")
    return snapshot_id


def validate_content_hash(content_hash: str) -> str:
    """Validate a content hash is hex-only."""
    if not content_hash or not _HEX_PATTERN.match(content_hash):
        raise ValueError(f"Invalid content_hash: {content_hash!r}")
    return content_hash

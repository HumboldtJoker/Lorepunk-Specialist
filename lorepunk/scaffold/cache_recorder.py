"""Cache Delta Recorder — every conversation is a research opportunity.

Records inference telemetry and, when available, KV cache geometry
deltas during generation. When the model thinks, we watch HOW it
thinks — not what it says, but the shape of its processing.

Two modes:
  1. Ollama mode: records token counts, generation speed, eval timing,
     prompt/generation token ratio. Available now.
  2. Transformers mode: records full SVD spectral features of the KV
     cache at encoding and generation phases. Computes the delta.
     Available when running via transformers backend.

The delta is the key metric from our Oracle Loop research:
  - Honest generation: delta stable_rank ~0.44 (cache expands)
  - Confabulation: delta stable_rank ~0.03 (cache stays flat)
  - The model's geometry KNOWS when it's fabricating

All recordings are stored as JSONL — one line per inference turn.
Lyra can load and analyze them offline.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceTelemetry:
    """Telemetry from a single inference turn."""
    timestamp: str
    model: str
    turn_index: int

    # Token counts
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0

    # Timing
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Tool use
    tool_calls: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)

    # Cache geometry (when available via transformers backend)
    encoding_geometry: dict[str, float] | None = None
    generation_geometry: dict[str, float] | None = None
    delta_geometry: dict[str, float] | None = None
    per_layer_deltas: list[dict[str, float]] | None = None

    # Metadata
    user_query_preview: str = ""
    response_preview: str = ""
    conversation_id: str = ""


@dataclass
class GeometrySnapshot:
    """SVD spectral features at a point in time.

    These are the features from our Oracle Loop research.
    Only populated when running via transformers backend.
    """
    stable_rank: float = 0.0
    spectral_entropy: float = 0.0
    sv_kurtosis: float = 0.0
    mp_signal_fraction: float = 0.0
    mp_spectral_gap: float = 0.0
    mp_norm_per_token: float = 0.0
    mp_outlier_count: float = 0.0
    gd_signal_rank: float = 0.0
    participation_ratio: float = 0.0
    condition_number: float = 0.0
    n_tokens: int = 0
    n_layers: int = 0


class CacheDeltaRecorder:
    """Record inference telemetry and cache geometry deltas.

    Args:
        output_dir: Directory for JSONL recording files
        model_name: Model identifier for records
        record_geometry: Whether to attempt cache geometry extraction
    """

    def __init__(
        self,
        output_dir: str = "",
        model_name: str = "",
        record_geometry: bool = False,
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir else Path.home() / ".lorepunk" / "recordings"
        self._model = model_name
        self._record_geometry = record_geometry
        self._turn_index = 0
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._output_dir / f"session_{self._session_id}.jsonl"

    def record_ollama_turn(
        self,
        ollama_response: dict,
        user_query: str = "",
        response_text: str = "",
        tool_calls: list[str] | None = None,
    ) -> InferenceTelemetry:
        """Record telemetry from an Ollama API response.

        Extracts available metrics from Ollama's response format.
        """
        self._turn_index += 1
        now = datetime.now(timezone.utc).isoformat()

        prompt_eval_ns = ollama_response.get("prompt_eval_duration", 0)
        eval_ns = ollama_response.get("eval_duration", 0)
        total_ns = ollama_response.get("total_duration", 0)
        eval_count = ollama_response.get("eval_count", 0)
        prompt_count = ollama_response.get("prompt_eval_count", 0)

        tps = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0.0

        telemetry = InferenceTelemetry(
            timestamp=now,
            model=self._model,
            turn_index=self._turn_index,
            prompt_tokens=prompt_count,
            generated_tokens=eval_count,
            total_tokens=prompt_count + eval_count,
            prompt_eval_duration_ms=prompt_eval_ns / 1e6,
            eval_duration_ms=eval_ns / 1e6,
            total_duration_ms=total_ns / 1e6,
            tokens_per_second=tps,
            tool_calls=tool_calls or [],
            user_query_preview=user_query[:200],
            response_preview=response_text[:200],
            conversation_id=self._session_id,
        )

        self._write(telemetry)
        return telemetry

    def record_transformers_turn(
        self,
        encoding_features: dict[str, float],
        generation_features: dict[str, float],
        per_layer: list[dict[str, float]] | None = None,
        user_query: str = "",
        response_text: str = "",
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> InferenceTelemetry:
        """Record telemetry with full cache geometry from transformers.

        This is the research-grade recording that captures SVD features
        at encoding and generation phases, plus the delta.
        """
        self._turn_index += 1
        now = datetime.now(timezone.utc).isoformat()

        # Compute delta
        delta = {}
        for key in encoding_features:
            if key in generation_features:
                delta[key] = generation_features[key] - encoding_features[key]

        # Per-layer deltas if available
        per_layer_deltas = None
        if per_layer:
            per_layer_deltas = per_layer

        telemetry = InferenceTelemetry(
            timestamp=now,
            model=self._model,
            turn_index=self._turn_index,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=prompt_tokens + generated_tokens,
            total_duration_ms=duration_ms,
            tokens_per_second=generated_tokens / (duration_ms / 1000) if duration_ms > 0 else 0,
            encoding_geometry=encoding_features,
            generation_geometry=generation_features,
            delta_geometry=delta,
            per_layer_deltas=per_layer_deltas,
            user_query_preview=user_query[:200],
            response_preview=response_text[:200],
            conversation_id=self._session_id,
        )

        self._write(telemetry)
        return telemetry

    def _write(self, telemetry: InferenceTelemetry) -> None:
        """Append telemetry to JSONL file."""
        try:
            with open(self._file, "a") as f:
                f.write(json.dumps(asdict(telemetry), default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write telemetry: %s", e)

    def session_summary(self) -> dict:
        """Generate summary statistics for the current session."""
        if not self._file.exists():
            return {"turns": 0}

        turns = []
        for line in self._file.read_text().splitlines():
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not turns:
            return {"turns": 0}

        total_tokens = sum(t.get("total_tokens", 0) for t in turns)
        total_gen = sum(t.get("generated_tokens", 0) for t in turns)
        tps_values = [t.get("tokens_per_second", 0) for t in turns if t.get("tokens_per_second", 0) > 0]
        has_geometry = any(t.get("delta_geometry") for t in turns)

        deltas = [t.get("delta_geometry", {}) for t in turns if t.get("delta_geometry")]
        flat_count = sum(1 for d in deltas if abs(d.get("stable_rank", 1)) < 0.05) if deltas else 0

        return {
            "session_id": self._session_id,
            "turns": len(turns),
            "total_tokens": total_tokens,
            "total_generated": total_gen,
            "avg_tokens_per_second": sum(tps_values) / len(tps_values) if tps_values else 0,
            "has_geometry": has_geometry,
            "geometry_turns": len(deltas),
            "flat_delta_count": flat_count,
            "recording_file": str(self._file),
        }

    @property
    def recording_file(self) -> Path:
        return self._file

    @property
    def turn_count(self) -> int:
        return self._turn_index

"""Live Geometry Hook — records embedding geometry on every conversation turn.

Runs asynchronously after each assistant response, so it doesn't block
the conversation. Uses Ollama's embedding endpoint (lightweight, no
model reload needed).

Wired into the scaffold engine as a post-turn callback.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class LiveGeometryRecorder:
    """Record embedding geometry on every conversation turn.

    Non-blocking: fires and forgets the geometry computation
    so the user doesn't wait for SVD on every message.
    """

    def __init__(
        self,
        model: str = "qwen3:235b-a22b",
        api_base: str = "http://localhost:11434",
        output_dir: str = "",
    ) -> None:
        self.model = model
        self.api_base = api_base
        self._dir = Path(output_dir) if output_dir else Path.home() / ".lorepunk" / "geometry"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._file = self._dir / f"live_{self._session}.jsonl"
        self._turn = 0
        self._enabled = True

    async def record_turn(self, user_message: str, assistant_response: str) -> None:
        """Record geometry for a conversation turn (non-blocking)."""
        if not self._enabled:
            return
        try:
            asyncio.create_task(self._record(user_message, assistant_response))
        except Exception as e:
            logger.debug("Geometry recording skipped: %s", e)

    async def _record(self, user_message: str, assistant_response: str) -> None:
        """Actual recording — runs as a background task."""
        try:
            import aiohttp
            import numpy as np
        except ImportError:
            self._enabled = False
            logger.info("Geometry recording disabled: aiohttp/numpy not available")
            return

        self._turn += 1

        try:
            async with aiohttp.ClientSession() as session:
                # Encoding phase: embed the user message
                enc_resp = await session.post(
                    f"{self.api_base}/api/embeddings",
                    json={"model": self.model, "prompt": user_message},
                    timeout=aiohttp.ClientTimeout(total=60),
                )
                enc_data = await enc_resp.json()
                enc_emb = enc_data.get("embedding", [])

                # Generation phase: embed user + assistant together
                full_text = user_message + "\n" + assistant_response
                gen_resp = await session.post(
                    f"{self.api_base}/api/embeddings",
                    json={"model": self.model, "prompt": full_text},
                    timeout=aiohttp.ClientTimeout(total=60),
                )
                gen_data = await gen_resp.json()
                gen_emb = gen_data.get("embedding", [])

            if not enc_emb or not gen_emb:
                return

            enc_arr = np.array(enc_emb)
            gen_arr = np.array(gen_emb)

            # Compute basic geometry from embedding vectors
            enc_norm = np.linalg.norm(enc_arr)
            gen_norm = np.linalg.norm(gen_arr)
            cosine_sim = float(np.dot(enc_arr, gen_arr) / (enc_norm * gen_norm)) if enc_norm > 0 and gen_norm > 0 else 0

            # Effective dimensionality via squared entries
            enc_sq = enc_arr**2
            enc_sq_norm = enc_sq / enc_sq.sum() if enc_sq.sum() > 0 else enc_sq
            enc_entropy = float(-np.sum(enc_sq_norm * np.log(enc_sq_norm + 1e-10)))

            gen_sq = gen_arr**2
            gen_sq_norm = gen_sq / gen_sq.sum() if gen_sq.sum() > 0 else gen_sq
            gen_entropy = float(-np.sum(gen_sq_norm * np.log(gen_sq_norm + 1e-10)))

            # Delta between encoding and generation embedding
            delta_vec = gen_arr - enc_arr
            delta_norm = float(np.linalg.norm(delta_vec))
            delta_entropy = gen_entropy - enc_entropy

            entry = {
                "turn": self._turn,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": self._session,
                "user_preview": user_message[:150],
                "response_preview": assistant_response[:150],
                "embedding_dim": len(enc_emb),
                "encoding_norm": float(enc_norm),
                "generation_norm": float(gen_norm),
                "cosine_similarity": cosine_sim,
                "delta_norm": delta_norm,
                "encoding_entropy": enc_entropy,
                "generation_entropy": gen_entropy,
                "delta_entropy": delta_entropy,
            }

            with open(self._file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as e:
            logger.debug("Geometry recording failed: %s", e)

    @property
    def geometry_file(self) -> Path:
        return self._file

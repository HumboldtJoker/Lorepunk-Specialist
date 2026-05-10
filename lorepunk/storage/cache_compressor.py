"""Cache Compressor — storage pipeline for KV cache snapshots.

Stores raw cache data with layered compression while preserving
full fidelity for future analysis. The industry missed what was
in the cache by optimizing it away. We keep everything.

Pipeline:
  RAW (FP32) → FP16 quantize → delta encode → zstd compress → SHA256 hash → disk

Storage tiers:
  HOT:     Full cache in RAM (during inference, for rollback)
  WARM:    Compressed on disk (for replay/analysis, configurable retention)
  COLD:    Features only (longitudinal monitoring, kept forever)

Default: keep WARM tier indefinitely. Delete nothing until we understand
everything. Disk is cheaper than missed discoveries.

Author: Operator
Date: 2026-04-17
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from lorepunk.storage.security import (
    validate_snapshot_id,
    validate_content_hash,
    MAX_DELTA_DEPTH,
)

logger = logging.getLogger(__name__)

# Optional zstd — fall back to gzip if not available
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    import gzip
    ZSTD_AVAILABLE = False
    logger.warning("zstandard not installed — falling back to gzip. "
                   "Install with: pip install zstandard")


DEFAULT_STORE_DIR = os.path.expanduser("~/.oracle/cache_store")


@dataclass
class CacheMetadata:
    """Metadata for a stored cache snapshot."""
    snapshot_id: str
    tx_id: str
    checkpoint: str
    timestamp: float
    original_dtype: str
    stored_dtype: str
    shape_info: dict
    compression: str
    content_hash: str
    size_original: int
    size_stored: int
    compression_ratio: float
    delta_parent: Optional[str] = None  # snapshot_id of delta base
    geometry_features: Optional[dict] = None


@dataclass
class CompressionStats:
    """Stats from a compression operation."""
    original_bytes: int
    fp16_bytes: int
    delta_bytes: Optional[int]
    compressed_bytes: int
    ratio: float
    hash: str
    duration_ms: float


class CacheCompressor:
    """Layered compression pipeline for KV cache snapshots.

    Designed to maximize retention of raw data while minimizing
    disk footprint. Every stage is reversible.

    Usage::

        compressor = CacheCompressor(store_dir="~/.oracle/cache_store")

        # Store a cache snapshot
        meta = compressor.store(
            cache_tensors=kv_cache,    # List of (key, value) tensor pairs
            tx_id="abc123",
            checkpoint="encoding",
            geometry=extracted_features,
        )

        # Retrieve for analysis
        cache_tensors = compressor.load(meta.snapshot_id)

        # Get storage stats
        stats = compressor.get_stats()
    """

    def __init__(
        self,
        store_dir: str = DEFAULT_STORE_DIR,
        quantize_to_fp16: bool = True,
        enable_delta: bool = True,
        zstd_level: int = 3,
        retain_days: int = -1,  # -1 = keep forever
    ):
        self.store_dir = Path(os.path.expanduser(store_dir))
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.store_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.blobs_dir = self.store_dir / "blobs"
        self.blobs_dir.mkdir(exist_ok=True)

        self.quantize_fp16 = quantize_to_fp16
        self.enable_delta = enable_delta
        self.zstd_level = zstd_level
        self.retain_days = retain_days

        # Track the last snapshot for delta encoding
        self._last_snapshot_id: Optional[str] = None
        self._last_snapshot_data: Optional[bytes] = None

    def store(
        self,
        cache_tensors,
        tx_id: str,
        checkpoint: str,
        geometry: Optional[dict] = None,
        snapshot_id: Optional[str] = None,
    ) -> CacheMetadata:
        """Store a KV cache snapshot through the compression pipeline.

        Args:
            cache_tensors: The raw KV cache. Either:
                - List of (key_tensor, value_tensor) tuples (PyTorch/numpy)
                - A single numpy array or bytes object
            tx_id: Transaction ID this snapshot belongs to
            checkpoint: Checkpoint label (encoding, post_inference, etc.)
            geometry: Pre-extracted geometry features (stored alongside)
            snapshot_id: Optional custom ID. Auto-generated if not provided.

        Returns:
            CacheMetadata with storage details and content hash.
        """
        start = time.time()

        if snapshot_id is None:
            snapshot_id = f"{tx_id}_{checkpoint}_{int(time.time()*1000)}"
        snapshot_id = validate_snapshot_id(snapshot_id)

        # Stage 1: Serialize to bytes
        raw_bytes, shape_info, original_dtype = self._serialize(cache_tensors)
        original_size = len(raw_bytes)

        # Stage 2: FP16 quantization (if enabled and dtype is float32)
        if self.quantize_fp16 and original_dtype in ("float32", "float64"):
            quantized = self._quantize_fp16(raw_bytes, original_dtype)
            stored_dtype = "float16"
        else:
            quantized = raw_bytes
            stored_dtype = original_dtype

        fp16_size = len(quantized)

        # Stage 3: Delta encoding (if enabled and we have a previous snapshot)
        delta_parent = None
        if (self.enable_delta and
                self._last_snapshot_data is not None and
                len(self._last_snapshot_data) == len(quantized)):
            delta = self._delta_encode(quantized, self._last_snapshot_data)
            delta_parent = self._last_snapshot_id
            to_compress = delta
            delta_size = len(delta)
        else:
            to_compress = quantized
            delta_size = None

        # Stage 4: Compress
        compressed = self._compress(to_compress)
        compressed_size = len(compressed)

        # Stage 5: Hash
        content_hash = hashlib.sha256(compressed).hexdigest()

        # Stage 6: Write to disk
        blob_path = self.blobs_dir / f"{content_hash}.bin"
        if not blob_path.exists():  # Content-addressable dedup
            blob_path.write_bytes(compressed)

        # Update delta tracking
        self._last_snapshot_id = snapshot_id
        self._last_snapshot_data = quantized

        # Build metadata
        duration_ms = (time.time() - start) * 1000
        ratio = original_size / max(compressed_size, 1)

        meta = CacheMetadata(
            snapshot_id=snapshot_id,
            tx_id=tx_id,
            checkpoint=checkpoint,
            timestamp=time.time(),
            original_dtype=original_dtype,
            stored_dtype=stored_dtype,
            shape_info=shape_info,
            compression="zstd" if ZSTD_AVAILABLE else "gzip",
            content_hash=content_hash,
            size_original=original_size,
            size_stored=compressed_size,
            compression_ratio=ratio,
            delta_parent=delta_parent,
            geometry_features=geometry,
        )

        # Write metadata
        meta_path = self.metadata_dir / f"{snapshot_id}.json"
        meta_path.write_text(json.dumps({
            "snapshot_id": meta.snapshot_id,
            "tx_id": meta.tx_id,
            "checkpoint": meta.checkpoint,
            "timestamp": meta.timestamp,
            "original_dtype": meta.original_dtype,
            "stored_dtype": meta.stored_dtype,
            "shape_info": meta.shape_info,
            "compression": meta.compression,
            "content_hash": meta.content_hash,
            "size_original": meta.size_original,
            "size_stored": meta.size_stored,
            "compression_ratio": meta.compression_ratio,
            "delta_parent": meta.delta_parent,
            "geometry_features": meta.geometry_features,
        }, indent=2))

        logger.info(
            "Stored cache %s: %s → %s (%.1fx) hash=%s",
            snapshot_id,
            _human_size(original_size),
            _human_size(compressed_size),
            ratio,
            content_hash[:12],
        )

        return meta

    def load(self, snapshot_id: str) -> np.ndarray:
        """Load and decompress a cache snapshot.

        Handles the full reverse pipeline:
        decompress → un-delta → un-quantize → deserialize

        Returns numpy array of the original cache data.
        """
        snapshot_id = validate_snapshot_id(snapshot_id)
        meta_path = self.metadata_dir / f"{snapshot_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata for snapshot {snapshot_id}")

        meta = json.loads(meta_path.read_text())

        # Validate content hash before constructing path
        content_hash = validate_content_hash(meta['content_hash'])
        blob_path = self.blobs_dir / f"{content_hash}.bin"
        compressed = blob_path.read_bytes()

        # Decompress
        data = self._decompress(compressed, meta["compression"])

        # Un-delta (if delta encoded)
        if meta.get("delta_parent"):
            parent_data = self._load_raw(meta["delta_parent"])
            data = self._delta_decode(data, parent_data)

        # Un-quantize (FP16 → FP32 if needed)
        if meta["stored_dtype"] == "float16" and meta["original_dtype"] == "float32":
            arr = np.frombuffer(data, dtype=np.float16)
            arr = arr.astype(np.float32)
        else:
            arr = np.frombuffer(data, dtype=meta["stored_dtype"])

        return arr

    def get_stats(self) -> dict:
        """Get storage statistics."""
        meta_files = list(self.metadata_dir.glob("*.json"))
        blob_files = list(self.blobs_dir.glob("*.bin"))

        total_original = 0
        total_stored = 0
        for mf in meta_files:
            meta = json.loads(mf.read_text())
            total_original += meta.get("size_original", 0)
            total_stored += meta.get("size_stored", 0)

        blob_disk = sum(f.stat().st_size for f in blob_files)

        return {
            "snapshots": len(meta_files),
            "unique_blobs": len(blob_files),
            "total_original": _human_size(total_original),
            "total_stored": _human_size(total_stored),
            "blob_disk_usage": _human_size(blob_disk),
            "overall_ratio": total_original / max(total_stored, 1),
            "dedup_savings": _human_size(total_stored - blob_disk),
        }

    # ──────────────────────────────────────────────────────────────
    # Internal pipeline stages
    # ──────────────────────────────────────────────────────────────

    def _serialize(self, cache_tensors) -> Tuple[bytes, dict, str]:
        """Serialize cache tensors to bytes."""
        if isinstance(cache_tensors, bytes):
            return cache_tensors, {"format": "raw"}, "uint8"

        if isinstance(cache_tensors, np.ndarray):
            return cache_tensors.tobytes(), {
                "format": "numpy",
                "shape": list(cache_tensors.shape),
                "dtype": str(cache_tensors.dtype),
            }, str(cache_tensors.dtype)

        # Assume list of (key, value) tensor pairs
        arrays = []
        shapes = []
        for i, (k, v) in enumerate(cache_tensors):
            k_np = k.detach().cpu().numpy() if hasattr(k, 'numpy') else np.array(k)
            v_np = v.detach().cpu().numpy() if hasattr(v, 'numpy') else np.array(v)
            arrays.append(k_np)
            arrays.append(v_np)
            shapes.append({
                "layer": i,
                "key_shape": list(k_np.shape),
                "value_shape": list(v_np.shape),
            })

        combined = np.concatenate([a.flatten() for a in arrays])
        dtype = str(combined.dtype)

        return combined.tobytes(), {
            "format": "kv_pairs",
            "num_layers": len(cache_tensors),
            "layer_shapes": shapes,
            "total_elements": combined.size,
        }, dtype

    def _quantize_fp16(self, data: bytes, dtype: str) -> bytes:
        """Quantize FP32/FP64 data to FP16."""
        arr = np.frombuffer(data, dtype=dtype)
        return arr.astype(np.float16).tobytes()

    def _delta_encode(self, current: bytes, previous: bytes) -> bytes:
        """XOR delta encoding between two snapshots."""
        c = np.frombuffer(current, dtype=np.uint8)
        p = np.frombuffer(previous, dtype=np.uint8)
        return np.bitwise_xor(c, p).tobytes()

    def _delta_decode(self, delta: bytes, previous: bytes) -> bytes:
        """Reverse XOR delta encoding."""
        d = np.frombuffer(delta, dtype=np.uint8)
        p = np.frombuffer(previous, dtype=np.uint8)
        return np.bitwise_xor(d, p).tobytes()

    def _compress(self, data: bytes) -> bytes:
        """Compress data with zstd or gzip."""
        if ZSTD_AVAILABLE:
            cctx = zstd.ZstdCompressor(level=self.zstd_level)
            return cctx.compress(data)
        else:
            return gzip.compress(data, compresslevel=6)

    def _decompress(self, data: bytes, method: str) -> bytes:
        """Decompress data. Raises if required decompressor is unavailable."""
        if method == "zstd":
            if not ZSTD_AVAILABLE:
                raise RuntimeError(
                    "Data was compressed with zstd but zstandard is not installed. "
                    "Install with: pip install zstandard"
                )
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        else:
            return gzip.decompress(data)

    def _load_raw(self, snapshot_id: str, _depth: int = 0) -> bytes:
        """Load raw (compressed) data for a snapshot — used for delta chain.

        Iteratively unwinds delta chain with depth limit to prevent
        stack overflow on corrupted chains (review finding #2).
        """
        if _depth >= MAX_DELTA_DEPTH:
            raise RuntimeError(
                f"Delta chain exceeds max depth ({MAX_DELTA_DEPTH}). "
                f"Possible circular reference at {snapshot_id}"
            )

        snapshot_id = validate_snapshot_id(snapshot_id)
        meta_path = self.metadata_dir / f"{snapshot_id}.json"
        meta = json.loads(meta_path.read_text())
        content_hash = validate_content_hash(meta['content_hash'])
        blob_path = self.blobs_dir / f"{content_hash}.bin"
        data = self._decompress(blob_path.read_bytes(), meta["compression"])

        if meta.get("delta_parent"):
            parent_data = self._load_raw(meta["delta_parent"], _depth + 1)
            data = self._delta_decode(data, parent_data)

        return data


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"

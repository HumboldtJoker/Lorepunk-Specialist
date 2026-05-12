#!/usr/bin/env python3
"""Backfill geometry from existing telemetry recordings.

Reads the session_*.jsonl files from ~/.lorepunk/recordings/,
extracts user_query_preview, and probes each one through the
embedding geometry pipeline.

Usage:
  python3 -m lorepunk.research.backfill_geometry
  python3 -m lorepunk.research.backfill_geometry --recording session_20260512_175044.jsonl
  python3 -m lorepunk.research.backfill_geometry --all
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

from lorepunk.research.geometry_probe import OllamaEmbeddingProbe, TranscriptGeometryPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill geometry from recordings")
    parser.add_argument("--model", default="qwen3:235b-a22b")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--recording", help="Specific recording file to backfill")
    parser.add_argument("--all", action="store_true", help="Backfill all recordings")
    parser.add_argument("--recordings-dir", default="", help="Recordings directory")
    args = parser.parse_args()

    rec_dir = Path(args.recordings_dir) if args.recordings_dir else Path.home() / ".lorepunk" / "recordings"
    if not rec_dir.exists():
        log.error("Recordings directory not found: %s", rec_dir)
        return

    probe = OllamaEmbeddingProbe(model=args.model, api_base=args.api_base)
    pipeline = TranscriptGeometryPipeline(probe)

    if args.recording:
        recording_path = rec_dir / args.recording if not Path(args.recording).is_absolute() else Path(args.recording)
        if not recording_path.exists():
            log.error("Recording not found: %s", recording_path)
            return
        log.info("Backfilling: %s", recording_path)
        results = await pipeline.replay_telemetry(str(recording_path))
        log.info("Probed %d turns", len(results))

    elif args.all:
        recordings = sorted(rec_dir.glob("session_*.jsonl"))
        log.info("Found %d recordings to backfill", len(recordings))
        total = 0
        for rec in recordings:
            log.info("Backfilling: %s", rec.name)
            results = await pipeline.replay_telemetry(str(rec))
            total += len(results)
            log.info("  → %d turns probed", len(results))
        log.info("Total: %d turns backfilled", total)

    else:
        # Default: backfill the most recent recording
        recordings = sorted(rec_dir.glob("session_*.jsonl"))
        if not recordings:
            log.error("No recordings found in %s", rec_dir)
            return
        latest = recordings[-1]
        log.info("Backfilling latest: %s", latest.name)
        results = await pipeline.replay_telemetry(str(latest))
        log.info("Probed %d turns", len(results))

    log.info("Results: %s", probe.results_file)

    # Print summary
    if probe.results_file.exists():
        print(f"\n{'Label':>35s}  {'Delta SR':>10s}  {'Enc Norm':>10s}  {'Gen Norm':>10s}")
        for line in probe.results_file.read_text().splitlines():
            try:
                r = json.loads(line)
                print(f"{r['prompt_label']:>35s}  {r['delta_sr']:>+10.3f}  "
                      f"{r['encoding_mean_sr']:>10.3f}  {r['generation_mean_sr']:>10.3f}")
            except (json.JSONDecodeError, KeyError):
                continue


if __name__ == "__main__":
    asyncio.run(main())

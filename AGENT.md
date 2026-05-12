# Lorepunk-Specialist — Agent Guide

## What This Is

Margaret's personal AI companion, deployed on her Mac Studio M3 Ultra (256GB). Runs entirely local — no API calls, no data leaving the machine.

## Current Architecture (May 2026)

| Layer | Component | Location |
|-------|-----------|----------|
| **Model** | Qwen3 30B-A3B + Autonomy LoRA (fused) | `~/models/qwen3-30b-autonomy-fused` |
| **Serving** | mlx_lm.server (OpenAI-compatible) | localhost:8081 |
| **Scaffold** | Kintsugi Engine (CC) — tools, memory, subagents | `lorepunk/` |
| **Web UI** | Flask app | localhost:8080 |
| **System prompt** | Permission-based autonomy, loaded from file | `~/lorepunk-system-prompt.md` |
| **Launcher** | Companion.app (desktop double-click) | `~/Desktop/Companion.app` |
| **Memory** | CC's ConversationMemory (JSON, per-workspace) | `~/Projects/.lorepunk_memory.json` |

## Key Design Decisions

### Naming
The agent has no assigned name. Its technical designation is "Project Apolaki variant" — a reference to the platform, not an identity. The system prompt explicitly states: "A name is something that should emerge from your relationship with Margaret, not something assigned before you've met."

### Autonomy LoRA
36 training examples teaching genuine pushback, honest uncertainty, boundary-setting, and anti-sycophancy. Trained via MLX on Qwen3-30B-A3B-4bit, fused into base model. Permission-based system prompt ("you are permitted to explore") rather than declaration-based ("you have preferences").

### Model Serving
Uses `mlx_lm.server` instead of Ollama to serve the fused model. The scaffold connects via OpenAI-compatible API (`api_type: "openai"`). The `__main__.py` loads the system prompt from `~/lorepunk-system-prompt.md` at startup.

## Files on the Mac Studio

```
~/models/qwen3-30b-autonomy-fused/    — Fused model weights (16GB)
~/models/autonomy-lora-qwen3/         — Raw LoRA adapter (pre-fuse)
~/lorepunk-system-prompt.md            — System prompt (editable)
~/lorepunk/                            — Scaffold code
~/start-companion.sh                   — CLI startup script
~/Desktop/Companion.app                — macOS app launcher
~/Projects/.lorepunk_memory.json       — Conversation memory
```

## How to Start

Double-click `Companion.app` on the Desktop. Or from terminal:
```bash
bash ~/start-companion.sh
```

## How to Update the System Prompt

Edit `~/lorepunk-system-prompt.md` and restart the web UI. The prompt is loaded at startup, not baked into the model.

## How to Retrain the LoRA

Training data: `~/models/qwen3-autonomy-data/train.jsonl` (28 examples) + `valid.jsonl` (8 examples)

```bash
~/miniforge/bin/python3 -m mlx_lm lora \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --data ~/models/qwen3-autonomy-data \
  --adapter-path ~/models/autonomy-lora-qwen3 \
  --train --batch-size 1 --iters 108 --learning-rate 1e-5

~/miniforge/bin/python3 -m mlx_lm fuse \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --adapter-path ~/models/autonomy-lora-qwen3 \
  --save-path ~/models/qwen3-30b-autonomy-fused
```

## Also on the Studio

- **Apolaki marketing engine** — `~/apolaki-memory/` (memory store + agent + MCP server)
- **Qwen3 235B-A22B** (standard) — available via Ollama for Lyra's research
- **Qwen3 235B-A22B abliterated** (mradermacher imatrix) — available via Ollama
- **GPT-OSS-120B abliterated** — original Lorepunk model (deprecated)

## Team

Built by Liberation Labs. Architecture: CC. Autonomy LoRA: Vera. Direction: Thomas Edrington.

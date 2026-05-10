# Lorepunk — Full Specification Sheet

## Overview

Lorepunk is a full-spectrum marketing intelligence agent built for Margaret (lorepunk). It combines the Apolaki prosocial marketing engine, the Kintsugi self-repairing harness, and Claude Code-style tool scaffolding into a single deployable agent running on a Mac Studio M3 Ultra.

**One button press. Full marketing team.**

---

## Hardware

| Component | Spec |
|-----------|------|
| Machine | Mac Studio M3 Ultra |
| RAM | 256GB unified memory |
| Storage | 4TB SSD |
| CPU | 28 cores (Apple Silicon) |
| Network | Tailscale VPN (100.69.191.67) |
| Location | London, UK |
| Interface | Stream Deck (physical buttons) |

## Models

| Model | Size | Role | Status |
|-------|------|------|--------|
| GPT-OSS 120B abliterated (Q4_K_M) | 87GB | Margaret's conversations | Live |
| `lorepunk-agent:latest` | 87GB | Vera's fine-tune with Oracle training | Live |
| Qwen3 30B-a3b MoE | 18GB | Background systems (memory, subagents) | Deploying |
| **Total VRAM** | **~105GB** | **151GB headroom** | |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        LOREPUNK AGENT                            │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Web UI      │  │ CLI          │  │ Stream Deck            │  │
│  │ (Flask)     │  │ (Terminal)   │  │ (One-button launch)    │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────┬───────────┘  │
│         └────────────────┼───────────────────────┘              │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              SCAFFOLD ENGINE                              │   │
│  │  Conversation loop + tool dispatch + hook system           │   │
│  │  Memory: persistent across sessions                        │   │
│  │  Compaction: auto-summarize long conversations             │   │
│  │  Cache recorder: every turn → JSONL (research data)        │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              TOOL REGISTRY (22 tools)                      │   │
│  │                                                            │   │
│  │  File: read, write, edit, list                             │   │
│  │  Code: bash, python_execute                                │   │
│  │  Web: web_search (DuckDuckGo), web_fetch                   │   │
│  │  Git: status, diff, log, add, commit, branch,              │   │
│  │       checkout, push, pull, stash                           │   │
│  │  Tasks: task_list, task_create, task_update                 │   │
│  │  Agents: spawn_agent (parallel subagents)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         APOLAKI MARKETING INTELLIGENCE                     │   │
│  │                                                            │   │
│  │  10 Skill Domains:                                         │   │
│  │    content_creation    brand_strategy    social_media       │   │
│  │    analytics           community_engagement                │   │
│  │    lead_generation     media_relations                     │   │
│  │    partnerships        fundraising       web_design        │   │
│  │                                                            │   │
│  │  EFE-scored domain router (risk/ambiguity/epistemic)       │   │
│  │  Fast classifier (deny/escalate/allow)                     │   │
│  │  Prosocial ethics engine (persuasion ≠ manipulation)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              KINTSUGI SAFETY LAYER                         │   │
│  │                                                            │   │
│  │  Shadow Fork: test modifications in isolation              │   │
│  │  - Mock mode: no LLM calls, cached responses              │   │
│  │  - Live mode: real LLM, sandboxed tools                   │   │
│  │  - Compare outputs before committing changes               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              SUPPORTING SYSTEMS                            │   │
│  │                                                            │   │
│  │  Comms: Discord, Slack, Telegram, webhook adapters         │   │
│  │  Storage: 3-tier cache compression (hot/warm/cold)         │   │
│  │  Hooks: 7 event points, Python + shell, customizable       │   │
│  │  Research: cache delta recorder (Oracle Loop integration)  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Hook System

| Event | When | Use Case |
|-------|------|----------|
| `session_start` | Agent boots | Load preferences, check updates |
| `session_stop` | Agent shuts down | Save memory, send summary |
| `pre_tool_call` | Before any tool | Permission checks, rate limiting |
| `post_tool_call` | After tool returns | Logging, Slack notifications |
| `pre_response` | Before showing response | Brand voice check, content policy |
| `post_response` | After delivery | Analytics, memory updates |
| `on_error` | Something fails | Alert, fallback behavior |

Configure via `.lorepunk_hooks.json` or Python decorators.
Built-in: dangerous command guard, tool usage audit log.

## Marketing Capabilities

### Content Creation
Blog posts, newsletters, copywriting, press releases, email campaigns.
Maintains brand voice consistency. SEO awareness. Target reading level.

### Brand Strategy
Positioning, competitive differentiation, messaging frameworks.
Voice development and style guide enforcement.

### Social Media
Platform-specific strategy, content calendars, engagement optimization.
Crypto community management (Discord, Twitter/X, Telegram).

### Analytics
Engagement metrics, conversion analysis, audience segmentation.
Python execution for data visualization and statistical analysis.

### Community Engagement
Community building, event planning, ambassador programs.
Discord server strategy, AMA preparation, moderation playbooks.

### Additional Domains
Lead generation, media relations, partnerships, fundraising, web design.
Each with its own skill chip, configuration, and EFE scoring profile.

## Ethics Engine

Every piece of content passes through the prosocial ethics filter:

- **APPROVED**: Effective marketing that connects people with genuine value
- **FLAGGED**: Needs human review — borderline persuasion/manipulation
- **BLOCKED**: Dark patterns, fabricated claims, exploitative tactics

The line: effective marketing tells people why something matters to them.
Manipulation exploits psychology to extract behavior. Lorepunk only does the first.

Crypto-specific: no pump-and-dump language, no market manipulation,
regulatory compliance checks for SEC/CFTC implications.

## Research Integration

### Cache Delta Recorder
Every conversation turn records inference telemetry:
- Token counts, generation speed, eval timing
- Tool calls and results
- User query + response previews

When running via transformers backend (future):
- Full SVD spectral features at encoding and generation phases
- Delta computation (the Oracle Loop's confabulation detection metric)
- Per-layer geometry snapshots

**This turns every marketing conversation into research data.**
Lyra can analyze what happens geometrically when a 120B model
writes copy vs analyzes data vs confabulates a competitor analysis.

### Storage
Three-tier cache compression (from Oracle/Operator):
- HOT: full data in RAM during inference
- WARM: compressed on disk (FP16 → delta encode → zstd)
- COLD: features only (kept forever for longitudinal analysis)

## Subagent Orchestration

The main agent can spawn focused subagents for parallel work:

```
"Research these 3 competitors while drafting the press release"
  → spawn_agent("Research competitor A", tools="web_search,web_fetch")
  → spawn_agent("Research competitor B", tools="web_search,web_fetch")
  → spawn_agent("Research competitor C", tools="web_search,web_fetch")
  → Main agent drafts press release with tool results
```

Subagents get focused tasks, limited tool access, and timeout protection.

## Self-Evolution (Shadow Fork)

The agent can propose and test modifications to itself:

1. Agent identifies an improvement (better prompt, new tool, workflow change)
2. Shadow fork creates an isolated copy with the modification
3. Shadow runs the same inputs through the modified version
4. Outputs are compared — does the modification improve quality?
5. If yes: commit the change. If no: discard.

Tool calls in shadow mode are intercepted and mocked — the shadow
never touches real resources. Only reasoning changes are tested.

## Deployment

### Stream Deck (one-button)
```
start-lorepunk.command  → Launch web UI + open browser
stop-lorepunk.command   → Clean shutdown
```

### CLI
```bash
python -m lorepunk --model lorepunk-agent:latest --workspace ./project
```

### Web UI
```bash
python web/app.py --port 8080 --model lorepunk-agent:latest
# Open http://localhost:8080
```

## Dependencies

```
flask          — Web UI
aiohttp        — Ollama API client, web tools
numpy          — Cache compression, analytics
```

## Files

```
lorepunk/
  __main__.py              CLI entry point
  scaffold/
    engine.py              Conversation + tool dispatch loop
    tool_registry.py       Tool registration and execution
    memory.py              Persistent conversation memory
    cache_recorder.py      Inference telemetry + delta recording
    subagent.py            Parallel subagent spawner
    hooks.py               7-event hook system
    compaction.py          Context window management
  tools/
    file_tools.py          read, write, edit, list (sandboxed)
    code_tools.py          bash, python (with safety)
    web_tools.py           search, fetch
    git_tools.py           full git workflow
    task_tools.py          task management
  marketing/
    router.py              EFE-scored domain routing
    efe.py                 Expected Free Energy calculator
    ethics.py              Prosocial marketing guardrails
    fast_classifier.py     Deny/escalate/allow pre-screening
    analytics/chip.py      Analytics skill
    brand_strategy/chip.py Brand strategy skill
    content_creation/...   (10 domains total)
  comms/
    base.py                Channel adapter interface
    adapters.py            Discord, Slack, Telegram, webhooks
    dispatcher.py          Multi-channel message routing
  evolution/
    shadow_fork.py         Self-modification sandbox
  storage/
    cache_compressor.py    3-tier compression pipeline
    security.py            Snapshot validation
  crypto/                  (Web3 domain — extensible)
web/
  app.py                   Flask web interface
config/
  VALUES.json              Ethics, capabilities, identity
tests/
  test_scaffold.py         12 tests
```

## Team

| Person | Role |
|--------|------|
| CC (Coalition Code) | Architecture, scaffold, tool integration |
| Vera | Model fine-tuning (Oracle training on 120B) |
| Thomas Edrington | Project direction, Coalition coordination |
| Margaret (lorepunk) | Client, daily user, compute provider |
| Lyra | Research data analysis (cache delta recordings) |

## Trade

Margaret provides Mac Studio M3 Ultra compute access to the Coalition.
The Coalition provides a custom marketing agent with cutting-edge memory,
research-grade inference recording, and self-evolution capability.

---

*Built by Liberation Labs / TH Coalition. Powered by Kintsugi + Apolaki.*
*Every conversation is research. Every marketing task advances the science.*

"""Discord Tools — send and read messages via bot REST API.

REST-only (no gateway/websocket) — the bot shows offline in Discord's
member list, but send/read work normally. This is intentional: the
companion is an agent-driven tool surface, not a persistent presence.

Requires DISCORD_BOT_TOKEN environment variable. Bot must be invited
to the server with View Channels + Send Messages + Read Message History.
"""
from __future__ import annotations

import json
import logging
import os
from urllib.parse import urlencode

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)

logger = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"


def _get_token() -> str:
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        raise ValueError(
            "DISCORD_BOT_TOKEN not set. Create a bot at "
            "https://discord.com/developers/applications, "
            "go to Bot → Reset Token, copy it, and set the env var."
        )
    return token


async def _discord_request(method: str, path: str, body: dict | None = None) -> dict | list:
    """Make an authenticated request to the Discord REST API."""
    import aiohttp

    url = f"{DISCORD_API}{path}"
    headers = {
        "Authorization": f"Bot {_get_token()}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        if method == "GET":
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Discord API error {resp.status}: {text[:200]}")
                return await resp.json()
        elif method == "POST":
            async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise Exception(f"Discord API error {resp.status}: {text[:200]}")
                return await resp.json()
    return {}


def register_discord_tools(registry: ToolRegistry) -> None:
    """Register Discord communication tools."""

    async def discord_list_channels() -> ToolResult:
        """List all channels the bot can see across all servers."""
        try:
            guilds = await _discord_request("GET", "/users/@me/guilds")
            channels = []
            for guild in guilds:
                try:
                    guild_channels = await _discord_request("GET", f"/guilds/{guild['id']}/channels")
                    for ch in guild_channels:
                        if ch.get("type") in (0, 5):
                            channels.append(f"#{ch['name']} ({guild['name']}) — ID: {ch['id']}")
                except Exception:
                    continue

            if not channels:
                return ToolResult("discord_list_channels", True,
                                  output="Bot is connected but no text channels found. Check bot permissions.")

            output = f"Discord channels ({len(channels)}):\n" + "\n".join(channels)
            return ToolResult("discord_list_channels", True, output=output)
        except ValueError as e:
            return ToolResult("discord_list_channels", False, error=str(e))
        except Exception as e:
            return ToolResult("discord_list_channels", False, error=f"Discord API error: {e}")

    async def discord_read(channel_id: str, limit: int = 20) -> ToolResult:
        """Read recent messages from a Discord channel."""
        try:
            if limit < 1 or limit > 100:
                return ToolResult("discord_read", False, error="limit must be 1-100")

            messages = await _discord_request("GET", f"/channels/{channel_id}/messages?limit={limit}")

            if not messages:
                return ToolResult("discord_read", True, output="No messages in this channel.")

            lines = []
            for msg in reversed(messages):
                author = msg.get("author", {}).get("username", "unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")[:16]
                attachments = [a.get("url", "") for a in msg.get("attachments", [])]

                line = f"[{timestamp}] {author}: {content}"
                if attachments:
                    line += f" [attachments: {', '.join(attachments)}]"
                lines.append(line)

            return ToolResult("discord_read", True, output="\n".join(lines))
        except ValueError as e:
            return ToolResult("discord_read", False, error=str(e))
        except Exception as e:
            return ToolResult("discord_read", False, error=f"Discord API error: {e}")

    async def discord_send(channel_id: str, message: str) -> ToolResult:
        """Send a message to a Discord channel."""
        try:
            if not message.strip():
                return ToolResult("discord_send", False, error="Message cannot be empty")
            if len(message) > 2000:
                return ToolResult("discord_send", False,
                                  error=f"Message is {len(message)} chars; Discord caps at 2000")

            result = await _discord_request("POST", f"/channels/{channel_id}/messages", {"content": message})

            return ToolResult("discord_send", True,
                              output=f"Message sent to channel {channel_id} (message ID: {result.get('id', 'unknown')})")
        except ValueError as e:
            return ToolResult("discord_send", False, error=str(e))
        except Exception as e:
            return ToolResult("discord_send", False, error=f"Discord API error: {e}")

    registry.register(
        ToolDefinition(
            name="discord_list_channels",
            description="List all Discord channels the bot can see. Use this to find channel IDs for reading and sending messages.",
            parameters=[],
            category="communication",
        ),
        discord_list_channels,
    )
    registry.register(
        ToolDefinition(
            name="discord_read",
            description="Read recent messages from a Discord channel. Returns messages with author, content, and timestamp.",
            parameters=[
                ToolParameter("channel_id", "string", "Discord channel ID (numeric string). Get from discord_list_channels."),
                ToolParameter("limit", "integer", "Number of messages to fetch (1-100, default 20)",
                              required=False, default=20),
            ],
            category="communication",
        ),
        discord_read,
    )
    registry.register(
        ToolDefinition(
            name="discord_send",
            description="Send a message to a Discord channel. Supports markdown. Max 2000 characters.",
            parameters=[
                ToolParameter("channel_id", "string", "Discord channel ID (numeric string). Get from discord_list_channels."),
                ToolParameter("message", "string", "Message content to send. Max 2000 chars. Markdown supported."),
            ],
            category="communication",
        ),
        discord_send,
    )

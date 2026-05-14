"""Communication Tools — send messages across platforms.

Wraps the comms dispatcher as scaffold tools so the companion
can send notifications, briefings, and alerts through conversation.
"""
from __future__ import annotations

import os
import logging

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)
from lorepunk.comms.dispatcher import CommsDispatcher, Message, Urgency
from lorepunk.comms.adapters import (
    DiscordWebhookAdapter, SlackWebhookAdapter,
    TelegramAdapter, LogAdapter,
)

logger = logging.getLogger(__name__)


def _build_dispatcher() -> CommsDispatcher:
    """Build a dispatcher from available environment variables."""
    dispatcher = CommsDispatcher()
    dispatcher.register(LogAdapter())

    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if discord_webhook:
        dispatcher.register(DiscordWebhookAdapter(discord_webhook))

    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if slack_webhook:
        dispatcher.register(SlackWebhookAdapter(slack_webhook))

    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if telegram_token:
        dispatcher.register(TelegramAdapter(telegram_token, telegram_chat))

    return dispatcher


def register_comms_tools(registry: ToolRegistry) -> None:
    """Register communication tools."""
    dispatcher = _build_dispatcher()

    async def comms_send(message: str, channel: str = "", urgency: str = "normal", title: str = "") -> ToolResult:
        """Send a message through a communication channel."""
        urgency_map = {
            "low": Urgency.LOW, "normal": Urgency.NORMAL,
            "high": Urgency.HIGH, "critical": Urgency.CRITICAL,
        }
        urg = urgency_map.get(urgency.lower(), Urgency.NORMAL)
        msg = Message(content=message, urgency=urg, title=title)

        if channel:
            result = await dispatcher.send(msg, channel)
        else:
            result = await dispatcher.broadcast(msg)

        if result.any_succeeded:
            return ToolResult("comms_send", True,
                              output=result.summary())
        else:
            failed = ", ".join(result.failed_channels)
            return ToolResult("comms_send", False,
                              error=f"Delivery failed on: {failed}")

    async def comms_channels() -> ToolResult:
        """List available communication channels and their status."""
        all_channels = dispatcher.channels
        connected = dispatcher.connected_channels
        lines = [f"Communication channels ({len(all_channels)} registered):"]
        for ch in all_channels:
            status = "connected" if ch in connected else "not connected"
            lines.append(f"  - {ch}: {status}")
        return ToolResult("comms_channels", True, output="\n".join(lines))

    async def comms_crisis(message: str, title: str = "ALERT") -> ToolResult:
        """Send a crisis alert to ALL channels simultaneously."""
        result = await dispatcher.crisis_alert(message, title)
        return ToolResult("comms_crisis", True, output=result.summary())

    async def comms_briefing(content: str, channels: str = "") -> ToolResult:
        """Send a briefing/summary to specified channels."""
        ch_list = [c.strip() for c in channels.split(",")] if channels else None
        msg = Message(content=content, title="Briefing", urgency=Urgency.LOW)
        result = await dispatcher.broadcast(msg, channels=ch_list)
        return ToolResult("comms_briefing", True, output=result.summary())

    async def comms_history(limit: int = 10) -> ToolResult:
        """View recent message delivery history."""
        log = dispatcher.get_dispatch_log(limit)
        if not log:
            return ToolResult("comms_history", True, output="No messages sent yet.")
        lines = [f"Last {len(log)} dispatches:"]
        for entry in log:
            ts = entry.dispatched_at.strftime("%H:%M")
            lines.append(f"  [{ts}] {entry.message.content[:60]}... → {entry.summary()}")
        return ToolResult("comms_history", True, output="\n".join(lines))

    registry.register(
        ToolDefinition(
            name="comms_send",
            description="Send a message through a communication channel (Discord, Slack, Telegram, or all). Use for notifications, updates, and outreach.",
            parameters=[
                ToolParameter("message", "string", "Message content to send"),
                ToolParameter("channel", "string", "Channel name (discord, slack, telegram, log). Leave empty for all.",
                              required=False, default=""),
                ToolParameter("urgency", "string", "Urgency level: low, normal, high, critical",
                              required=False, default="normal"),
                ToolParameter("title", "string", "Optional title/subject",
                              required=False, default=""),
            ],
            category="communication",
        ),
        comms_send,
    )
    registry.register(
        ToolDefinition(
            name="comms_channels",
            description="List available communication channels and whether they are connected.",
            parameters=[],
            category="communication",
        ),
        comms_channels,
    )
    registry.register(
        ToolDefinition(
            name="comms_crisis",
            description="Send a crisis alert to ALL channels simultaneously. Use only for genuine emergencies.",
            parameters=[
                ToolParameter("message", "string", "Crisis alert message"),
                ToolParameter("title", "string", "Alert title", required=False, default="ALERT"),
            ],
            category="communication",
        ),
        comms_crisis,
    )
    registry.register(
        ToolDefinition(
            name="comms_briefing",
            description="Send a briefing or summary to specific channels. Good for daily reports, weekly summaries, campaign updates.",
            parameters=[
                ToolParameter("content", "string", "Briefing content"),
                ToolParameter("channels", "string", "Comma-separated channel names (empty for all)",
                              required=False, default=""),
            ],
            category="communication",
        ),
        comms_briefing,
    )
    registry.register(
        ToolDefinition(
            name="comms_history",
            description="View recent message delivery history — what was sent, where, and whether it was delivered.",
            parameters=[
                ToolParameter("limit", "integer", "Number of recent entries (default 10)",
                              required=False, default=10),
            ],
            category="communication",
        ),
        comms_history,
    )

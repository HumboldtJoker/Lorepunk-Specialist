"""Web Tools — research, fetch, and analyze web content.

A marketing agent needs to see the landscape:
  - What are competitors saying?
  - What's trending in the space?
  - What does a specific URL contain?
  - What are people discussing on forums/social media?

Uses aiohttp for fetching. Search via DuckDuckGo (no API key needed).
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)

logger = logging.getLogger(__name__)


def register_web_tools(registry: ToolRegistry) -> None:
    """Register web research tools."""

    async def web_fetch(url: str, extract_text: bool = True) -> ToolResult:
        """Fetch a URL and return its content."""
        try:
            import aiohttp
        except ImportError:
            return ToolResult("web_fetch", False, error="aiohttp required")

        try:
            headers = {"User-Agent": "Lorepunk/1.0 (Marketing Research Agent)"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return ToolResult("web_fetch", False,
                                          error=f"HTTP {resp.status}")
                    html = await resp.text()

            if extract_text:
                text = _html_to_text(html)
                return ToolResult("web_fetch", True, output=text[:15000])
            return ToolResult("web_fetch", True, output=html[:15000])
        except Exception as e:
            return ToolResult("web_fetch", False, error=str(e))

    async def web_search(query: str, max_results: int = 5) -> ToolResult:
        """Search the web using DuckDuckGo (no API key needed)."""
        try:
            import aiohttp
        except ImportError:
            return ToolResult("web_search", False, error="aiohttp required")

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Lorepunk/1.0 (Marketing Research Agent)"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    html = await resp.text()

            results = _parse_ddg_results(html, max_results)
            if not results:
                return ToolResult("web_search", True,
                                  output="No results found. Try different search terms.")

            output_lines = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                output_lines.append(f"{i}. {r['title']}")
                output_lines.append(f"   {r['url']}")
                output_lines.append(f"   {r['snippet']}")
                output_lines.append("")

            return ToolResult("web_search", True, output="\n".join(output_lines))
        except Exception as e:
            return ToolResult("web_search", False, error=str(e))

    registry.register(
        ToolDefinition(
            name="web_fetch", category="web",
            description="Fetch and extract text from a URL. Use for reading articles, competitor pages, documentation.",
            parameters=[
                ToolParameter("url", "string", "URL to fetch"),
                ToolParameter("extract_text", "boolean", "Extract text from HTML (default true)",
                              required=False, default=True),
            ],
        ),
        web_fetch,
    )

    registry.register(
        ToolDefinition(
            name="web_search", category="web",
            description="Search the web. Use for competitor research, trend analysis, fact-checking.",
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("max_results", "integer", "Maximum results (default 5)",
                              required=False, default=5),
            ],
        ),
        web_search,
    )


def _html_to_text(html: str) -> str:
    """Simple HTML to text extraction."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.S)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.S)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)
    text = re.sub(r'&nbsp;', ' ', text)
    return text.strip()


def _parse_ddg_results(html: str, max_results: int) -> list[dict]:
    """Parse DuckDuckGo HTML results."""
    results = []
    result_blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.S,
    )

    for url, title, snippet in result_blocks[:max_results]:
        title = re.sub(r'<[^>]+>', '', title).strip()
        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        if url.startswith("//duckduckgo.com/l/?uddg="):
            from urllib.parse import unquote
            url = unquote(url.split("uddg=")[1].split("&")[0])
        results.append({"title": title, "url": url, "snippet": snippet})

    return results

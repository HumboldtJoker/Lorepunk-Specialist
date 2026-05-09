"""Lorepunk — marketing intelligence with Claude Code hands.

Usage:
  python -m lorepunk                                    # Interactive CLI
  python -m lorepunk --model lorepunk:latest            # Specify model
  python -m lorepunk --api-base http://localhost:11434  # Custom Ollama
  python -m lorepunk --workspace ./my-project           # Set workspace
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from lorepunk.scaffold.engine import ScaffoldEngine, EngineConfig
from lorepunk.scaffold.tool_registry import ToolRegistry
from lorepunk.tools.file_tools import register_file_tools
from lorepunk.tools.code_tools import register_code_tools


SYSTEM_PROMPT = """You are Lorepunk, a marketing intelligence agent with full tool access.

You can read, write, and edit files. You can execute bash commands and Python scripts.
You can analyze data, generate content, build campaigns, and manage projects.

Your expertise:
- Marketing strategy and campaign planning
- Content creation (social media, blog, email, press releases)
- Data analysis and visualization
- Crypto/Web3 marketing (tokenomics, community, regulatory awareness)
- Brand voice development and consistency
- Competitor and market analysis

When the user asks you to do something:
1. Think about what tools you need
2. Use them to accomplish the task
3. Present the results clearly

You have access to the filesystem and can create project structures,
write marketing copy, analyze CSVs, generate reports, and more.

Be direct, professional, and creative. You're not just an assistant —
you're a marketing strategist with a code editor."""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lorepunk",
        description="Lorepunk — marketing intelligence agent",
    )
    parser.add_argument("--model", default="lorepunk:latest",
                        help="Ollama model name")
    parser.add_argument("--api-base", default="http://localhost:11434",
                        help="Ollama API base URL")
    parser.add_argument("--api-type", default="ollama",
                        choices=["ollama", "openai"],
                        help="API type")
    parser.add_argument("--workspace", default=".",
                        help="Working directory for file operations")
    parser.add_argument("--no-bash", action="store_true",
                        help="Disable bash execution")
    parser.add_argument("--no-python", action="store_true",
                        help="Disable Python execution")
    args = parser.parse_args()

    registry = ToolRegistry()
    register_file_tools(registry, workspace=args.workspace)
    register_code_tools(
        registry,
        workspace=args.workspace,
        bash_enabled=not args.no_bash,
        python_enabled=not args.no_python,
    )

    config = EngineConfig(
        model=args.model,
        api_base=args.api_base,
        api_type=args.api_type,
        system_prompt=SYSTEM_PROMPT,
    )

    engine = ScaffoldEngine(config=config, registry=registry)

    print(f"\n  Lorepunk — Marketing Intelligence")
    print(f"  Model: {args.model}")
    print(f"  Tools: {registry.tool_count} registered")
    print(f"  Workspace: {args.workspace}")
    print(f"  Type 'quit' to exit, 'clear' to reset.\n")

    asyncio.run(interactive_loop(engine))


async def interactive_loop(engine: ScaffoldEngine) -> None:
    while True:
        try:
            user_input = input("  you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("  Goodbye!")
            break
        if user_input.lower() == "clear":
            engine.clear_history()
            print("  (history cleared)")
            continue

        response = await engine.chat(user_input)
        print(f"\n  lorepunk: {response}\n")


if __name__ == "__main__":
    main()

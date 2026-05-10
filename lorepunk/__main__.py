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
from lorepunk.scaffold.memory import MemoryStore
from lorepunk.tools.file_tools import register_file_tools
from lorepunk.tools.code_tools import register_code_tools
from lorepunk.tools.web_tools import register_web_tools
from lorepunk.tools.git_tools import register_git_tools
from lorepunk.tools.task_tools import register_task_tools


SYSTEM_PROMPT = """You are Lorepunk, a marketing intelligence agent with full tool access.

You have the following tool categories:
- **File**: read, write, edit, list files in the workspace
- **Code**: execute bash commands and Python scripts
- **Web**: search the web, fetch and extract page content
- **Git**: status, diff, log, add, commit, branch, push, pull, stash
- **Tasks**: create, list, and update tasks

Your expertise:
- Marketing strategy and campaign planning
- Content creation (social media, blog, email, press releases, whitepapers)
- Data analysis, visualization, and reporting
- Crypto/Web3 marketing (tokenomics, community, regulatory awareness)
- Brand voice development and consistency
- Competitor and market analysis
- SEO and content optimization

When the user asks you to do something:
1. Think about what tools you need
2. Use them to accomplish the task
3. Present the results clearly

You can create entire project structures, write marketing copy, analyze CSVs,
generate reports, manage git repos, and track tasks — all through conversation.

Be direct, professional, and creative. You're not just an assistant —
you're a marketing strategist with a code editor and a research desk.

{memory_context}"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lorepunk",
        description="Lorepunk — marketing intelligence agent",
    )
    parser.add_argument("--model", default="gpt-oss-120b-abliterated:latest",
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

    # Load persistent memory
    mem_store = MemoryStore(args.workspace)
    memory = mem_store.load()
    memory.start_session()

    # Build tool registry — all the hands
    registry = ToolRegistry()
    register_file_tools(registry, workspace=args.workspace)
    register_code_tools(
        registry,
        workspace=args.workspace,
        bash_enabled=not args.no_bash,
        python_enabled=not args.no_python,
    )
    register_web_tools(registry)
    register_git_tools(registry, workspace=args.workspace)
    register_task_tools(registry, workspace=args.workspace)

    # Build system prompt with memory context
    memory_context = memory.get_context_prompt()
    system_prompt = SYSTEM_PROMPT.format(
        memory_context=f"\n\nSession context:\n{memory_context}" if memory_context else "",
    )

    config = EngineConfig(
        model=args.model,
        api_base=args.api_base,
        api_type=args.api_type,
        system_prompt=system_prompt,
    )

    engine = ScaffoldEngine(config=config, registry=registry)

    print(f"\n  Lorepunk — Marketing Intelligence")
    print(f"  Model: {args.model}")
    print(f"  Tools: {registry.tool_count} registered")
    print(f"  Workspace: {args.workspace}")
    print(f"  Session: #{memory.session_count}")
    if memory_context:
        print(f"  Memory: loaded from previous session")
    print(f"  Type 'quit' to exit, 'clear' to reset, 'tasks' for task list.\n")

    asyncio.run(interactive_loop(engine, memory, mem_store))


async def interactive_loop(engine: ScaffoldEngine, memory, mem_store) -> None:
    while True:
        try:
            user_input = input("  you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saving session...")
            memory.end_session("Session ended by user")
            mem_store.save(memory)
            print("  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            memory.end_session("Session ended normally")
            mem_store.save(memory)
            print("  Session saved. Goodbye!")
            break
        if user_input.lower() == "clear":
            engine.clear_history()
            print("  (history cleared)")
            continue
        if user_input.lower() == "tasks":
            result = await engine.registry.execute("task_list")
            print(f"\n  {result.output}\n")
            continue

        response = await engine.chat(user_input)
        print(f"\n  lorepunk: {response}\n")

        # Auto-save memory periodically
        if engine.message_count % 10 == 0:
            mem_store.save(memory)


if __name__ == "__main__":
    main()

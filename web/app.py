"""Lorepunk Web UI — marketing intelligence in your browser.

A clean chat interface backed by the full scaffold engine.
Margaret opens a browser, asks a question, and Lorepunk
researches, writes, analyzes, and delivers — right there.

Run:
  python -m lorepunk.web --port 8080 --workspace ./my-project
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

try:
    from flask import Flask, render_template_string, request, jsonify, session
except ImportError:
    raise ImportError("Flask required: pip install flask")

from lorepunk.scaffold.engine import ScaffoldEngine, EngineConfig
from lorepunk.scaffold.tool_registry import ToolRegistry
from lorepunk.scaffold.memory import MemoryStore
from lorepunk.scaffold.cache_recorder import CacheDeltaRecorder
from lorepunk.scaffold.subagent import SubagentSpawner, register_subagent_tools
from lorepunk.tools.file_tools import register_file_tools
from lorepunk.tools.code_tools import register_code_tools
from lorepunk.tools.web_tools import register_web_tools
from lorepunk.tools.git_tools import register_git_tools
from lorepunk.tools.task_tools import register_task_tools

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lorepunk — Marketing Intelligence</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: #16213e;
            padding: 16px 24px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header h1 { font-size: 20px; color: #e94560; font-weight: 600; }
        .header .status {
            font-size: 12px; color: #888;
            margin-left: auto;
        }
        .header .status .dot {
            display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; background: #4ecca3; margin-right: 4px;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .message.user {
            align-self: flex-end;
            background: #0f3460;
            color: #fff;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            align-self: flex-start;
            background: #16213e;
            border: 1px solid #0f3460;
            border-bottom-left-radius: 4px;
        }
        .message.tool {
            align-self: flex-start;
            background: #1a1a2e;
            border: 1px solid #333;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 13px;
            color: #4ecca3;
            max-width: 90%;
            padding: 8px 12px;
            border-radius: 8px;
        }
        .message.tool .tool-name {
            color: #e94560;
            font-weight: bold;
            font-size: 11px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .message .timestamp {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }
        .input-area {
            background: #16213e;
            padding: 16px 24px;
            border-top: 1px solid #0f3460;
            display: flex;
            gap: 12px;
        }
        .input-area textarea {
            flex: 1;
            background: #1a1a2e;
            border: 1px solid #333;
            color: #eee;
            padding: 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: none;
            min-height: 48px;
            max-height: 120px;
        }
        .input-area textarea:focus {
            outline: none;
            border-color: #e94560;
        }
        .input-area button {
            background: #e94560;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background 0.2s;
        }
        .input-area button:hover { background: #c73e54; }
        .input-area button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        .thinking {
            color: #888;
            font-style: italic;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse { 50% { opacity: 0.5; } }
        .clear-btn {
            background: transparent;
            color: #888;
            border: 1px solid #333;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }
        .clear-btn:hover { border-color: #e94560; color: #e94560; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Lorepunk</h1>
        <span style="color: #888; font-size: 14px;">Marketing Intelligence</span>
        <button class="clear-btn" onclick="clearChat()">Clear</button>
        <div class="status">
            <span class="dot"></span>
            {{ model }}
        </div>
    </div>

    <div class="chat-container" id="chat">
        <div class="message assistant">
            Hello! I'm Lorepunk, your marketing intelligence agent.

I can research competitors, write copy, analyze data, manage campaigns,
and coordinate strategy — all through conversation.

What are we working on today?
        </div>
    </div>

    <div class="input-area">
        <textarea id="input" placeholder="Ask Lorepunk anything..."
                  onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
        <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('sendBtn');

        function addMessage(role, content) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            if (role === 'tool') {
                const parts = content.split('\\n');
                const toolName = parts[0] || 'tool';
                const toolOutput = parts.slice(1).join('\\n');
                div.innerHTML = '<div class="tool-name">' + escapeHtml(toolName) + '</div>' + escapeHtml(toolOutput);
            } else {
                div.textContent = content;
            }
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function sendMessage() {
            const msg = input.value.trim();
            if (!msg) return;

            input.value = '';
            sendBtn.disabled = true;

            addMessage('user', msg);
            const thinking = addMessage('assistant', 'Thinking...');
            thinking.classList.add('thinking');

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg}),
                });
                const data = await response.json();

                chat.removeChild(thinking);

                if (data.tool_calls && data.tool_calls.length > 0) {
                    for (const tc of data.tool_calls) {
                        addMessage('tool', tc.name + '\\n' + tc.result);
                    }
                }

                addMessage('assistant', data.response);
            } catch (err) {
                chat.removeChild(thinking);
                addMessage('assistant', 'Error: ' + err.message);
            }

            sendBtn.disabled = false;
            input.focus();
        }

        async function clearChat() {
            await fetch('/clear', {method: 'POST'});
            chat.innerHTML = '';
            addMessage('assistant', 'Conversation cleared. What are we working on?');
        }

        input.focus();
    </script>
</body>
</html>"""


def create_app(
    model: str = "gpt-oss-120b-abliterated:latest",
    api_base: str = "http://localhost:11434",
    workspace: str = ".",
) -> Flask:
    """Create the Flask web application."""
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    # Build engine
    registry = ToolRegistry()
    register_file_tools(registry, workspace=workspace)
    register_code_tools(registry, workspace=workspace)
    register_web_tools(registry)
    register_git_tools(registry, workspace=workspace)
    register_task_tools(registry, workspace=workspace)

    spawner = SubagentSpawner(registry, default_model=model, default_api_base=api_base)
    register_subagent_tools(registry, spawner)

    mem_store = MemoryStore(workspace)
    memory = mem_store.load()
    memory.start_session()

    memory_context = memory.get_context_prompt()
    from lorepunk.__main__ import SYSTEM_PROMPT
    system_prompt = SYSTEM_PROMPT.format(
        memory_context=f"\n\nSession context:\n{memory_context}" if memory_context else "",
    )

    config = EngineConfig(
        model=model, api_base=api_base, api_type="ollama",
        system_prompt=system_prompt,
    )
    engine = ScaffoldEngine(config=config, registry=registry)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE, model=model)

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        message = data.get("message", "")
        if not message:
            return jsonify({"response": "Empty message", "tool_calls": []})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Track tool calls
        tool_log = []
        original_execute = registry.execute

        async def logging_execute(tool_name, **kwargs):
            result = await original_execute(tool_name, **kwargs)
            tool_log.append({
                "name": tool_name,
                "result": result.to_message()[:500],
            })
            return result

        registry.execute = logging_execute

        try:
            response = loop.run_until_complete(engine.chat(message))
        finally:
            registry.execute = original_execute
            loop.close()

        # Auto-save
        if engine.message_count % 5 == 0:
            mem_store.save(memory)

        return jsonify({
            "response": response,
            "tool_calls": tool_log,
        })

    @app.route("/clear", methods=["POST"])
    def clear():
        engine.clear_history()
        return jsonify({"status": "cleared"})

    @app.route("/status")
    def status():
        return jsonify({
            "model": model,
            "tools": registry.tool_count,
            "messages": engine.message_count,
            "session": memory.session_count,
        })

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lorepunk Web UI")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default="gpt-oss-120b-abliterated:latest")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()

    app = create_app(model=args.model, api_base=args.api_base, workspace=args.workspace)

    print(f"\n  Lorepunk Web UI")
    print(f"  Model: {args.model}")
    print(f"  Workspace: {args.workspace}")
    print(f"  Open: http://localhost:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

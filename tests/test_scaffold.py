"""Tests for the scaffold engine and tools."""
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from lorepunk.scaffold.tool_registry import (
    ToolRegistry, ToolDefinition, ToolParameter, ToolResult,
)
from lorepunk.tools.file_tools import register_file_tools
from lorepunk.tools.code_tools import register_code_tools


class TestToolRegistry:
    def test_register_tool(self):
        registry = ToolRegistry()
        defn = ToolDefinition(name="test", description="A test tool")

        async def executor():
            return ToolResult("test", True, output="ok")

        registry.register(defn, executor)
        assert registry.tool_count == 1
        assert registry.get_tool("test") is not None

    def test_schema_generation(self):
        registry = ToolRegistry()
        defn = ToolDefinition(
            name="greet", description="Say hello",
            parameters=[ToolParameter("name", "string", "Who to greet")],
        )

        async def executor(name=""):
            return ToolResult("greet", True, output=f"Hello {name}")

        registry.register(defn, executor)
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "greet"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        registry = ToolRegistry()
        defn = ToolDefinition(name="add", description="Add numbers")

        async def executor(a=0, b=0):
            return ToolResult("add", True, output=str(a + b))

        registry.register(defn, executor)
        result = await registry.execute("add", a=3, b=4)
        assert result.success
        assert result.output == "7"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent")
        assert not result.success
        assert "Unknown" in result.error


class TestFileTools:
    @pytest.mark.asyncio
    async def test_read_write_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            register_file_tools(registry, workspace=tmpdir)

            result = await registry.execute("write_file",
                file_path=f"{tmpdir}/test.txt", content="Hello World")
            assert result.success

            result = await registry.execute("read_file",
                file_path=f"{tmpdir}/test.txt")
            assert result.success
            assert "Hello World" in result.output

            result = await registry.execute("edit_file",
                file_path=f"{tmpdir}/test.txt",
                old_string="Hello", new_string="Goodbye")
            assert result.success

            result = await registry.execute("read_file",
                file_path=f"{tmpdir}/test.txt")
            assert "Goodbye World" in result.output

    @pytest.mark.asyncio
    async def test_list_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            register_file_tools(registry, workspace=tmpdir)
            Path(f"{tmpdir}/a.txt").write_text("a")
            Path(f"{tmpdir}/b.txt").write_text("b")

            result = await registry.execute("list_files", directory=tmpdir)
            assert result.success
            assert "a.txt" in result.output
            assert "b.txt" in result.output

    @pytest.mark.asyncio
    async def test_sandbox(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            register_file_tools(registry, workspace=tmpdir)

            result = await registry.execute("read_file", file_path="/etc/passwd")
            assert not result.success
            assert "outside workspace" in result.error.lower() or "denied" in result.error.lower()


class TestCodeTools:
    @pytest.mark.asyncio
    async def test_bash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            register_code_tools(registry, workspace=tmpdir)

            result = await registry.execute("bash", command="echo hello")
            assert result.success
            assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_bash_blocklist(self):
        registry = ToolRegistry()
        register_code_tools(registry)

        result = await registry.execute("bash", command="rm -rf /")
        assert not result.success
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_python(self):
        registry = ToolRegistry()
        register_code_tools(registry)

        result = await registry.execute("python_execute", code="print(2 + 2)")
        assert result.success
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_python_error(self):
        registry = ToolRegistry()
        register_code_tools(registry)

        result = await registry.execute("python_execute", code="raise ValueError('bad')")
        assert not result.success
        assert "bad" in result.error

    @pytest.mark.asyncio
    async def test_disabled(self):
        registry = ToolRegistry()
        register_code_tools(registry, bash_enabled=False, python_enabled=False)

        assert registry.tool_count == 0

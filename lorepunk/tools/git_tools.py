"""Git Tools — version control from conversation.

The agent can check status, diff, commit, branch, and push.
Essential for any code-adjacent workflow.
"""
from __future__ import annotations

from lorepunk.scaffold.tool_registry import (
    ToolDefinition, ToolParameter, ToolResult, ToolRegistry,
)
from lorepunk.tools.code_tools import register_code_tools


def register_git_tools(registry: ToolRegistry, workspace: str = ".") -> None:
    """Register git operation tools."""

    async def _run_git(*args: str) -> ToolResult:
        """Run a git command in the workspace using exec (not shell) to prevent injection."""
        import asyncio
        import shlex

        blocked_args = ["--force", "-f", "--hard", "-D"]
        blocked_combos = [("push", "--force"), ("push", "-f"), ("reset", "--hard"), ("clean", "-f"), ("branch", "-D")]
        for combo in blocked_combos:
            if all(a in args for a in combo):
                return ToolResult("git", False, error=f"Blocked destructive operation: {' '.join(combo)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                return ToolResult("git", False, output=output[:5000],
                                  error=f"Exit {proc.returncode}: {errors[:2000]}")
            return ToolResult("git", True, output=(output + errors)[:10000])
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult("git", False, error="Git command timed out")
        except Exception as e:
            return ToolResult("git", False, error=str(e))

    async def git_status() -> ToolResult:
        return await _run_git("status")

    async def git_diff(staged: bool = False) -> ToolResult:
        if staged:
            return await _run_git("diff", "--staged")
        return await _run_git("diff")

    async def git_log(n: int = 10) -> ToolResult:
        return await _run_git("log", "--oneline", "-n", str(int(n)))

    async def git_add(files: str = ".") -> ToolResult:
        import shlex
        parts = shlex.split(files)
        if any(p.startswith("-") for p in parts):
            return ToolResult("git_add", False, error="Flags not allowed in file paths")
        return await _run_git("add", "--", *parts)

    async def git_commit(message: str) -> ToolResult:
        return await _run_git("commit", "-m", message)

    async def git_branch(name: str = "") -> ToolResult:
        if name:
            if name.startswith("-"):
                return ToolResult("git_branch", False, error="Invalid branch name")
            return await _run_git("checkout", "-b", name)
        return await _run_git("branch", "-a")

    async def git_checkout(ref: str) -> ToolResult:
        if ref.startswith("-"):
            return ToolResult("git_checkout", False, error="Invalid ref")
        return await _run_git("checkout", ref)

    SAFE_REMOTES = {"origin", "upstream", "fork"}

    async def git_push(remote: str = "origin", branch: str = "") -> ToolResult:
        if remote not in SAFE_REMOTES:
            return ToolResult("git_push", False,
                              error=f"Remote '{remote}' not in allowed list: {SAFE_REMOTES}")
        args = ["push", remote]
        if branch:
            args.append(branch)
        return await _run_git(*args)

    async def git_pull(remote: str = "origin") -> ToolResult:
        if remote not in SAFE_REMOTES:
            return ToolResult("git_pull", False,
                              error=f"Remote '{remote}' not in allowed list: {SAFE_REMOTES}")
        return await _run_git("pull", remote)

    async def git_stash(action: str = "push") -> ToolResult:
        if action not in ("push", "pop", "list", "drop"):
            return ToolResult("git_stash", False, error=f"Invalid stash action: {action}")
        return await _run_git("stash", action)

    tools = [
        ("git_status", "Show working tree status — modified, staged, untracked files.",
         [], git_status),
        ("git_diff", "Show changes in working tree or staged files.",
         [ToolParameter("staged", "boolean", "Show staged changes", required=False, default=False)], git_diff),
        ("git_log", "Show recent commit history.",
         [ToolParameter("n", "integer", "Number of commits to show", required=False, default=10)], git_log),
        ("git_add", "Stage files for commit.",
         [ToolParameter("files", "string", "Files to stage (space-separated, or '.' for all)", required=False, default=".")], git_add),
        ("git_commit", "Create a commit with a message.",
         [ToolParameter("message", "string", "Commit message")], git_commit),
        ("git_branch", "List branches or create a new one.",
         [ToolParameter("name", "string", "New branch name (empty to list)", required=False, default="")], git_branch),
        ("git_checkout", "Switch to a branch or ref.",
         [ToolParameter("ref", "string", "Branch or ref to checkout")], git_checkout),
        ("git_push", "Push commits to remote.",
         [ToolParameter("remote", "string", "Remote name", required=False, default="origin"),
          ToolParameter("branch", "string", "Branch name", required=False, default="")], git_push),
        ("git_pull", "Pull latest from remote.",
         [ToolParameter("remote", "string", "Remote name", required=False, default="origin")], git_pull),
        ("git_stash", "Stash or restore working changes.",
         [ToolParameter("action", "string", "push, pop, list, or drop", required=False, default="push")], git_stash),
    ]

    for name, desc, params, executor in tools:
        registry.register(
            ToolDefinition(name=name, description=desc, parameters=params, category="git"),
            executor,
        )

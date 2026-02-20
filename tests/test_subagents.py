"""Tests for subagent delegation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apecode.subagents import SubagentRunner
from apecode.tools import ToolContext, create_default_registry


class _DelegateModel:
    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        _ = tools
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break
        return {"role": "assistant", "content": f"delegated:{last_user}"}


def test_subagent_runner_executes_task(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    tools = create_default_registry(ctx)
    runner = SubagentRunner(
        model=_DelegateModel(),
        parent_tools=tools,
        base_system_prompt="base prompt",
    )
    output = runner.run(task="inspect src", profile="general")
    assert output == "delegated:inspect src"


def test_subagent_uses_read_only_tools(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    tools = create_default_registry(ctx)
    runner = SubagentRunner(
        model=_DelegateModel(),
        parent_tools=tools,
        base_system_prompt="base prompt",
    )
    sub_tools = runner._build_subagent_tools()
    names = sub_tools.list_tool_names()
    assert "read_file" in names
    assert "write_file" not in names
    assert "exec_command" not in names

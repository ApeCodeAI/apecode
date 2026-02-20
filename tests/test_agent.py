"""Tests for nano agent loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apecode.agent import AgentConfig, NanoCodeAgent
from apecode.system_prompt import build_system_prompt
from apecode.tools import ToolContext, create_default_registry


class FakeModel:
    """A deterministic fake model for agent-loop tests."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        _ = (messages, tools)
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "note.txt"}),
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Done: file inspected."}


class LoopingModel:
    """A model that always requests tools (to test max step guard)."""

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        _ = (messages, tools)
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_loop",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        }


def test_agent_runs_tool_then_finishes(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    tools = create_default_registry(ctx)
    agent = NanoCodeAgent(
        model=FakeModel(),
        tools=tools,
        system_prompt=build_system_prompt(tmp_path),
        config=AgentConfig(max_steps=5),
    )
    output = agent.run("read note")
    assert "Done" in output


def test_max_steps_guard(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    tools = create_default_registry(ctx)
    agent = NanoCodeAgent(
        model=LoopingModel(),
        tools=tools,
        system_prompt=build_system_prompt(tmp_path),
        config=AgentConfig(max_steps=2),
    )
    with pytest.raises(RuntimeError):
        agent.run("loop forever")

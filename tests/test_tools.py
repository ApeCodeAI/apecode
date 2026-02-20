"""Tests for nano tooling layer."""

from __future__ import annotations

import json
from pathlib import Path

from apecode.tools import ToolContext, create_default_registry


def test_write_and_read_file(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    registry = create_default_registry(ctx)

    write_result = registry.execute(
        "write_file",
        json.dumps({"path": "hello.txt", "content": "line1\nline2\n"}),
    )
    assert "wrote" in write_result

    read_result = registry.execute(
        "read_file",
        json.dumps({"path": "hello.txt", "start_line": 1, "num_lines": 2}),
    )
    assert "line1" in read_result
    assert "line2" in read_result


def test_path_escape_is_blocked(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    registry = create_default_registry(ctx)

    escaped = registry.execute(
        "read_file",
        json.dumps({"path": "../outside.txt"}),
    )
    assert "path escapes workspace" in escaped


def test_reject_mutating_tool(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: False)
    registry = create_default_registry(ctx)
    result = registry.execute(
        "write_file",
        json.dumps({"path": "a.txt", "content": "x"}),
    )
    assert result == "Rejected by user."


def test_read_only_blocks_mutating_tool(tmp_path: Path) -> None:
    ctx = ToolContext(
        cwd=tmp_path,
        ask_approval=lambda _a, _p: True,
        sandbox_mode="read-only",
    )
    registry = create_default_registry(ctx)
    result = registry.execute(
        "write_file",
        json.dumps({"path": "blocked.txt", "content": "x"}),
    )
    assert result == "blocked by sandbox policy: read-only"


def test_never_policy_blocks_mutating_tool(tmp_path: Path) -> None:
    ctx = ToolContext(
        cwd=tmp_path,
        ask_approval=lambda _a, _p: True,
        approval_policy="never",
    )
    registry = create_default_registry(ctx)
    result = registry.execute(
        "write_file",
        json.dumps({"path": "blocked.txt", "content": "x"}),
    )
    assert result == "blocked by approval policy: never"


def test_update_plan(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    registry = create_default_registry(ctx)

    result = registry.execute(
        "update_plan",
        json.dumps(
            {
                "plan": [
                    {"step": "inspect", "status": "completed"},
                    {"step": "implement", "status": "in_progress"},
                ]
            }
        ),
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["plan_size"] == 2
    assert len(ctx.plan) == 2

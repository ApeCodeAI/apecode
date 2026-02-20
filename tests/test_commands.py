"""Tests for slash commands and skills."""

from __future__ import annotations

from pathlib import Path

from apecode.commands import create_default_commands
from apecode.skills import SkillCatalog
from apecode.tools import ToolContext, create_default_registry


def test_tools_and_plan_commands(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    ctx.plan = [{"step": "inspect", "status": "completed"}]
    tools = create_default_registry(ctx)
    commands = create_default_commands(tools=tools, skills=SkillCatalog.from_roots([]))

    tools_result = commands.run("/tools")
    assert tools_result is not None
    assert "read_file" in tools_result.output

    plan_result = commands.run("/plan")
    assert plan_result is not None
    assert "inspect" in plan_result.output


def test_skill_command_builds_agent_input(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Demo skill\n\nUse a short style for explanations.",
        encoding="utf-8",
    )

    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    tools = create_default_registry(ctx)
    skills = SkillCatalog.from_roots([tmp_path / "skills"])
    commands = create_default_commands(tools=tools, skills=skills)

    result = commands.run("/skill demo explain this file")
    assert result is not None
    assert "Running skill `demo`" in result.output
    assert result.agent_input is not None
    assert "Use a short style" in result.agent_input
    assert "explain this file" in result.agent_input


def test_unknown_command() -> None:
    ctx = ToolContext(cwd=Path.cwd(), ask_approval=lambda _a, _p: True)
    commands = create_default_commands(
        tools=create_default_registry(ctx),
        skills=SkillCatalog.from_roots([]),
    )
    result = commands.run("/does-not-exist")
    assert result is not None
    assert "Unknown command" in result.output


def test_delegate_commands() -> None:
    class FakeSubagents:
        def list_profiles(self):
            return [{"name": "general", "description": "General"}]

        def run(self, *, task: str, profile: str = "general") -> str:
            return f"{profile}:{task}"

    ctx = ToolContext(cwd=Path.cwd(), ask_approval=lambda _a, _p: True)
    commands = create_default_commands(
        tools=create_default_registry(ctx),
        skills=SkillCatalog.from_roots([]),
        subagents=FakeSubagents(),  # type: ignore[arg-type]
    )
    subagents_result = commands.run("/subagents")
    assert subagents_result is not None
    assert "general" in subagents_result.output

    delegate_result = commands.run("/delegate reviewer:: check tests")
    assert delegate_result is not None
    assert "reviewer:check tests" in delegate_result.output

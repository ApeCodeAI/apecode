"""Slash command framework for interactive use."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from apecode.skills import SkillCatalog
from apecode.subagents import SubagentProxy
from apecode.tools import ToolRegistry

CommandHandler = Callable[[str], "CommandResult"]


@dataclass(slots=True)
class CommandResult:
    """Result of one slash command execution."""

    output: str
    agent_input: str | None = None
    should_exit: bool = False


@dataclass(slots=True)
class SlashCommand:
    """Command metadata and handler."""

    name: str
    description: str
    usage: str
    handler: CommandHandler


class CommandRegistry:
    """Registry for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand, *, replace: bool = False) -> None:
        if not replace and command.name in self._commands:
            raise ValueError(f"command already registered: /{command.name}")
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def list_commands(self) -> list[SlashCommand]:
        return [self._commands[name] for name in sorted(self._commands)]

    def run(self, raw_input: str) -> CommandResult | None:
        if not raw_input.startswith("/"):
            return None
        payload = raw_input[1:].strip()
        if not payload:
            return CommandResult(output="Empty slash command. Use /help.")
        if " " in payload:
            name, args = payload.split(" ", 1)
            args = args.strip()
        else:
            name, args = payload, ""
        command = self.get(name)
        if command is None:
            return CommandResult(output=f"Unknown command: /{name}. Use /help.")
        return command.handler(args)


def create_template_command(
    *,
    name: str,
    description: str,
    usage: str,
    output: str,
    agent_input_template: str | None = None,
) -> SlashCommand:
    """Build a slash command from a simple text template."""

    def _handler(args: str) -> CommandResult:
        agent_input: str | None = None
        if agent_input_template is not None:
            agent_input = agent_input_template.replace("{args}", args.strip())
        return CommandResult(output=output, agent_input=agent_input)

    return SlashCommand(
        name=name,
        description=description,
        usage=usage,
        handler=_handler,
    )


def create_default_commands(
    *,
    tools: ToolRegistry,
    skills: SkillCatalog,
    subagents: SubagentProxy | None = None,
) -> CommandRegistry:
    """Build built-in slash commands."""
    registry = CommandRegistry()

    def _help(_args: str) -> CommandResult:
        lines = ["Available commands:"]
        for command in registry.list_commands():
            lines.append(f"- /{command.name}: {command.description} ({command.usage})")
        return CommandResult(output="\n".join(lines))

    def _tools(_args: str) -> CommandResult:
        names = tools.list_tool_names()
        if not names:
            return CommandResult(output="No tools registered.")
        return CommandResult(output="Tools:\n" + "\n".join(f"- {name}" for name in names))

    def _skills(_args: str) -> CommandResult:
        records = skills.list_skills()
        if not records:
            return CommandResult(output="No skills found.")
        lines = ["Skills:"]
        for skill in records:
            lines.append(f"- {skill.name}: {skill.description}")
        return CommandResult(output="\n".join(lines))

    def _skill(args: str) -> CommandResult:
        if not args:
            return CommandResult(output="Usage: /skill <name> [extra request]")
        parts = args.split(" ", 1)
        name = parts[0].strip().lower()
        extra = parts[1].strip() if len(parts) > 1 else ""
        skill = skills.get(name)
        if skill is None:
            return CommandResult(output=f"Skill not found: {name}")
        body = skill.read_text()
        if extra:
            body = f"{body}\n\nUser request:\n{extra}"
        return CommandResult(output=f"Running skill `{name}`...", agent_input=body)

    def _plan(_args: str) -> CommandResult:
        plan = tools.context.plan
        if not plan:
            return CommandResult(output="Plan is empty.")
        lines = ["Current plan:"]
        for item in plan:
            lines.append(f"- [{item['status']}] {item['step']}")
        return CommandResult(output="\n".join(lines))

    def _exit(_args: str) -> CommandResult:
        return CommandResult(output="Bye.", should_exit=True)

    def _subagents(_args: str) -> CommandResult:
        if subagents is None:
            return CommandResult(output="Subagents are disabled.")
        profiles = subagents.list_profiles()
        lines = ["Subagent profiles:"]
        for profile in profiles:
            lines.append(f"- {profile['name']}: {profile['description']}")
        return CommandResult(output="\n".join(lines))

    def _delegate(args: str) -> CommandResult:
        if subagents is None:
            return CommandResult(output="Subagents are disabled.")
        if not args.strip():
            return CommandResult(output="Usage: /delegate [profile::] <task>")
        if "::" in args:
            profile, task = args.split("::", 1)
            profile = profile.strip().lower()
            task = task.strip()
        else:
            profile, task = "general", args.strip()
        if not task:
            return CommandResult(output="Usage: /delegate [profile::] <task>")
        try:
            output = subagents.run(task=task, profile=profile)
        except ValueError as exc:
            return CommandResult(output=str(exc))
        except RuntimeError as exc:
            return CommandResult(output=f"Subagent execution failed: {exc}")
        return CommandResult(output=f"Subagent `{profile}`:\n{output}")

    registry.register(
        SlashCommand(
            name="help",
            description="Show available slash commands.",
            usage="/help",
            handler=_help,
        )
    )
    registry.register(
        SlashCommand(
            name="tools",
            description="List currently registered tools.",
            usage="/tools",
            handler=_tools,
        )
    )
    registry.register(
        SlashCommand(
            name="skills",
            description="List discovered skills.",
            usage="/skills",
            handler=_skills,
        )
    )
    registry.register(
        SlashCommand(
            name="skill",
            description="Run one skill as a prompt template.",
            usage="/skill <name> [extra request]",
            handler=_skill,
        )
    )
    registry.register(
        SlashCommand(
            name="plan",
            description="Print the latest in-memory plan.",
            usage="/plan",
            handler=_plan,
        )
    )
    registry.register(
        SlashCommand(
            name="subagents",
            description="List available subagent profiles.",
            usage="/subagents",
            handler=_subagents,
        )
    )
    registry.register(
        SlashCommand(
            name="delegate",
            description="Delegate a focused sub-task to a subagent.",
            usage="/delegate [profile::] <task>",
            handler=_delegate,
        )
    )
    registry.register(
        SlashCommand(
            name="exit",
            description="Exit current session.",
            usage="/exit",
            handler=_exit,
        )
    )
    return registry

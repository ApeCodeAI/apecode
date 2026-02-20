"""Minimal subagent delegation for ApeCode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apecode.agent import AgentCallbacks, AgentConfig, ChatModel, NanoCodeAgent
from apecode.tools import ToolContext, ToolRegistry


@dataclass(frozen=True, slots=True)
class SubagentProfile:
    """Role prompt used by delegated subagents."""

    name: str
    description: str
    prompt: str


DEFAULT_SUBAGENT_PROFILES: tuple[SubagentProfile, ...] = (
    SubagentProfile(
        name="general",
        description="General-purpose delegate for focused task execution.",
        prompt=("You are a delegated helper agent. Focus only on the assigned sub-task, keep answers concise, and report concrete results."),
    ),
    SubagentProfile(
        name="reviewer",
        description="Review code changes and identify bugs/risks.",
        prompt=("You are a code reviewer subagent. Prioritize correctness, regressions, and missing tests. Provide findings first, then a short summary."),
    ),
    SubagentProfile(
        name="researcher",
        description="Inspect codebase context and summarize findings.",
        prompt=("You are a research subagent. Gather high-signal facts from files/tools, state assumptions clearly, and return a structured summary."),
    ),
)


class SubagentRunner:
    """Executes delegated prompts with an isolated read-only tool runtime."""

    def __init__(
        self,
        *,
        model: ChatModel,
        parent_tools: ToolRegistry,
        base_system_prompt: str,
        max_steps: int = 8,
        profiles: tuple[SubagentProfile, ...] = DEFAULT_SUBAGENT_PROFILES,
        callbacks: AgentCallbacks | None = None,
    ) -> None:
        self._model = model
        self._parent_tools = parent_tools
        self._base_system_prompt = base_system_prompt
        self._max_steps = max(1, max_steps)
        self._profiles = {profile.name: profile for profile in profiles}
        self._callbacks = callbacks

    def list_profiles(self) -> list[SubagentProfile]:
        return [self._profiles[name] for name in sorted(self._profiles)]

    def run(self, *, task: str, profile: str = "general") -> str:
        profile_obj = self._profiles.get(profile)
        if profile_obj is None:
            raise ValueError(f"unknown subagent profile: {profile}")
        if not task.strip():
            raise ValueError("task cannot be empty")

        tools = self._build_subagent_tools()
        agent = NanoCodeAgent(
            model=self._model,
            tools=tools,
            system_prompt=(f"{self._base_system_prompt}\n\n# Subagent profile: {profile_obj.name}\n{profile_obj.prompt}\n"),
            config=AgentConfig(max_steps=self._max_steps),
            callbacks=self._callbacks,
        )
        return agent.run(task)

    def _build_subagent_tools(self) -> ToolRegistry:
        parent_context = self._parent_tools.context
        sub_context = ToolContext(
            cwd=parent_context.cwd,
            ask_approval=parent_context.ask_approval,
            sandbox_mode="read-only",
            approval_policy="never",
            plan=[],
        )
        registry = ToolRegistry(sub_context)
        for tool in self._parent_tools.list_tools():
            if tool.mutating:
                continue
            if tool.name in {"update_plan"}:
                continue
            registry.register(tool)
        return registry


class SubagentProxy:
    """Small adapter used by slash commands to decouple dependencies."""

    def __init__(self, runner: SubagentRunner):
        self._runner = runner

    def list_profiles(self) -> list[dict[str, Any]]:
        return [{"name": profile.name, "description": profile.description} for profile in self._runner.list_profiles()]

    def run(self, *, task: str, profile: str = "general") -> str:
        return self._runner.run(task=task, profile=profile)

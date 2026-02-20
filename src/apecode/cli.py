"""CLI entry point for ApeCode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from apecode import __version__
from apecode.agent import AgentCallbacks, AgentConfig, NanoCodeAgent
from apecode.commands import (
    CommandRegistry,
    create_default_commands,
    create_template_command,
)
from apecode.console import (
    InputSession,
    ask_approval,
    console,
    print_agent,
    print_error,
    print_plan,
    print_status,
    print_thinking,
    print_tool_call,
    print_tool_result,
    set_status,
)
from apecode.mcp import McpBridge, load_mcp_tools
from apecode.model_adapters import ModelError, create_model_client
from apecode.plugins import load_plugins
from apecode.skills import SkillCatalog
from apecode.subagents import SubagentProxy, SubagentRunner
from apecode.system_prompt import build_system_prompt
from apecode.tools import (
    ApprovalPolicy,
    SandboxMode,
    ToolContext,
    create_default_registry,
)


@dataclass(slots=True)
class AppRuntime:
    """Assembled runtime for CLI execution."""

    agent: NanoCodeAgent
    commands: CommandRegistry
    mcp_bridge: McpBridge | None = None

    def close(self) -> None:
        if self.mcp_bridge is not None:
            self.mcp_bridge.close()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"apecode {__version__}")
        raise typer.Exit()


def _approval_prompt(yolo_state: dict[str, bool], action: str, preview: str) -> bool:
    if yolo_state["enabled"]:
        return True
    result = ask_approval(action, preview)
    if result and preview == "":
        pass
    return result


def _collect_skill_roots(cwd: Path, arg_values: list[str]) -> list[Path]:
    roots = [Path(item).expanduser() for item in arg_values]
    roots.append(cwd / "skills")
    return roots


def _collect_mcp_configs(cwd: Path, arg_values: list[str]) -> list[Path]:
    paths = [Path(item).expanduser() for item in arg_values]
    paths.append(cwd / ".mcp.json")
    paths.append(cwd / "apecode_mcp.json")
    return paths


def _register_plugin_commands(runtime_commands: CommandRegistry, plugin_commands) -> tuple[int, list[str]]:
    loaded = 0
    errors: list[str] = []
    for spec in plugin_commands:
        try:
            runtime_commands.register(
                create_template_command(
                    name=spec.name,
                    description=f"[plugin:{spec.plugin_name}] {spec.description}",
                    usage=spec.usage,
                    output=spec.output,
                    agent_input_template=spec.agent_input_template,
                )
            )
            loaded += 1
        except ValueError as exc:
            errors.append(str(exc))
    return loaded, errors


def _make_callbacks(tool_context: ToolContext, *, indent: str = "") -> AgentCallbacks:
    """Build display callbacks. indent is used for subagent nesting."""

    def _on_tool_result(name: str, result: str) -> None:
        # Special display for plan updates
        if name == "update_plan":
            print_plan(tool_context.plan)
        else:
            print_tool_result(name, result)

    return AgentCallbacks(
        on_status=set_status,
        on_thinking=lambda text: print_thinking(text),
        on_tool_call=lambda name, args: print_tool_call(name, args),
        on_tool_result=_on_tool_result,
    )


def _build_runtime(
    *,
    provider: str,
    model: str,
    max_steps: int,
    timeout: int,
    temperature: float | None,
    cwd: Path,
    sandbox_mode: SandboxMode,
    approval_policy: ApprovalPolicy,
    yolo: bool,
    plugin_dirs: list[str],
    mcp_configs: list[str],
    skill_dirs: list[str],
) -> AppRuntime:
    if yolo:
        approval_policy = ApprovalPolicy.ALWAYS
    cwd = cwd.expanduser().resolve()
    yolo_state = {"enabled": yolo}
    tool_context = ToolContext(
        cwd=cwd,
        ask_approval=lambda action, preview: _approval_prompt(yolo_state, action, preview),
        sandbox_mode=sandbox_mode,
        approval_policy=approval_policy,
    )
    tools = create_default_registry(tool_context)
    plugin_dir_paths = [Path(item) for item in plugin_dirs]
    plugin_dir_paths.append(cwd / "plugins")
    plugin_result = load_plugins(tools, plugin_dir_paths)
    if plugin_result.tool_names:
        print_status(f"[plugin] loaded {len(plugin_result.tool_names)} tools")
    for error in plugin_result.errors:
        print_error(f"[plugin] {error}")

    mcp_config_paths = _collect_mcp_configs(cwd, mcp_configs)
    mcp_bridge = load_mcp_tools(tools, mcp_config_paths)
    if mcp_bridge.tool_names:
        print_status(f"[mcp] loaded {len(mcp_bridge.tool_names)} tools")
    for error in mcp_bridge.errors:
        print_error(f"[mcp] {error}")

    skill_roots = _collect_skill_roots(cwd, skill_dirs)
    skills = SkillCatalog.from_roots(skill_roots)
    if plugin_result.skills:
        initial_count = len(skills.list_skills())
        skills = skills.with_additional(plugin_result.skills)
        merged_count = len(skills.list_skills()) - initial_count
        if merged_count > 0:
            print_status(f"[plugin] loaded {merged_count} skills")
    _dir_entries: list[str] = []
    try:
        for item in sorted(cwd.iterdir()):
            _dir_entries.append(f"{item.name}{'/' if item.is_dir() else ''}")
    except OSError:
        pass
    _dir_listing = "\n".join(_dir_entries) if _dir_entries else None
    base_prompt = build_system_prompt(cwd, skills_overview=skills.format_for_system_prompt(), dir_listing=_dir_listing)

    model_client = create_model_client(
        provider=provider,
        model=model,
        timeout=timeout,
        temperature=temperature,
    )

    # Subagent callbacks show indented output
    sub_callbacks = _make_callbacks(tool_context, indent="    ")
    subagents = SubagentProxy(
        SubagentRunner(
            model=model_client,
            parent_tools=tools,
            base_system_prompt=base_prompt,
            max_steps=min(8, max(2, max_steps)),
            callbacks=sub_callbacks,
        )
    )
    commands = create_default_commands(tools=tools, skills=skills, subagents=subagents)
    loaded_command_count, command_errors = _register_plugin_commands(commands, plugin_result.commands)
    if loaded_command_count > 0:
        print_status(f"[plugin] loaded {loaded_command_count} commands")
    for error in command_errors:
        print_error(f"[plugin] {error}")
    return AppRuntime(
        agent=NanoCodeAgent(
            model=model_client,
            tools=tools,
            system_prompt=base_prompt,
            config=AgentConfig(max_steps=max(1, max_steps)),
            callbacks=_make_callbacks(tool_context),
        ),
        commands=commands,
        mcp_bridge=mcp_bridge,
    )


def _execute_agent_turn(agent: NanoCodeAgent, text: str) -> tuple[bool, str]:
    try:
        return True, agent.run(text)
    except (ModelError, RuntimeError) as exc:
        return False, str(exc)


def _run_repl(runtime: AppRuntime) -> int:
    print_status("ApeCode nano agent. Type /exit to quit. Alt+Enter for multi-line.")
    command_names = [cmd.name for cmd in runtime.commands.list_commands()]
    session = InputSession(command_names=command_names)
    while True:
        try:
            user_input = session.prompt()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return 0
        if not user_input:
            continue

        command_result = runtime.commands.run(user_input)
        if command_result is not None:
            print_agent(command_result.output)
            if command_result.should_exit:
                return 0
            if command_result.agent_input is None:
                continue
            ok, output = _execute_agent_turn(runtime.agent, command_result.agent_input)
            if ok:
                print_agent(output)
            else:
                print_error(output)
            continue

        ok, output = _execute_agent_turn(runtime.agent, user_input)
        if ok:
            print_agent(output)
        else:
            print_error(output)


app = typer.Typer(
    name="ape",
    help="ApeCode - nano terminal code agent",
    add_completion=False,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Annotated[
        list[str] | None,
        typer.Argument(help="One-shot prompt. If empty, starts REPL."),
    ] = None,
    provider: Annotated[str, typer.Option(envvar="APECODE_PROVIDER", help="Model provider.")] = "openai",
    model: Annotated[str, typer.Option(envvar="APECODE_MODEL", help="Model name.")] = "gpt-4.1-mini",
    max_steps: Annotated[int, typer.Option(help="Max agent loop steps.")] = 20,
    timeout: Annotated[int, typer.Option(help="Model request timeout in seconds.")] = 120,
    temperature: Annotated[
        float | None,
        typer.Option(help="Model temperature. Provider default if omitted."),
    ] = None,
    cwd: Annotated[str, typer.Option(help="Workspace directory.")] = "",
    sandbox_mode: Annotated[SandboxMode, typer.Option(envvar="APECODE_SANDBOX_MODE", help="Sandbox mode.")] = SandboxMode.WORKSPACE_WRITE,
    approval_policy: Annotated[
        ApprovalPolicy,
        typer.Option(envvar="APECODE_APPROVAL_POLICY", help="Approval policy."),
    ] = ApprovalPolicy.ON_REQUEST,
    plugin_dir: Annotated[list[str] | None, typer.Option(help="Plugin directory (can be repeated).")] = None,
    mcp_config: Annotated[
        list[str] | None,
        typer.Option(help="MCP config JSON file (can be repeated)."),
    ] = None,
    skill_dir: Annotated[
        list[str] | None,
        typer.Option(help="Skill root directory (can be repeated)."),
    ] = None,
    yolo: Annotated[bool, typer.Option("--yolo", help="Shortcut for --approval-policy always.")] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version.",
        ),
    ] = None,
) -> None:
    """ApeCode - nano terminal code agent."""
    workspace = Path(cwd) if cwd else Path.cwd()

    runtime: AppRuntime | None = None
    try:
        runtime = _build_runtime(
            provider=provider,
            model=model,
            max_steps=max_steps,
            timeout=timeout,
            temperature=temperature,
            cwd=workspace,
            sandbox_mode=sandbox_mode,
            approval_policy=approval_policy,
            yolo=yolo,
            plugin_dirs=plugin_dir or [],
            mcp_configs=mcp_config or [],
            skill_dirs=skill_dir or [],
        )
    except RuntimeError as exc:
        print_error(f"ApeCode setup error: {exc}")
        raise typer.Exit(code=1) from None

    try:
        prompt_text = " ".join(prompt).strip() if prompt else ""
        if not prompt_text:
            code = _run_repl(runtime)
            raise typer.Exit(code=code)

        command_result = runtime.commands.run(prompt_text)
        if command_result is not None:
            print_agent(command_result.output)
            if command_result.should_exit:
                raise typer.Exit()
            if command_result.agent_input is None:
                raise typer.Exit()
            prompt_text = command_result.agent_input

        ok, output = _execute_agent_turn(runtime.agent, prompt_text)
        if not ok:
            print_error(f"ApeCode runtime error: {output}")
            raise typer.Exit(code=2)
        print_agent(output)
    finally:
        if runtime is not None:
            runtime.close()

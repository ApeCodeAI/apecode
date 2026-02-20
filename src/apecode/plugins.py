"""Declarative plugin loader for tools, slash commands, and skills."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apecode.skills import Skill
from apecode.tools import Tool, ToolContext, ToolRegistry

PLUGIN_MANIFEST_NAME = "apecode_plugin.json"


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "item"


def _iter_manifest_files(plugin_dirs: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for raw_dir in plugin_dirs:
        plugin_dir = raw_dir.expanduser().resolve()
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            continue

        direct_manifest = plugin_dir / PLUGIN_MANIFEST_NAME
        if direct_manifest.exists() and direct_manifest.is_file():
            manifests.append(direct_manifest)

        for child in sorted(plugin_dir.iterdir()):
            if not child.is_dir():
                continue
            nested_manifest = child / PLUGIN_MANIFEST_NAME
            if nested_manifest.exists() and nested_manifest.is_file():
                manifests.append(nested_manifest)
    return manifests


@dataclass(frozen=True, slots=True)
class PluginToolSpec:
    """One plugin tool declaration."""

    plugin_name: str
    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool
    timeout_sec: int
    command: str | None = None
    argv: list[str] | None = None
    workdir: Path | None = None


@dataclass(frozen=True, slots=True)
class PluginCommandSpec:
    """One plugin slash command declaration."""

    plugin_name: str
    name: str
    description: str
    usage: str
    output: str
    agent_input_template: str | None = None


@dataclass(slots=True)
class PluginLoadResult:
    """Loaded plugin contributions."""

    tool_names: list[str] = field(default_factory=list)
    commands: list[PluginCommandSpec] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ParsedManifest:
    plugin_name: str
    tools: list[PluginToolSpec]
    commands: list[PluginCommandSpec]
    skills: list[Skill]


def _parse_tools(payload: dict[str, Any], *, manifest_path: Path, plugin_name: str) -> list[PluginToolSpec]:
    raw_tools = payload.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError("`tools` must be a list")

    tools: list[PluginToolSpec] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            raise ValueError("tool entry must be an object")

        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("tool.name is required")

        description = str(item.get("description", "")).strip() or f"Plugin tool `{name}`"
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("tool.parameters must be an object")

        command = str(item.get("command", "")).strip() or None
        argv_value = item.get("argv")
        argv = [str(value) for value in argv_value] if isinstance(argv_value, list) else None
        if not command and not argv:
            raise ValueError(f"tool `{name}` must provide `command` or `argv`")
        if command and argv:
            raise ValueError(f"tool `{name}` cannot provide both `command` and `argv`")

        timeout_sec = max(1, min(int(item.get("timeout_sec", 120)), 1800))
        tools.append(
            PluginToolSpec(
                plugin_name=plugin_name,
                name=name,
                description=description,
                parameters=parameters,
                mutating=bool(item.get("mutating", False)),
                timeout_sec=timeout_sec,
                command=command,
                argv=argv,
                workdir=manifest_path.parent,
            )
        )
    return tools


def _parse_commands(payload: dict[str, Any], *, plugin_name: str) -> list[PluginCommandSpec]:
    raw_commands = payload.get("commands", [])
    if not isinstance(raw_commands, list):
        raise ValueError("`commands` must be a list")

    commands: list[PluginCommandSpec] = []
    for item in raw_commands:
        if not isinstance(item, dict):
            raise ValueError("command entry must be an object")
        raw_name = str(item.get("name", "")).strip()
        if not raw_name:
            raise ValueError("command.name is required")
        name = _sanitize_name(raw_name)
        description = str(item.get("description", "")).strip() or f"Plugin command `{name}`"
        usage = str(item.get("usage", "")).strip() or f"/{name} [args]"
        output = str(item.get("output", "")).strip() or f"Running plugin command `/{name}`..."
        template = str(item.get("agent_input_template", "")).strip() or None
        commands.append(
            PluginCommandSpec(
                plugin_name=plugin_name,
                name=name,
                description=description,
                usage=usage,
                output=output,
                agent_input_template=template,
            )
        )
    return commands


def _parse_skills(payload: dict[str, Any], *, manifest_path: Path, plugin_name: str) -> list[Skill]:
    raw_skills = payload.get("skills", [])
    if not isinstance(raw_skills, list):
        raise ValueError("`skills` must be a list")

    skills: list[Skill] = []
    for item in raw_skills:
        if not isinstance(item, dict):
            raise ValueError("skill entry must be an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("skill.name is required")

        description = str(item.get("description", "")).strip()
        inline_content = item.get("content")
        file_path = str(item.get("file", "")).strip()

        content_text: str
        path: Path | None = None
        if isinstance(inline_content, str) and inline_content.strip():
            content_text = inline_content.strip()
        elif file_path:
            path = (manifest_path.parent / file_path).resolve()
            if not path.exists() or not path.is_file():
                raise ValueError(f"skill `{name}` file not found: {file_path}")
            content_text = path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            raise ValueError(f"skill `{name}` requires `content` or `file`")

        derived_description = "No description."
        for raw_line in content_text.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                derived_description = line[:160]
                break

        skills.append(
            Skill(
                name=name,
                description=description or derived_description,
                path=path,
                inline_content=None if path is not None else content_text,
                source=f"plugin:{plugin_name}",
            )
        )
    return skills


def _parse_manifest(manifest_path: Path) -> _ParsedManifest:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    plugin_name = str(payload.get("name", manifest_path.parent.name)).strip() or manifest_path.parent.name
    return _ParsedManifest(
        plugin_name=plugin_name,
        tools=_parse_tools(payload, manifest_path=manifest_path, plugin_name=plugin_name),
        commands=_parse_commands(payload, plugin_name=plugin_name),
        skills=_parse_skills(payload, manifest_path=manifest_path, plugin_name=plugin_name),
    )


def _build_tool_handler(spec: PluginToolSpec):
    def _handler(_ctx: ToolContext, args: dict[str, Any]) -> str:
        if spec.argv:
            command: str | list[str] = spec.argv
            shell = False
        else:
            command = spec.command or ""
            shell = True

        proc = subprocess.run(
            command,
            shell=shell,
            cwd=spec.workdir,
            input=json.dumps(args, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=spec.timeout_sec,
            check=False,
        )
        output = (proc.stdout or "").strip()
        error_text = (proc.stderr or "").strip()
        if proc.returncode != 0:
            detail = error_text or output or f"exit_code={proc.returncode}"
            return f"plugin `{spec.plugin_name}` tool `{spec.name}` failed: {detail}"
        if not output:
            return f"plugin `{spec.plugin_name}` tool `{spec.name}` finished with empty output"
        if len(output) > 8000:
            return output[:8000] + "\n... (truncated)"
        return output

    return _handler


def load_plugins(registry: ToolRegistry, plugin_dirs: list[Path]) -> PluginLoadResult:
    """Load plugins and register declarative plugin tools."""
    result = PluginLoadResult()
    existing_tool_names = set(registry.list_tool_names())

    for manifest in _iter_manifest_files(plugin_dirs):
        try:
            parsed = _parse_manifest(manifest)
        except Exception as exc:
            result.errors.append(f"invalid plugin manifest `{manifest}`: {exc}")
            continue

        for spec in parsed.tools:
            namespaced = f"{_sanitize_name(spec.plugin_name)}__{_sanitize_name(spec.name)}"
            if namespaced in existing_tool_names:
                result.errors.append(f"duplicate plugin tool ignored: {namespaced}")
                continue
            registry.register(
                Tool(
                    name=namespaced,
                    description=f"[plugin:{spec.plugin_name}] {spec.description}",
                    parameters=spec.parameters,
                    handler=_build_tool_handler(spec),
                    mutating=spec.mutating,
                )
            )
            existing_tool_names.add(namespaced)
            result.tool_names.append(namespaced)

        result.commands.extend(parsed.commands)
        result.skills.extend(parsed.skills)

    return result

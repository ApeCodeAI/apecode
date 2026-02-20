"""Minimal MCP integration via fastmcp Client."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import Client

from apecode.tools import Tool, ToolRegistry


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "tool"


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """One MCP stdio server entry."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    timeout_sec: int = 30


@dataclass(slots=True)
class McpBridge:
    """Loaded MCP registration result."""

    tool_names: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def close(self) -> None:
        """No-op: sessions are opened per request in this nano implementation."""


def _parse_mcp_config(path: Path) -> list[McpServerConfig]:
    """Parse `.mcp.json` with `mcpServers` entries."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    servers: list[McpServerConfig] = []

    raw_mcp_servers = payload.get("mcpServers")
    if isinstance(raw_mcp_servers, dict):
        for name, item in raw_mcp_servers.items():
            if not isinstance(item, dict):
                continue
            command = str(item.get("command", "")).strip()
            if not command:
                continue
            raw_args = item.get("args", [])
            args = [str(value) for value in raw_args] if isinstance(raw_args, list) else []
            timeout_sec = max(5, min(int(item.get("timeout_sec", 30)), 300))
            servers.append(
                McpServerConfig(
                    name=str(name).strip(),
                    command=command,
                    args=args,
                    timeout_sec=timeout_sec,
                )
            )

    return servers


def _make_client(server: McpServerConfig) -> Client:
    """Create a fastmcp Client for a single MCP server."""
    config = {"mcpServers": {server.name: {"command": server.command, "args": server.args}}}
    return Client(config)


async def _list_tools_async(server: McpServerConfig) -> list[Any]:
    async with _make_client(server) as client:
        return await client.list_tools()


async def _call_tool_async(server: McpServerConfig, tool_name: str, arguments: dict[str, Any]) -> Any:
    async with _make_client(server) as client:
        return await client.call_tool(tool_name, arguments)


def _render_tool_result(result: Any, *, server_name: str, tool_name: str) -> str:
    is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    content = list(getattr(result, "content", []) or [])

    chunks: list[str] = []
    for item in content:
        item_type = str(getattr(item, "type", ""))
        if item_type == "text":
            text = str(getattr(item, "text", ""))
            if text:
                chunks.append(text)
            continue
        if hasattr(item, "model_dump"):
            chunks.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
        else:
            chunks.append(str(item))

    rendered = "\n".join(part for part in chunks if part.strip()).strip()
    if not rendered:
        rendered = f"MCP `{server_name}/{tool_name}` returned empty result."
    if is_error:
        return f"MCP `{server_name}/{tool_name}` failed: {rendered}"
    return rendered


def load_mcp_tools(registry: ToolRegistry, config_paths: list[Path]) -> McpBridge:
    """Load MCP tools from config files and register them into ToolRegistry."""
    bridge = McpBridge()
    seen_servers: set[str] = set()

    for raw_path in config_paths:
        path = raw_path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            continue
        try:
            servers = _parse_mcp_config(path)
        except Exception as exc:
            bridge.errors.append(f"invalid MCP config `{path}`: {exc}")
            continue

        for server in servers:
            if server.name in seen_servers:
                continue
            seen_servers.add(server.name)
            try:
                tools = asyncio.run(asyncio.wait_for(_list_tools_async(server), timeout=server.timeout_sec))
            except Exception as exc:
                bridge.errors.append(f"MCP server `{server.name}` unavailable: {exc}")
                continue

            for raw_tool in tools:
                raw_name = str(getattr(raw_tool, "name", "")).strip()
                if not raw_name:
                    continue

                namespaced = f"mcp__{_sanitize_name(server.name)}__{_sanitize_name(raw_name)}"
                description = str(getattr(raw_tool, "description", "")).strip()
                schema = getattr(raw_tool, "inputSchema", None)
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}

                annotations = getattr(raw_tool, "annotations", None)
                read_only = bool(getattr(annotations, "readOnlyHint", False)) if annotations else False

                def _handler(_ctx, args, *, _server=server, _tool_name=raw_name):
                    try:
                        result = asyncio.run(
                            asyncio.wait_for(
                                _call_tool_async(_server, _tool_name, args),
                                timeout=_server.timeout_sec,
                            )
                        )
                    except Exception as exc:
                        return f"MCP `{_server.name}/{_tool_name}` invocation error: {exc}"
                    return _render_tool_result(
                        result,
                        server_name=_server.name,
                        tool_name=_tool_name,
                    )

                registry.register(
                    Tool(
                        name=namespaced,
                        description=description or f"[mcp:{server.name}] call MCP tool `{raw_name}`",
                        parameters=schema,
                        handler=_handler,
                        mutating=not read_only,
                    )
                )
                bridge.tool_names.append(namespaced)

    return bridge

"""Tests for minimal MCP stdio loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from apecode.mcp import load_mcp_tools
from apecode.tools import ToolContext, create_default_registry


def test_load_and_execute_mcp_tool(tmp_path: Path) -> None:
    server_script = tmp_path / "fake_mcp_server.py"
    server_script.write_text(
        (
            "from fastmcp import FastMCP\n"
            "mcp=FastMCP('demo')\n"
            "@mcp.tool(description='Echo text')\n"
            "def echo(text: str) -> str:\n"
            "  return 'mcp:'+text\n"
            "if __name__=='__main__':\n"
            "  mcp.run(transport='stdio')\n"
        ),
        encoding="utf-8",
    )
    config = tmp_path / "apecode_mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "demo": {
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    registry = create_default_registry(ctx)
    bridge = load_mcp_tools(registry, [config])
    try:
        assert len(bridge.tool_names) == 1
        tool_name = bridge.tool_names[0]
        result = registry.execute(tool_name, json.dumps({"text": "hello"}))
        assert "mcp:hello" in result
    finally:
        bridge.close()

"""Tests for declarative plugin loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from apecode.plugins import load_plugins
from apecode.skills import SkillCatalog
from apecode.tools import ToolContext, create_default_registry


def test_load_plugin_tool_command_and_skill(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "echo"
    plugin_dir.mkdir(parents=True)
    script = plugin_dir / "tool.py"
    script.write_text(
        ("import json,sys\nargs=json.loads(sys.stdin.read() or '{}')\nprint('echo:'+str(args.get('text','')))\n"),
        encoding="utf-8",
    )
    manifest = plugin_dir / "apecode_plugin.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "EchoPlugin",
                "tools": [
                    {
                        "name": "echo_text",
                        "description": "Echo input text",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                        "argv": [sys.executable, str(script)],
                    }
                ],
                "commands": [
                    {
                        "name": "quick-review",
                        "description": "Review with plugin template",
                        "usage": "/quick-review <task>",
                        "output": "Running quick review...",
                        "agent_input_template": "Review task:\n{args}",
                    }
                ],
                "skills": [
                    {
                        "name": "plugin-skill",
                        "description": "Loaded from plugin manifest",
                        "content": "# Plugin skill\n\nKeep output concise.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    registry = create_default_registry(ctx)
    result = load_plugins(registry, [tmp_path / "plugins"])
    assert len(result.tool_names) == 1
    assert len(result.commands) == 1
    assert len(result.skills) == 1
    assert result.errors == []

    tool_output = registry.execute(result.tool_names[0], json.dumps({"text": "hello"}))
    assert tool_output == "echo:hello"

    merged_skills = SkillCatalog.from_roots([]).with_additional(result.skills)
    skill = merged_skills.get("plugin-skill")
    assert skill is not None
    assert "Keep output concise." in skill.read_text()


def test_invalid_plugin_manifest_is_reported(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "broken"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "apecode_plugin.json").write_text(
        json.dumps({"name": "Broken", "tools": [{"name": "missing-command"}]}),
        encoding="utf-8",
    )

    ctx = ToolContext(cwd=tmp_path, ask_approval=lambda _a, _p: True)
    registry = create_default_registry(ctx)
    result = load_plugins(registry, [tmp_path / "plugins"])
    assert result.tool_names == []
    assert result.commands == []
    assert result.skills == []
    assert len(result.errors) == 1

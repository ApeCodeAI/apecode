"""Tooling layer for the nano code agent."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

ApprovalCallback = Callable[[str, str], bool]


class SandboxMode(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class ApprovalPolicy(StrEnum):
    ON_REQUEST = "on-request"
    ALWAYS = "always"
    NEVER = "never"


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


@dataclass(slots=True)
class ToolContext:
    """Shared mutable context for tool handlers."""

    cwd: Path
    ask_approval: ApprovalCallback
    sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    approval_policy: ApprovalPolicy = ApprovalPolicy.ON_REQUEST
    plan: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cwd = self.cwd.expanduser().resolve()
        self.sandbox_mode = SandboxMode(self.sandbox_mode)
        self.approval_policy = ApprovalPolicy(self.approval_policy)

    def resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (self.cwd / candidate).resolve()
        if self.sandbox_mode != "danger-full-access" and not _is_within(self.cwd, resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved


ToolHandler = Callable[[ToolContext, dict[str, Any]], str]


@dataclass(slots=True)
class Tool:
    """Function tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    mutating: bool = False


class ToolRegistry:
    """Runtime registry for all callable tools."""

    def __init__(self, context: ToolContext):
        self.context = context
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        """Return registered tools sorted by name."""
        return [self._tools[name] for name in sorted(self._tools)]

    def list_tool_names(self) -> list[str]:
        """Return registered tool names sorted alphabetically."""
        return [tool.name for tool in self.list_tools()]

    def as_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.list_tools()
        ]

    def execute(self, name: str, arguments_json: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}"

        try:
            arguments = json.loads(arguments_json or "{}")
            if not isinstance(arguments, dict):
                return "Tool arguments must be a JSON object."
        except json.JSONDecodeError as exc:
            return f"Invalid tool arguments JSON: {exc}"

        if tool.mutating:
            if self.context.sandbox_mode == "read-only":
                return "blocked by sandbox policy: read-only"

            if self.context.approval_policy == "never":
                return "blocked by approval policy: never"

            if self.context.approval_policy == "on-request":
                preview = json.dumps(arguments, ensure_ascii=True, indent=2)[:600]
                if not self.context.ask_approval(f"{name}", preview):
                    return "Rejected by user."

        try:
            return tool.handler(self.context, arguments)
        except Exception as exc:  # pragma: no cover - defensive
            return f"Tool execution failed: {exc}"


def _list_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args.get("path", "."))
    recursive = bool(args.get("recursive", True))
    max_entries = int(args.get("max_entries", 200))
    max_entries = max(1, min(max_entries, 2000))
    root = ctx.resolve_path(raw_path)
    if not root.exists():
        return f"path does not exist: {root}"
    if root.is_file():
        return str(root.relative_to(ctx.cwd))

    entries: list[str] = []
    iterator = sorted(root.rglob("*")) if recursive else sorted(root.iterdir())
    for item in iterator:
        rel = item.relative_to(ctx.cwd)
        entries.append(f"{rel}/" if item.is_dir() else str(rel))
        if len(entries) >= max_entries:
            entries.append(f"... truncated at {max_entries} entries")
            break
    return "\n".join(entries) if entries else "(empty directory)"


def _read_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args["path"])
    start_line = int(args.get("start_line", 1))
    num_lines = int(args.get("num_lines", 200))
    start_line = max(1, start_line)
    num_lines = max(1, min(num_lines, 2000))
    path = ctx.resolve_path(raw_path)
    if not path.exists() or not path.is_file():
        return f"file not found: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start_idx = start_line - 1
    chunk = lines[start_idx : start_idx + num_lines]
    if not chunk:
        return "(no content)"
    rendered = [f"{line_number:>6}\t{line_text}" for line_number, line_text in enumerate(chunk, start=start_line)]
    return "\n".join(rendered)


def _write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args["path"])
    content = str(args.get("content", ""))
    mode = str(args.get("mode", "overwrite"))
    if mode not in {"overwrite", "append"}:
        return "mode must be one of: overwrite, append"
    path = ctx.resolve_path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "overwrite":
        path.write_text(content, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
    return f"wrote {len(content)} bytes to {path.relative_to(ctx.cwd)} ({mode})"


def _replace_in_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_path = str(args["path"])
    old = str(args["old"])
    new = str(args["new"])
    count = int(args.get("count", 1))
    count = max(1, count)
    path = ctx.resolve_path(raw_path)
    if not path.exists() or not path.is_file():
        return f"file not found: {path}"
    content = path.read_text(encoding="utf-8", errors="replace")
    replaced, n = content.replace(old, new, count), content.count(old)
    if n == 0:
        return "no replacements made"
    path.write_text(replaced, encoding="utf-8")
    return f"applied replacements in {path.relative_to(ctx.cwd)}"


def _grep_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    pattern = str(args["pattern"])
    raw_path = str(args.get("path", "."))
    include = args.get("glob")
    max_results = int(args.get("max_results", 200))
    max_results = max(1, min(max_results, 2000))
    root = ctx.resolve_path(raw_path)
    if shutil.which("rg"):
        cmd = ["rg", "--line-number", "--no-heading", pattern, str(root)]
        if include:
            cmd.extend(["-g", str(include)])
        proc = subprocess.run(
            cmd,
            cwd=ctx.cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode not in (0, 1):
            return f"rg failed: {proc.stderr.strip()}"
        lines = proc.stdout.splitlines()[:max_results]
        return "\n".join(lines) if lines else "(no matches)"

    matches: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                matches.append(f"{path.relative_to(ctx.cwd)}:{line_no}:{line}")
                if len(matches) >= max_results:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "(no matches)"


def _exec_command(ctx: ToolContext, args: dict[str, Any]) -> str:
    command = str(args["command"])
    timeout = int(args.get("timeout_sec", 120))
    timeout = max(1, min(timeout, 1800))
    proc = subprocess.run(
        command,
        shell=True,
        cwd=ctx.cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    output = output.strip()
    if len(output) > 6000:
        output = output[:6000] + "\n... (truncated)"
    return f"exit_code={proc.returncode}\n{output}"


def _update_plan(ctx: ToolContext, args: dict[str, Any]) -> str:
    plan = args.get("plan")
    if not isinstance(plan, list):
        return "plan must be a list of {step, status}"
    normalized: list[dict[str, str]] = []
    for item in plan:
        if not isinstance(item, dict):
            return "each plan item must be an object"
        step = str(item.get("step", "")).strip()
        status = str(item.get("status", "")).strip()
        if not step:
            return "plan step cannot be empty"
        if status not in {"pending", "in_progress", "completed"}:
            return "status must be pending | in_progress | completed"
        normalized.append({"step": step, "status": status})
    ctx.plan = normalized
    return json.dumps({"ok": True, "plan_size": len(ctx.plan)})


def create_default_registry(context: ToolContext) -> ToolRegistry:
    """Build default nano agent tools."""
    registry = ToolRegistry(context)
    registry.register(
        Tool(
            name="list_files",
            description=(
                "List files and directories under a given path. "
                "Returns one entry per line; directories have a trailing slash. "
                "Prefer this tool over exec_command with ls or find for directory exploration. "
                "By default lists recursively up to 200 entries. "
                "Use recursive=false for a shallow listing of large directories."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to list. Defaults to the workspace root ('.').",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true (default), list all files recursively. Set to false for a shallow listing.",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum number of entries to return. Defaults to 200. Range: 1-2000.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=_list_files,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description=(
                "Read the contents of a file with line numbers. "
                "Output format is line-numbered (6-digit padded line number followed by a tab and the line content). "
                "Prefer this tool over exec_command with cat, head, or tail. "
                "Reads up to 200 lines by default starting from line 1. "
                "Use start_line and num_lines to read specific sections of large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file to read.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-based line number to start reading from. Defaults to 1.",
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": "Number of lines to read. Defaults to 200. Maximum: 2000.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_read_file,
        )
    )
    registry.register(
        Tool(
            name="grep_files",
            description=(
                "Search for a text pattern across files. Uses ripgrep when available, "
                "falling back to a built-in scanner. Supports full regex syntax. "
                "Prefer this tool over exec_command with grep or rg. "
                "Output format is file:line_number:matching_line. "
                "Use the glob parameter to restrict to specific file types (e.g., '*.py', '*.{ts,tsx}')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for in file contents.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Defaults to the workspace root ('.').",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter which files are searched (e.g., '*.py', '*.{js,ts}', 'src/**/*.rs').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return. Defaults to 200. Range: 1-2000.",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            handler=_grep_files,
        )
    )
    registry.register(
        Tool(
            name="write_file",
            description=(
                "Write content to a file. MUTATING operation — creates a new file or overwrites/appends to an existing one. "
                "Parent directories are created automatically if they do not exist. "
                "Prefer replace_in_file for targeted edits to existing files. "
                "Use this tool when creating new files or when the entire file content needs to be replaced."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode. 'overwrite' (default) replaces the entire file. 'append' adds content to the end.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_write_file,
            mutating=True,
        )
    )
    registry.register(
        Tool(
            name="replace_in_file",
            description=(
                "Replace exact text in an existing file. PREFERRED tool for editing existing files — "
                "always use read_file first to see the current content, then specify the exact old text to replace. "
                "MUTATING operation. The old string must match exactly (including whitespace and indentation). "
                "If no match is found, no changes are made."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file to edit.",
                    },
                    "old": {
                        "type": "string",
                        "description": "The exact text to find and replace. Must match the file content exactly, including whitespace.",
                    },
                    "new": {
                        "type": "string",
                        "description": "The replacement text. Can be empty to delete the matched text.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum number of occurrences to replace. Defaults to 1.",
                    },
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
            handler=_replace_in_file,
            mutating=True,
        )
    )
    registry.register(
        Tool(
            name="exec_command",
            description=(
                "Execute a shell command and return its output (stdout + stderr) and exit code. "
                "MUTATING operation. Use this for tasks like running tests, git operations, build commands, "
                "installing packages, or other system commands. "
                "Do NOT use this for file listing (use list_files), reading files (use read_file), "
                "or searching file contents (use grep_files) — dedicated tools are faster and safer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute. Runs via the system shell.",
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Timeout in seconds. Defaults to 120. Range: 1-1800.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            handler=_exec_command,
            mutating=True,
        )
    )
    registry.register(
        Tool(
            name="update_plan",
            description=(
                "Create or update a lightweight task plan for the current session. "
                "Use this for tasks that involve 3 or more steps to help track progress. "
                "Each call replaces the entire plan — always include all steps with their current statuses. "
                "The plan is displayed to the user for visibility."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "description": "The complete list of plan steps. Each call replaces the entire plan.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {
                                    "type": "string",
                                    "description": "A concise description of this task step.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Current status of this step.",
                                },
                            },
                            "required": ["step", "status"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
            handler=_update_plan,
        )
    )
    return registry

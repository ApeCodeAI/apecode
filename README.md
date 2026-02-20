# ApeCode 🦧

A nano terminal code agent in Python — a minimal but complete implementation of a tool-calling AI agent (like Claude Code / Codex CLI / Kimi CLI), built for learning and experimentation.

Powered by [ApeCode.ai](https://apecode.ai)

## Features

- **Tool-calling agent loop** — `user → model → tool calls → tool results → model → response`, with configurable max-steps guard
- **Multi-provider model adapters** — OpenAI, Anthropic, and Kimi (OpenAI-compatible), all conforming to a unified `ChatModel` protocol
- **7 built-in tools** — `list_files`, `read_file`, `write_file`, `replace_in_file`, `grep_files`, `exec_command`, `update_plan`
- **Sandbox + approval model** — `SandboxMode` (read-only / workspace-write / danger-full-access) restricts path mutations; `ApprovalPolicy` (on-request / always / never) controls interactive confirmation for mutating operations
- **Plugin system** — declarative `apecode_plugin.json` manifests contribute tools, slash commands, and skills
- **MCP integration** — load external tools from `.mcp.json` / `apecode_mcp.json` via the `fastmcp` SDK
- **Slash commands** — `/help`, `/tools`, `/skills`, `/skill`, `/plan`, `/subagents`, `/delegate`, `/exit`
- **Subagent delegation** — isolated read-only agents with three default profiles: `general`, `reviewer`, `researcher`
- **Skill templates** — discoverable from `skills/*/SKILL.md` directories or plugins
- **REPL + one-shot mode** — interactive session with prompt-toolkit (history, tab-completion, multi-line via Alt+Enter) or single-prompt execution
- **Thinking model support** — displays `reasoning_content` from thinking models (e.g. Kimi K2.5)
- **AGENTS.md chain** — walks from workspace root to filesystem root, loading `AGENTS.md` files for project-specific instructions

## Installation

```bash
uv sync
```

Dependencies: `openai`, `anthropic`, `fastmcp`, `typer`, `rich`, `prompt-toolkit`.

## Usage

### API keys

```bash
export OPENAI_API_KEY=your_key       # for provider=openai (default)
export ANTHROPIC_API_KEY=your_key    # for provider=anthropic
export KIMI_API_KEY=your_key         # for provider=kimi
```

### Interactive REPL

```bash
uv run ape
```

### One-shot mode

```bash
uv run ape "read README.md and summarize project structure"
```

### CLI flags

```bash
uv run ape --provider openai --model gpt-4.1-mini    # default
uv run ape --provider anthropic --model claude-sonnet-4-20250514
uv run ape --provider kimi --model kimi-k2.5
uv run ape --max-steps 30 --timeout 180
uv run ape --cwd /path/to/repo
uv run ape --sandbox-mode read-only --approval-policy never
uv run ape --plugin-dir ./plugins
uv run ape --mcp-config ./.mcp.json
uv run ape --skill-dir ./custom-skills
uv run ape --yolo "apply a simple refactor in src/"
uv run ape --version
```

### Slash commands (inside REPL)

```
/help                                          — list all commands
/tools                                         — list registered tools
/skills                                        — list discovered skills
/skill concise-review review src/apecode/cli.py — run a skill with extra request
/plan                                          — show the current task plan
/subagents                                     — list subagent profiles
/delegate reviewer:: review src/apecode/cli.py — delegate to a subagent
/exit                                          — quit
```

## Architecture

```
user input
    │
    ▼
┌──────────────────────────────────────────────┐
│  cli.py — Typer app, _build_runtime, REPL    │
│  ┌────────────────────────────────────────┐  │
│  │  NanoCodeAgent (agent.py)              │  │
│  │  ┌──────────┐    ┌──────────────────┐  │  │
│  │  │ ChatModel │◄──│ model_adapters.py │  │  │
│  │  │ protocol  │   │ OpenAI/Anthropic/ │  │  │
│  │  │           │   │ Kimi adapters     │  │  │
│  │  └──────────┘    └──────────────────┘  │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │ ToolRegistry (tools.py)          │  │  │
│  │  │  7 built-in + plugin + MCP tools │  │  │
│  │  │  ToolContext: sandbox + approval  │  │  │
│  │  └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────┘  │
│  commands.py — slash command registry        │
│  plugins.py  — apecode_plugin.json loader    │
│  mcp.py      — fastmcp stdio bridge          │
│  skills.py   — SKILL.md discovery + catalog  │
│  subagents.py — isolated read-only delegates │
│  system_prompt.py — prompt builder + AGENTS.md│
│  console.py  — Rich + prompt-toolkit I/O     │
└──────────────────────────────────────────────┘
```

### Module breakdown

| Module | Purpose |
|---|---|
| `cli.py` | Typer entry point, assembles runtime (`_build_runtime`), runs REPL or one-shot |
| `agent.py` | `NanoCodeAgent` — the core tool-calling loop with `ChatModel` protocol |
| `tools.py` | `ToolRegistry`, `ToolContext` (sandbox/approval), 7 built-in tool handlers |
| `model_adapters.py` | `OpenAIChatCompletionsClient`, `AnthropicMessagesClient`, `KimiChatCompletionsClient` — all adapters convert to/from internal OpenAI message format |
| `commands.py` | `CommandRegistry` + `SlashCommand` — `/help`, `/tools`, `/exit`, etc. |
| `plugins.py` | Loads `apecode_plugin.json` manifests; registers tools, commands, skills |
| `mcp.py` | Parses `.mcp.json`, connects via `fastmcp.Client`, registers MCP tools |
| `skills.py` | `SkillCatalog` — discovers `SKILL.md` files, supports plugin-contributed skills |
| `subagents.py` | `SubagentRunner` — spawns isolated agents with read-only tools and capped steps |
| `system_prompt.py` | Builds system prompt with environment info, AGENTS.md chain, skill catalog |
| `console.py` | Rich console output (panels, spinners, tool call display) + prompt-toolkit input session |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APECODE_PROVIDER` | `openai` | Model provider (`openai` / `anthropic` / `kimi`) |
| `APECODE_MODEL` | `gpt-4.1-mini` | Model name |
| `APECODE_SANDBOX_MODE` | `workspace-write` | Sandbox mode (`read-only` / `workspace-write` / `danger-full-access`) |
| `APECODE_APPROVAL_POLICY` | `on-request` | Approval policy (`on-request` / `always` / `never`) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Custom OpenAI-compatible endpoint |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | Custom Anthropic endpoint |
| `ANTHROPIC_API_VERSION` | `2023-06-01` | Anthropic API version header |
| `KIMI_API_KEY` | — | Kimi API key |
| `KIMI_BASE_URL` | `https://api.moonshot.cn/v1` | Kimi endpoint |

## Plugin System

Place a plugin manifest as `apecode_plugin.json` in a plugin directory:

```json
{
  "name": "EchoPlugin",
  "tools": [
    {
      "name": "echo_text",
      "description": "Echo text from JSON args",
      "parameters": {
        "type": "object",
        "properties": { "text": { "type": "string" } },
        "required": ["text"],
        "additionalProperties": false
      },
      "argv": ["python3", "/absolute/path/to/tool.py"],
      "mutating": false,
      "timeout_sec": 60
    }
  ],
  "commands": [
    {
      "name": "quick-review",
      "description": "Run plugin prompt template",
      "usage": "/quick-review <task>",
      "output": "Running quick review...",
      "agent_input_template": "Review this task:\\n{args}"
    }
  ],
  "skills": [
    {
      "name": "plugin-skill",
      "description": "A plugin-provided skill",
      "content": "# Plugin Skill\\n\\nKeep output concise."
    }
  ]
}
```

- Tools use either `argv` (recommended) or `command` to specify the executable.
- Tool processes receive JSON arguments on `stdin` and write results to `stdout`.
- Commands support `{args}` placeholder in `agent_input_template`.
- Skills can use inline `content` or a `file` path relative to the manifest.

## MCP Config

Load MCP tools from `.mcp.json` or `apecode_mcp.json` in workspace root, or via `--mcp-config`:

```json
{
  "mcpServers": {
    "demo": {
      "command": "python3",
      "args": ["/absolute/path/to/mcp_server.py"],
      "timeout_sec": 30
    }
  }
}
```

## Skills

Create a skill as `skills/<name>/SKILL.md`:

```markdown
# concise-review

Review code and answer with concise bullet points.
```

Use inside REPL:

```
/skill concise-review review src/apecode/agent.py
```

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_tools.py -v

# Lint
uv run ruff check src/ tests/

# Lint with auto-fix
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/
```

## Project Structure

```
src/apecode/
├── __init__.py          # package version
├── __main__.py          # python -m apecode entry
├── cli.py               # Typer CLI app + runtime assembly
├── agent.py             # NanoCodeAgent core loop
├── tools.py             # tool registry + built-in tools
├── model_adapters.py    # model adapters (OpenAI/Anthropic/Kimi)
├── commands.py          # slash command framework
├── plugins.py           # plugin manifest loader
├── mcp.py               # MCP stdio bridge
├── skills.py            # skill discovery + catalog
├── subagents.py         # subagent delegation
├── system_prompt.py     # system prompt builder
└── console.py           # Rich + prompt-toolkit I/O
tests/
├── test_agent.py
├── test_commands.py
├── test_mcp.py
├── test_model_adapters.py
├── test_plugins.py
├── test_skills.py
├── test_subagents.py
└── test_tools.py
```

## License

Apache-2.0

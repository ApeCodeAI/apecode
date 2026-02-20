"""Centralized Rich console + prompt_toolkit helpers for terminal I/O."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# ── Rich output helpers ─────────────────────────────────────────────


def print_agent(text: str) -> None:
    """Render agent response as Markdown inside a panel."""
    console.print(Panel(Markdown(text), title="ape", border_style="green"))


def print_error(text: str) -> None:
    """Print an error message in red."""
    console.print(f"[bold red]error>[/bold red] {text}")


def print_status(text: str) -> None:
    """Print a dim status line."""
    if text:
        console.print(f"[dim]{text}[/dim]")


# ── Live spinner ─────────────────────────────────────────────────────

_active_spinner = None


def set_status(text: str) -> None:
    """Start or stop a live spinner. Empty string stops it."""
    global _active_spinner
    if _active_spinner is not None:
        _active_spinner.__exit__(None, None, None)
        _active_spinner = None
    if text:
        _active_spinner = console.status(f"[bold green]{text}[/bold green]")
        _active_spinner.__enter__()


def ask_approval(action: str, preview: str) -> bool:
    """Styled approval prompt for mutating tool calls."""
    console.print(Panel(preview, title=f"[yellow]approve: {action}[/yellow]", border_style="yellow"))
    answer = console.input("[yellow]Approve? [y/N/a=always][/yellow] ").strip().lower()
    if answer == "a":
        return True  # caller handles "always" state
    return answer in {"y", "yes", "a"}


@contextmanager
def status_spinner(text: str = "Thinking...") -> Generator[None]:
    """Context manager for a model thinking spinner."""
    with console.status(f"[bold green]{text}[/bold green]"):
        yield


# ── Agent event display ──────────────────────────────────────────────


def _extract_key_arg(name: str, arguments_json: str) -> str:
    """Pull out the most interesting argument for one-line display."""
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(args, dict):
        return ""
    # Prioritized keys — first match wins
    for key in ("path", "command", "pattern", "plan", "content"):
        if key in args:
            val = args[key]
            if isinstance(val, str):
                return val[:60]
            if isinstance(val, list):
                return f"[{len(val)} items]"
    # Fall back to first string value
    for val in args.values():
        if isinstance(val, str):
            return val[:60]
    return ""


def print_tool_call(name: str, arguments_json: str) -> None:
    """Show which tool is being called and its key argument."""
    key_arg = _extract_key_arg(name, arguments_json)
    suffix = f" [dim]({key_arg})[/dim]" if key_arg else ""
    console.print(f"  [bold blue]> {name}[/bold blue]{suffix}")


def print_tool_result(name: str, result: str) -> None:
    """Show a brief preview of the tool result."""
    is_error = result.startswith(("blocked by", "Rejected", "Unknown tool", "Tool execution failed"))
    marker = "[red]x[/red]" if is_error else "[green]ok[/green]"
    # Collect first few meaningful lines for preview
    lines: list[str] = []
    for line in result.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip metadata lines like "exit_code=0"
        if stripped.startswith("exit_code="):
            if stripped != "exit_code=0":
                is_error = True
                marker = "[red]x[/red]"
            continue
        lines.append(stripped)
        if len(lines) >= 3:
            break
    preview = " | ".join(lines) if lines else "(empty)"
    if len(preview) > 120:
        preview = preview[:117] + "..."
    console.print(f"  {marker} [dim]{preview}[/dim]")


def print_thinking(text: str) -> None:
    """Show model reasoning/thinking in dim italic."""
    # Truncate very long thinking to keep terminal manageable
    lines = text.strip().splitlines()
    shown = [*lines[:3], "...", *lines[-2:]] if len(lines) > 6 else lines
    body = "\n".join(shown)
    console.print(f"[dim italic]{body}[/dim italic]")


def print_plan(plan: list[dict[str, str]]) -> None:
    """Show the task plan with status markers."""
    if not plan:
        return
    for item in plan:
        step = item.get("step", "")
        status = item.get("status", "")
        if status == "completed":
            console.print(f"  [green]~[/green] [strike dim]{step}[/strike dim]")
        elif status == "in_progress":
            console.print(f"  [cyan]>[/cyan] {step}")
        else:
            console.print(f"  [dim]-[/dim] [dim]{step}[/dim]")


# ── Slash command completer ──────────────────────────────────────────


class _SlashCompleter(Completer):
    """Auto-complete slash commands typed at the prompt."""

    def __init__(self, command_names: Sequence[str]) -> None:
        self._names = sorted(command_names)

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        text = document.text_before_cursor
        if document.text_after_cursor.strip():
            return
        # Only complete if the whole input so far starts with "/"
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return
        # Extract the token being typed (the part after the last space)
        last_space = text.rfind(" ")
        if last_space >= 0:
            return  # don't complete args, only the command name itself
        token = stripped[1:]  # remove leading "/"
        for name in self._names:
            if name.startswith(token):
                yield Completion(
                    text=f"/{name}",
                    start_position=-len(stripped),
                    display=f"/{name}",
                )


# ── PromptSession wrapper ────────────────────────────────────────────


class InputSession:
    """Interactive input session backed by prompt_toolkit.

    Features over raw ``input()``:
    - Full readline-style editing (cursor movement, Home/End, kill-line, etc.)
    - Up/Down arrow history across the session
    - Alt+Enter / Ctrl+J for multi-line input, Enter to submit
    - Tab-completion for slash commands
    """

    def __init__(self, command_names: Sequence[str] = ()) -> None:
        kb = KeyBindings()

        @kb.add("escape", "enter", eager=True)  # Alt+Enter
        @kb.add("c-j", eager=True)  # Ctrl+J
        def _newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        self._session: PromptSession[str] = PromptSession(
            message=FormattedText([("bold ansibrightcyan", "you> ")]),
            prompt_continuation=FormattedText([("ansigray", " ... ")]),
            completer=_SlashCompleter(command_names) if command_names else None,
            complete_while_typing=True,
            key_bindings=kb,
            history=InMemoryHistory(),
            multiline=False,  # Enter submits; Alt+Enter / Ctrl+J for newlines
        )

    def prompt(self) -> str:
        """Read one user input. Raises EOFError / KeyboardInterrupt."""
        with patch_stdout(raw=True):
            return self._session.prompt().strip()

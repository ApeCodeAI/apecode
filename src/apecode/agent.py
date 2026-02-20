"""Core loop for the nano code agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from apecode.tools import ToolRegistry


class ChatModel(Protocol):
    """Protocol for chat model adapters."""

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return one assistant message."""


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


@dataclass(slots=True)
class AgentConfig:
    """Runtime knobs for agent execution."""

    max_steps: int = 20


@dataclass(slots=True)
class AgentCallbacks:
    """Optional event callbacks. Agent stays framework-agnostic — no Rich import."""

    on_status: Callable[[str], None] | None = None
    """Called with status text ("Thinking...") or empty string to clear."""

    on_thinking: Callable[[str], None] | None = None
    """Called when the model returns reasoning_content."""

    on_tool_call: Callable[[str, str], None] | None = None
    """Called before tool execution with (tool_name, arguments_json)."""

    on_tool_result: Callable[[str, str], None] | None = None
    """Called after tool execution with (tool_name, result_text)."""


class NanoCodeAgent:
    """A tiny tool-calling loop with Chat Completions."""

    def __init__(
        self,
        *,
        model: ChatModel,
        tools: ToolRegistry,
        system_prompt: str,
        config: AgentConfig | None = None,
        callbacks: AgentCallbacks | None = None,
        # Legacy single callback kept for backwards compat with tests
        on_tool_call: Callable[[str, str], None] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or AgentConfig()
        self.cb = callbacks or AgentCallbacks(on_tool_call=on_tool_call)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def _fire(self, name: str, *args: Any) -> None:
        fn = getattr(self.cb, name, None)
        if fn is not None:
            fn(*args)

    def run(self, user_input: str) -> str:
        """Run one user turn to completion."""
        self.messages.append({"role": "user", "content": user_input})
        for _ in range(self.config.max_steps):
            self._fire("on_status", "Thinking...")
            assistant = self.model.complete(
                messages=self.messages,
                tools=self.tools.as_openai_tools(),
            )
            self._fire("on_status", "")

            # Show thinking if present
            reasoning = assistant.get("reasoning_content")
            if reasoning:
                self._fire("on_thinking", str(reasoning))

            tool_calls = assistant.get("tool_calls") or []
            assistant_record: dict[str, Any] = {
                "role": "assistant",
                "content": assistant.get("content"),
            }
            # Preserve provider-specific fields (e.g. reasoning_content for thinking models)
            for key in ("reasoning_content",):
                if assistant.get(key):
                    assistant_record[key] = assistant[key]
            if tool_calls:
                assistant_record["tool_calls"] = tool_calls
            self.messages.append(assistant_record)

            if not tool_calls:
                return _coerce_text(assistant.get("content"))

            for call in tool_calls:
                call_id = str(call.get("id", ""))
                function = call.get("function") or {}
                name = str(function.get("name", ""))
                arguments = str(function.get("arguments", "{}"))
                self._fire("on_tool_call", name, arguments)
                result = self.tools.execute(name, arguments)
                self._fire("on_tool_result", name, result)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                    }
                )

        raise RuntimeError(f"max steps exceeded ({self.config.max_steps})")

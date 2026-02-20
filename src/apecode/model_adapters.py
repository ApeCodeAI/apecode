"""Model adapter layer for the nano agent using official provider SDKs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


class ModelError(RuntimeError):
    """Raised when model calls fail."""


def _coerce_text_content(content: Any) -> str:
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


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            text = _coerce_text_content(message.get("content"))
            if text:
                system_parts.append(text)
            continue

        if role == "user":
            converted.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": _coerce_text_content(message.get("content"))}],
                }
            )
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _coerce_text_content(message.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                raw_arguments = function.get("arguments", "{}")
                try:
                    tool_input = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    tool_input = {"_raw_arguments": str(raw_arguments)}
                if not isinstance(tool_input, dict):
                    tool_input = {"value": tool_input}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id", "")),
                        "name": str(function.get("name", "")),
                        "input": tool_input,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            converted.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(message.get("tool_call_id", "")),
                            "content": _coerce_text_content(message.get("content")),
                        }
                    ],
                }
            )

    return "\n\n".join(system_parts), converted


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or {}
        converted.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


def _anthropic_message_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
            continue
        if block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )
    result: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _openai_message_to_dict(message: Any) -> dict[str, Any]:
    content = message.content if hasattr(message, "content") else ""
    result: dict[str, Any] = {"role": "assistant", "content": content or ""}
    # Preserve reasoning_content for thinking models (e.g. Kimi K2.5)
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content
    raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
    if raw_tool_calls:
        normalized_calls: list[dict[str, Any]] = []
        for item in raw_tool_calls:
            function = getattr(item, "function", None)
            normalized_calls.append(
                {
                    "id": str(getattr(item, "id", "")),
                    "type": "function",
                    "function": {
                        "name": str(getattr(function, "name", "")),
                        "arguments": str(getattr(function, "arguments", "{}")),
                    },
                }
            )
        result["tool_calls"] = normalized_calls
    return result


def _require_openai_sdk():
    try:
        from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
    except Exception as exc:  # pragma: no cover - depends on env
        raise ModelError("OpenAI SDK is required. Install dependency `openai` (e.g. `uv pip install openai`).") from exc
    return OpenAI, APIError, APIConnectionError, APITimeoutError


def _require_anthropic_sdk():
    try:
        from anthropic import Anthropic, APIConnectionError, APIError, APITimeoutError
    except Exception as exc:  # pragma: no cover - depends on env
        raise ModelError("Anthropic SDK is required. Install dependency `anthropic` (e.g. `uv pip install anthropic`).") from exc
    return Anthropic, APIError, APIConnectionError, APITimeoutError


@dataclass(slots=True)
class OpenAIChatCompletionsClient:
    """OpenAI-compatible chat client based on official OpenAI SDK."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 120
    temperature: float = 0.0
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        OpenAI, _api_error, _conn_error, _timeout_error = _require_openai_sdk()
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send one completion request and return one assistant message."""
        _OpenAI, APIError, APIConnectionError, APITimeoutError = _require_openai_sdk()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.temperature,
            )
        except APITimeoutError as exc:
            raise ModelError("Request timed out") from exc
        except APIConnectionError as exc:
            raise ModelError(f"Network error: {exc}") from exc
        except APIError as exc:
            raise ModelError(f"Provider error: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelError(f"Unexpected OpenAI SDK error: {exc}") from exc

        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError) as exc:
            raise ModelError(f"Unexpected model response: {response}") from exc
        return _openai_message_to_dict(message)


@dataclass(slots=True)
class AnthropicMessagesClient:
    """Anthropic Messages API adapter with OpenAI-like tool-call shape."""

    api_key: str
    model: str
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"
    timeout: int = 120
    max_tokens: int = 4096
    temperature: float = 0.0
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        Anthropic, _api_error, _conn_error, _timeout_error = _require_anthropic_sdk()
        self._client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            default_headers={"anthropic-version": self.api_version},
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt, anthropic_messages = _openai_messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": _openai_tools_to_anthropic(tools),
        }
        if system_prompt:
            payload["system"] = system_prompt

        _Anthropic, APIError, APIConnectionError, APITimeoutError = _require_anthropic_sdk()
        try:
            response = self._client.messages.create(**payload)
        except APITimeoutError as exc:
            raise ModelError("Request timed out") from exc
        except APIConnectionError as exc:
            raise ModelError(f"Network error: {exc}") from exc
        except APIError as exc:
            raise ModelError(f"Provider error: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelError(f"Unexpected Anthropic SDK error: {exc}") from exc

        body = response.model_dump(mode="json", exclude_none=True)
        if not isinstance(body, dict):
            raise ModelError(f"Unexpected model response: {response}")
        return _anthropic_message_to_openai(body)


@dataclass(slots=True)
class KimiChatCompletionsClient(OpenAIChatCompletionsClient):
    """Kimi OpenAI-compatible adapter."""


def create_model_client(*, provider: str, model: str, timeout: int, temperature: float | None = None):
    """Create model client by provider name."""
    normalized = provider.strip().lower()
    if normalized == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return OpenAIChatCompletionsClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature if temperature is not None else 0.0,
        )

    if normalized == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        api_version = os.environ.get("ANTHROPIC_API_VERSION", "2023-06-01")
        return AnthropicMessagesClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            api_version=api_version,
            timeout=timeout,
            temperature=temperature if temperature is not None else 0.0,
        )

    if normalized == "kimi":
        api_key = os.environ.get("KIMI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("KIMI_API_KEY is required for provider=kimi")
        base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
        return KimiChatCompletionsClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature if temperature is not None else 1.0,
        )

    raise RuntimeError(f"unsupported provider: {provider}")

"""Tests for provider adapter conversion helpers."""

from __future__ import annotations

import json

from apecode.model_adapters import (
    _anthropic_message_to_openai,
    _openai_messages_to_anthropic,
)


def test_openai_to_anthropic_conversion() -> None:
    messages = [
        {"role": "system", "content": "You are agent."},
        {"role": "user", "content": "Please inspect file"},
        {
            "role": "assistant",
            "content": "I will call a tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "line content"},
    ]
    system_prompt, converted = _openai_messages_to_anthropic(messages)
    assert system_prompt == "You are agent."
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    tool_use = converted[1]["content"][1]
    assert tool_use["type"] == "tool_use"
    assert tool_use["name"] == "read_file"
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0]["type"] == "tool_result"


def test_anthropic_to_openai_conversion() -> None:
    message = {
        "content": [
            {"type": "text", "text": "Working on it."},
            {"type": "tool_use", "id": "abc", "name": "list_files", "input": {"path": "."}},
        ]
    }
    converted = _anthropic_message_to_openai(message)
    assert converted["role"] == "assistant"
    assert converted["content"] == "Working on it."
    assert converted["tool_calls"][0]["id"] == "abc"
    assert converted["tool_calls"][0]["function"]["name"] == "list_files"
